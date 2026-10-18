import argparse
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


workflow = module("workflow_frontdoor", "run.py")
builder = module("workflow_builder", "build.py")
rate = module("workflow_accuracy_rate", "adapters/run_accuracy_rate.py")
packager = module("workflow_source_packager", "package_sources.py")


class SourcePackagingTests(unittest.TestCase):
    def test_scoped_refresh_preserves_other_source(self):
        old=[dict(path='ours/src/a.cpp',source='/old/a.cpp'),dict(path='README.md',source='/old/README.md')]
        mapping={Path('README.md'):Path('/new/README.md'),Path('ours/workflow/interface.py'):Path('/new/interface.py'),
                 Path('ours/src/a.cpp'):Path('/new/unrelated.cpp'),Path('ours/test/new.py'):Path('/new/unrelated.py')}
        result=packager.scoped_workflow_map(Path('/bundle'),mapping,old)
        self.assertEqual(result[Path('ours/src/a.cpp')],Path('/bundle/ours/src/a.cpp'))
        self.assertNotIn(Path('ours/test/new.py'),result)
        self.assertEqual(result[Path('ours/workflow/interface.py')],Path('/new/interface.py'))

    def test_build_template_is_source_not_build_artifact(self):
        self.assertTrue(packager.eligible(Path("ours/workflow/configs/build-native.example.json")))
        self.assertFalse(packager.eligible(Path("ours/build-release/generated.cpp")))

    def test_eigen_extensionless_headers_are_required_source(self):
        self.assertTrue(packager.eligible(Path("thirdparty/RaBitQ-Library/include/rabitqlib/third/Eigen/Dense")))
        self.assertFalse(packager.eligible(Path("thirdparty/SAQ/cmake_test_discovery_123.json")))

    def test_data_results_and_binaries_are_not_sources(self):
        for name in ("datasets/config.json", "outputs/report.json", "ours/build/search",
                     "ours/src/scorer.o", "ours/src/model.bin", "ours/src/cache.pyc",
                     "ours/test/t2i-10M/validation.json"):
            self.assertFalse(packager.eligible(Path(name)), name)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.catalog = json.loads((HERE / "catalog.json").read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def jobs(self, dataset="t2i-10M", stages=("exp1",)):
        return workflow.plan(self.repo, self.catalog, dataset, stages, "/python")

    def test_plan_has_no_filesystem_writes(self):
        self.jobs(stages=("prepare", "exp1", "exp2", "exp3", "exp4", "exp5"))
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_exp34_is_one_accuracy_job(self):
        jobs = self.jobs(stages=("exp3", "exp4"))
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].stage, "exp34")
        self.assertEqual(jobs[0].command[-1], "inversion,traversal,report")
        self.assertNotIn("--repeats", jobs[0].command)

    def test_exp1_own_queue_disk_rerank_cap(self):
        cmd = self.jobs()[0].command
        self.assertIn("run_native100_capped_experiment1.py", cmd[2])
        self.assertEqual(cmd[-2:], ["--max-rerank-depth", "200"])

    def test_relative_rate_is_not_fixed64(self):
        cmd = self.jobs("webvid", ("exp5",))[0].command
        self.assertEqual(cmd[cmd.index("--budgets") + 1], "80,160,240,320")
        self.assertIn("ours/experimental/relative-rate-v1/train_remora.py", cmd)
        self.assertNotIn("--repeats", cmd)

    def test_rate_training_never_launches_timing_phase(self):
        a = argparse.Namespace(repo="/repo", dataset="T", source="s", prepared="p",
                               audit_source="a", output="o", budgets="32,64", beams="100,200",
                               queries=10000, gpu=2, trainer="", build_dir="")
        cmd = rate.training_command(a)
        self.assertEqual(cmd[cmd.index("--phases") + 1], "train,quality")
        self.assertEqual(cmd[cmd.index("--repeats") + 1], "1")

    def test_wordvector_truth_depth_and_blocked_old_audits(self):
        audit = self.jobs("wordvector-id", ("exp34",))[0]
        self.assertEqual(audit.command[audit.command.index("--inversion-ranks") + 1], "50,75,100")
        for job in self.jobs("wordvector-id", ("exp2", "exp5")):
            self.assertTrue(any(note.startswith("BLOCKED:") for note in job.notes))

    def test_rs5m_uses_top200_reference_for_accuracy_not_timing(self):
        audit = self.jobs("rs5m", ("exp34",))[0].command
        perf = self.jobs("rs5m", ("exp1",))[0].command
        self.assertTrue(audit[audit.index("--source") + 1].endswith("/reference"))
        self.assertIn("performance.config.json", perf[perf.index("--config") + 1])

    def test_frozen_output_overlap_rejected(self):
        frozen = self.catalog["datasets"]["t2i-10M"]["prepared"]
        for path in (frozen, frozen + "/new", "outputs/t2i-10M"):
            with self.assertRaisesRegex(ValueError, "separate"):
                workflow.plan(self.repo, self.catalog, "t2i-10M", ["exp34"], "/python", path)

    def test_terminal_config_rejected(self):
        spec = self.catalog["datasets"]["t2i-10M"]
        c = dict(dataset="t2i-10M", base_dim=200, side_bits=64, scoring_placement="terminal")
        with self.assertRaisesRegex(ValueError, "scoring_placement"):
            workflow.contract(c, spec)

    def test_inventory_excludes_outputs_and_cache(self):
        for name in ("ours/scripts/a.py", "outputs/never.py", "ours/scripts/__pycache__/never.py"):
            p = self.repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("pass\n")
        output = io.StringIO()
        workflow.inventory(self.repo, output)
        self.assertIn("ours/scripts/a.py", output.getvalue())
        self.assertNotIn("never.py", output.getvalue())

    def test_corrupt_or_missing_prepared_not_reused(self):
        spec = self.catalog["datasets"]["t2i-10M"]
        self.assertFalse(workflow.prepared_complete(self.repo, spec, ["opq-pq"]))

    def test_valid_prepared_headers_allow_reuse(self):
        spec = dict(dataset="test", dim=2, side_bits=8)
        (self.repo / "configuration.json").write_text(json.dumps(dict(config=dict(dataset="test", base_dim=2, side_bits=8, base_rows=3))))
        d = self.repo / "ji/opq-pq/b8/remora"
        d.mkdir(parents=True)
        (d / "side.bin").write_bytes(struct.pack("<8sIQIIIQQ", b"JIRCODE1", 3, 3, 2, 8, 1, 11, 12) + b"\0" * 3)
        (d / "online.bin").write_bytes(struct.pack("<8sIIIIQQQQ", b"JIONL002", 2, 2, 1, 8, 3, 11, 12, 256))
        (d / "training.json").write_text("{}")
        self.assertTrue(workflow.prepared_complete(self.repo, spec, ["opq-pq"]))


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.c = json.loads((HERE / "configs/build-native.example.json").read_text())
        self.c.update(queries="queries.fbin", knn="knn.ibin", graph="new/graph", summary="new/summary.json",
                      primary_model="model.bin", primary_codes="codes.bin", builder="builder", query_limit=2, topk=2)

    def tearDown(self):
        self.tmp.cleanup()

    def inputs(self):
        (self.repo / "builder").write_bytes(b"placeholder")
        for name in ("queries.fbin", "knn.ibin"):
            (self.repo / name).write_bytes(struct.pack("<II", 2, 2) + b"\0" * 16)
        (self.repo / "model.bin").write_bytes(struct.pack("<8sIIQIIIIIIQ", b"GPRIM01\0", 1, 4, 3, 2, 2, 1, 256, 1, 0, 99))
        (self.repo / "codes.bin").write_bytes(struct.pack("<8sIIQIIQ", b"GPCOD01\0", 1, 40, 3, 1, 0, 99) + b"\0" * 3)

    def test_builder_disables_ji_and_checks_parity(self):
        validation, cmd = builder.commands(self.repo, self.c)
        self.assertIn("--disable-ji", cmd)
        self.assertEqual(validation[-1], "--validate-only")
        self.assertNotIn("--force-knn", cmd)

    def test_test_truth_cannot_silently_be_primary_knn(self):
        self.c["knn_geometry"] = "fp32"
        with self.assertRaisesRegex(ValueError, "geometry"):
            builder.commands(self.repo, self.c)

    def test_native_header_pair_validates(self):
        self.inputs()
        builder.validate(self.repo, self.c)

    def test_imported_format_rejected_before_native_execution(self):
        self.inputs()
        (self.repo / "model.bin").write_bytes(b"OPQPQ002" + b"\0" * 64)
        with self.assertRaisesRegex(ValueError, "not interchangeable"):
            builder.validate(self.repo, self.c)

    def test_dimension_truncation_rejected(self):
        self.inputs()
        (self.repo / "model.bin").write_bytes(struct.pack("<8sIIQIIIIIIQ", b"GPRIM01\0", 1, 4, 3, 2, 1, 1, 256, 1, 0, 99))
        with self.assertRaisesRegex(ValueError, "preserve raw dimension"):
            builder.validate(self.repo, self.c)

    def test_missing_knn_never_falls_back_to_cpu_scan(self):
        self.inputs()
        (self.repo / "knn.ibin").unlink()
        with self.assertRaises(FileNotFoundError):
            builder.validate(self.repo, self.c)


if __name__ == "__main__":
    unittest.main()
