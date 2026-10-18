"""Exercise public commands, checkpoint reuse and candidate supervision."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ours.method import cli
from ours.method.training import load_pools
from protocol import matrix, write_matrix
from qvrr import Primary
from test_integration import fixture
import numpy as np


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model, self.codes, self.base, self.queries = fixture(self.root)
        primary = Primary(self.model, self.codes)
        rng = np.random.default_rng(73)
        for split, queries in (("train", self.queries[:12]), ("validation", self.queries[12:])):
            ids = np.stack([rng.choice(len(self.base), 20, replace=False) for _ in queries])
            exact = np.array([self.base[row] @ q for row, q in zip(ids, queries)])
            scores = np.array([primary.score(q, row) for row, q in zip(ids, queries)])
            np.savez(self.root / f"{split}.npz", query_ids=np.arange(len(queries)),
                     beams=np.full(len(queries), 20), offsets=np.arange(len(queries)+1)*20,
                     node_ids=ids.ravel(), targets=(exact-scores).ravel(),
                     exact_scores=exact.ravel(), primary_scores=scores.ravel())
        self.config = dict(method="opq-pq", base=str(self.root / "base.fbin"),
                           primary_model=str(self.model), primary_codes=str(self.codes),
                           alignment_queries=str(self.root / "metric.fbin"),
                           validation_queries=str(self.root / "select.fbin"),
                           train_pool=str(self.root / "train.npz"),
                           validation_pool=str(self.root / "validation.npz"),
                           output_root=str(self.root / "output"), bits=16, device="cpu",
                           residual_sample_size=512, ji_iterations=1, trainer_threads=1,
                           centered_epochs=1, recode_nodes=20, encode_batch_size=256,
                           ji_batch_size=64, pool_train_queries=4, pool_validation_queries=2,
                           pool_beams=[10, 20])
        self.path = self.root / "method.json"
        self.save_config()

    def save_config(self):
        self.path.write_text(json.dumps(self.config))

    def run_cli(self, action, *options):
        with redirect_stdout(io.StringIO()):
            return cli.main([action, "--config", str(self.path), *map(str, options)])

    def test_plan_does_not_create_outputs(self):
        for action in ("prepare", "alignment", "train", "initialize"):
            self.assertEqual(self.run_cli(action), 0)
        self.assertFalse((self.root / "output").exists())

    def test_alignment_and_full_training_are_independent(self):
        for action, variant in (("alignment", "alignment"), ("train", "remora")):
            self.run_cli(action, "--execute")
            output = self.root / "output" / variant
            summary = json.loads((output / "training.json").read_text())
            self.assertEqual(summary["variant"], variant)
            self.assertEqual(summary["candidate_conditioned"], variant == "remora")
            self.assertEqual(summary["support_regularization"], variant == "remora")
            self.assertFalse(summary["test_queries_used"])
            self.assertEqual((output / "side.bin").stat().st_size, 48 + 512*2)
            model = np.load(output / "model.npz")
            codes = np.frombuffer((output / "side.bin").read_bytes(), np.uint8, offset=48).reshape(512, 2)
            reconstructed = np.concatenate([model["books"][g, codes[:, g]] for g in range(2)], axis=1)
            self.assertTrue(np.isfinite(reconstructed).all())

    def test_staged_and_direct_training_export_identical_bytes(self):
        for variant in ("alignment", "remora"):
            checkpoint = self.root / f"{variant}.npz"
            direct, staged = self.root / f"{variant}-direct", self.root / f"{variant}-staged"
            self.run_cli("initialize", "--variant", variant, "--output", checkpoint, "--execute")
            self.run_cli("train", "--variant", variant, "--output", direct, "--execute")
            self.run_cli("train", "--variant", variant, "--alignment", checkpoint,
                         "--output", staged, "--execute")
            for name in ("online.bin", "side.bin"):
                self.assertEqual((direct / name).read_bytes(), (staged / name).read_bytes())

    def test_initialization_needs_no_validation_graph_or_pools(self):
        for key in ("validation_queries", "train_pool", "validation_pool"):
            self.config.pop(key)
        self.save_config()
        self.run_cli("initialize", "--execute")
        self.assertTrue((self.root / "output/alignment-balanced-b16.npz").is_file())

    def test_changed_checkpoint_inputs_are_rejected(self):
        checkpoint = self.root / "alignment.npz"
        self.run_cli("initialize", "--output", checkpoint, "--execute")
        with (self.root / "metric.fbin").open("r+b") as stream:
            stream.seek(8)
            stream.write(struct.pack("<f", 9.7))
        with self.assertRaisesRegex(ValueError, "inputs/settings/layout changed"):
            self.run_cli("train", "--alignment", checkpoint, "--execute")

    def test_completed_outputs_are_preserved(self):
        self.run_cli("alignment", "--execute")
        before = (self.root / "output/alignment/side.bin").read_bytes()
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.run_cli("alignment", "--execute")
        self.assertEqual(before, (self.root / "output/alignment/side.bin").read_bytes())

    def test_missing_pools_fail_before_training(self):
        (self.root / "train.npz").unlink()
        with self.assertRaisesRegex(ValueError, "Missing train_pool"):
            self.run_cli("train", "--execute")
        self.assertFalse((self.root / "output").exists())

    def test_prepare_builds_labels_from_primary_traversal(self):
        binary = self.root / "bin"
        binary.mkdir()
        for name in ("roar_primary_ji_search", "roar_exp234_audit"):
            (binary / name).touch()
        graph = self.root / "fixture.graph"
        graph.touch()
        self.config.update(graph=str(graph), build_dir=str(binary))
        for key in ("train_pool", "validation_pool"):
            self.config.pop(key)
        self.save_config()
        primary = Primary(self.model, self.codes)

        def native(command, **kwargs):
            prefix = command[command.index("--out-prefix")+1]
            queries = matrix(command[command.index("--queries")+1])
            self.assertEqual(command[command.index("--primary-kind")+1], "qvrr")
            if Path(command[0]).name == "roar_primary_ji_search":
                count = int(command[command.index("--query-limit")+1])
                lengths = np.arange(count) % 2 + 3
                offsets = np.r_[0, np.cumsum(lengths)].astype("<u8")
                nodes = np.concatenate([np.arange(k) + q for q, k in enumerate(lengths)]).astype("<u4")
                Path(prefix+".retained.csr").write_bytes(
                    struct.pack("<8sIQIQ", b"DRBVST01", 1, 0, count, len(nodes)) + offsets.tobytes() + nodes.tobytes())
            else:
                ids = matrix(command[command.index("--audit-ids")+1], "<u4")
                scores = np.stack([primary.score(q, row) for q, row in zip(queries, ids)])
                write_matrix(prefix+".scores.fbin", scores)

        with patch("ours.method.pools.subprocess.run", side_effect=native) as run:
            self.run_cli("prepare", "--execute")
        self.assertEqual(run.call_count, 8)
        for split, queries, count in (("train", self.queries[:12], 4),
                                      ("validation", self.queries[12:], 2)):
            pools = load_pools(self.root / "output/pools" / split / "pools.npz")
            self.assertEqual(pools.pool_count, count*2)
            owners = np.repeat(pools.query_ids, np.diff(pools.offsets))
            exact = np.sum(self.base[pools.node_ids] * queries[owners], axis=1)
            np.testing.assert_allclose(pools.targets, exact-pools.primary_scores, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
