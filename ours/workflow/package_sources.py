#!/usr/bin/env python3
"""Collect build/refiner/experiment source only; no datasets, results or binaries."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
SUFFIXES = {".py", ".sh", ".cpp", ".cc", ".c", ".h", ".hpp", ".hh", ".cxx", ".cuh",
            ".cmake", ".md", ".txt", ".json", ".yml", ".yaml", ".in", ".toml", ".ini", ".patch"}
BLOCK_DIRS = {".git", ".codex", ".agents", ".aws", "__pycache__", ".pytest_cache", "data", "datasets",
              "outputs", "results", "logs", "build", "dist", ".runtime", "node_modules", ".venv", "venv"}
NAMES = {"LICENSE", "LICENSE-MIT", "LICENSE-APACHE", "COPYING", "NOTICE", "Makefile", ".gitmodules"}
EXCLUDED_ARTIFACTS = {"ours/test/t2i-10M/validation.json",
                      "thirdparty/DiskANN/gperftools/docs/t-test1.times.txt"}


def eligible(rel):
    return (not any(p in BLOCK_DIRS or p.startswith("build-") or p.endswith(".egg-info") for p in rel.parts[:-1])
            and str(rel) not in EXCLUDED_ARTIFACTS
            and not rel.name.startswith("cmake_test_discovery_")
            and (rel.suffix in SUFFIXES or rel.name in NAMES
                 or ("Eigen" in rel.parts and not rel.suffix)))


def collect_tree(root, folder, mapping, prefix=Path()):
    origin = root / folder
    if not origin.exists():
        raise ValueError(f"Missing required source tree: {origin}")
    for current, dirs, files in os.walk(origin, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in BLOCK_DIRS and not d.startswith("build-") and not d.endswith(".egg-info"))
        for name in sorted(files):
            source = Path(current) / name
            rel = source.relative_to(root)
            if eligible(rel):
                if source.is_symlink():
                    raise ValueError(f"Resolve source symlink explicitly before packaging: {source}")
                mapping[prefix / rel] = source


def add_file(root, rel, mapping, prefix=Path()):
    source = root / rel
    if not source.is_file():
        raise ValueError(f"Missing required source file: {source}")
    mapping[prefix / rel] = source


def python_closure(root, mapping):
    """Include local imports and literal dynamic .py dependencies, not all prototypes."""
    folders = ("ours/python", "ours/test/t2i-1M/python", "ours/test/t2i-1B", "ours/scripts")
    by_name = {}
    for folder in folders:
        for source in (root / folder).glob("*.py"):
            by_name.setdefault(source.stem, []).append(source)
    examined = set()
    while True:
        pending = [p for key, p in mapping.items() if not str(key).startswith("build-source/")
                   and p.suffix == ".py" and p not in examined]
        if not pending:
            return
        for source in pending:
            examined.add(source)
            tree = ast.parse(source.read_text(), filename=str(source))
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(v.name.split(".")[0] for v in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module.split(".")[0])
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    modules.update(Path(v).stem for v in re.findall(r"[A-Za-z_][A-Za-z_0-9-]*\.py", node.value))
            for name in modules:
                for dependency in by_name.get(name, []):
                    mapping[dependency.relative_to(root)] = dependency


def source_map(remora, roargraph):
    mapping = {}
    for folder in ("include", "src", "tests", "thirdparty", "ours/include", "ours/src", "ours/tests",
                   "ours/experimental/traversal-fast-v1", "ours/experimental/relative-rate-v1",
                   "ours/test/t2i-10M", "ours/test/webvid", "ours/test/rs5m", "ours/test/wordvector-id",
                   "ours/test/t2i-1M-native100", "ours/transfer/cmake"):
        collect_tree(remora, folder, mapping)
    for rel in ("CMakeLists.txt", "LICENSE", "ours/CMakeLists.txt", "ours/test/t2i-1B/pipeline.py",
                "ours/test/t2i-1B/protocol.py", "ours/test/t2i-1B/config.example.json",
                "ours/test/t2i-1M/cpp/exp234_audit.cpp", "ours/test/t2i-1M/cpp/raw_score_native_parity.cpp",
                "ours/test/t2i-1M/cpp/terminal_rate_audit.cpp"):
        add_file(remora, rel, mapping)
    for stem in ("run_frozen_ablation", "run_frozen_experiment5", "run_frozen_experiments34",
                 "run_native100_capped_experiment1", "run_native100_experiment1",
                 "run_t2i10m_experiment5", "run_t2i10m_ablation", "run_t2i10m_experiments34",
                 "run_t2i1m_experiments345", "analyze_t2i10m_experiment1", "plot_t2i10m_memory_breakdown"):
        add_file(remora, f"ours/scripts/{stem}.py", mapping)
    # Primary training, matrix IO, export and parity dependencies used by the
    # graph pipeline's dynamic self.python(name) calls must remain available.
    for name in ("train_generic_primary.py", "audit_primary_scorer_parity.py",
                 "create_primary_quantizer_manifest.py", "build_cold_vector_store.py"):
        add_file(remora, "ours/python/" + name, mapping)
    python_closure(remora, mapping)
    # Latest code organization and Top-100-aware audit live in the workspace.
    collect_tree(roargraph, "ours/workflow", mapping)
    add_file(roargraph, "ours/test/wordvector-id/run_experiments34.sh", mapping)

    # Preserve the newer construction implementation separately from the frozen
    # experimental core. The shared thirdparty source is linked INSIDE the bundle.
    prefix = Path("build-source")
    for folder in ("include", "src", "tests", "ours/include", "ours/src", "ours/tests", "ours/transfer/cmake"):
        collect_tree(roargraph, folder, mapping, prefix)
    for rel in ("CMakeLists.txt", "LICENSE", "ours/CMakeLists.txt"):
        add_file(roargraph, rel, mapping, prefix)
    # Workspace CMake declares the same older audit/test sources.
    for p in (roargraph / "ours/test/t2i-1M/cpp").glob("*.cpp"):
        mapping[prefix / p.relative_to(roargraph)] = p
    for name in ("laion_opq_full.py", "laion_opq_build.py", "run_laion_opq_full.sh", "run_laion_opq_build.sh"):
        add_file(roargraph, "ours/scripts/" + name, mapping, prefix)
    # Offline reader bridge can be rebuilt against the frozen experiment adapter.
    add_file(roargraph, "ours/src/native_primary_reader_main.cpp", mapping)
    mapping[Path("ours/CMakeLists.txt")] = HERE / "code_bundle/formal-CMakeLists.txt"
    mapping[Path("README.md")] = HERE / "code_bundle/README.md"
    mapping[Path("requirements.txt")] = HERE / "code_bundle/requirements.txt"
    for name in ('build_graph.py', 'remora.py'):
        mapping[Path(name)] = HERE / 'code_bundle' / name
    # The organized trainer's compatibility shim imports this package. Keep
    # the full package and its public docs when exporting an organized tree.
    if (remora / 'ours/method').is_dir():
        collect_tree(remora, 'ours/method', mapping)
        add_file(remora, 'method.py', mapping)
        add_file(remora, 'README.md', mapping)
        if (remora / 'docs').is_dir():
            collect_tree(remora, 'docs', mapping)
    return mapping


def verify(destination, records):
    for r in records:
        p = destination / r["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f"Copy checksum mismatch: {p}")
        if p.suffix == ".py":
            ast.parse(p.read_text(), filename=str(p))
    # Confirm statically declared internal CMake source files are present.
    for project in (destination, destination / "build-source"):
        cmake = project / "ours/CMakeLists.txt"
        content = cmake.read_text()
        candidates = re.findall(r"(?<![A-Za-z0-9_/])(?:src|tests|test)/[A-Za-z0-9_./-]+\.(?:cpp|cc)", content)
        for name in candidates:
            if not (project / "ours" / name).is_file():
                raise ValueError(f"CMake source missing from bundle: {project / 'ours' / name}")
    for p in destination.rglob("*"):
        if p.is_file() and p.suffix in {".bin", ".npy", ".npz", ".fbin", ".ibin", ".svg", ".log", ".a", ".so", ".o", ".pyc", ".tar", ".gz"}:
            raise ValueError(f"Non-source artifact in bundle: {p}")


def scoped_workflow_map(destination, mapping, previous):
    """Refresh only the public interface/docs, preserving other frozen sources.

    Return destinations as their own source for untouched files; the copy loop
    skips them and retains their original provenance in SOURCE_MANIFEST.csv.
    """
    result={Path(row['path']): destination/row['path'] for row in previous}
    for rel, source in mapping.items():
        if rel.parts[:2]==('ours','workflow') or str(rel) in ('README.md','build_graph.py','remora.py'):
            result[rel]=source
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remora", type=Path, default=Path("/home/xzl/REMORA"))
    parser.add_argument("--roargraph", type=Path, default=Path("/home/xzl/RoarGraph"))
    parser.add_argument("--destination", type=Path, default=Path("/home/xzl/REMORA-code"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh an untouched bundle previously created by this collector")
    parser.add_argument("--refresh-workflow-only", action="store_true",
                        help="With --refresh, update interface/docs only; preserve other frozen sources")
    args = parser.parse_args()
    mapping = source_map(args.remora.resolve(strict=True), args.roargraph.resolve(strict=True))
    if args.refresh_workflow_only:
        if not args.refresh:
            raise ValueError('--refresh-workflow-only requires --refresh')
        manifest=args.destination/'SOURCE_MANIFEST.csv'
        if not manifest.is_file():
            raise ValueError('Scoped refresh needs the previous source manifest')
        mapping=scoped_workflow_map(args.destination.resolve(),mapping,list(csv.DictReader(manifest.open(newline=''))))
    total = sum(p.stat().st_size for p in mapping.values())
    print(json.dumps(dict(destination=str(args.destination), files=len(mapping), source_bytes=total,
                          includes_data=False, includes_results=False, includes_binaries=False), indent=2))
    if not args.execute:
        return
    destination = args.destination.resolve()
    if destination.exists() and not args.refresh:
        raise ValueError("Destination already exists; choose a new directory, never overwrite")
    if args.refresh:
        manifest = destination / "SOURCE_MANIFEST.csv"
        summary = destination / "SOURCE_SUMMARY.json"
        if not manifest.is_file() or not summary.is_file():
            raise ValueError("Refresh requires this collector's source manifest and summary")
        previous = list(csv.DictReader(manifest.open(newline="")))
        if json.loads(summary.read_text()).get("status") != "COPIED_AND_STATICALLY_VERIFIED":
            raise ValueError("Refusing to refresh an unverified directory")
        verify(destination, previous)
        previous_paths = {r["path"] for r in previous}
        obsolete = previous_paths - {str(p) for p in mapping}
        if any(not Path(p).name.startswith("cmake_test_discovery_") and p not in EXCLUDED_ARTIFACTS
               for p in obsolete):
            raise ValueError("Refresh cannot remove source files")
        for rel in mapping:
            if (destination / rel).exists() and str(rel) not in previous_paths:
                raise ValueError(f"Refusing to overwrite an untracked file: {rel}")
        # Only discard explicitly excluded generated validation/timing metadata;
        # every exact path/hash was verified against our previous manifest above.
        for rel in sorted(obsolete):
            (destination / rel).unlink()
    else:
        destination.mkdir(parents=False)
    records = []
    for rel, source in sorted(mapping.items()):
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        preserved=source.resolve()==target.resolve()
        if not preserved:
            shutil.copy2(source, target)
        provenance=next(r['source'] for r in previous if r['path']==str(rel)) if preserved else str(source)
        records.append(dict(path=str(rel), source=provenance, bytes=target.stat().st_size,
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    thirdparty_link = destination / "build-source/thirdparty"
    if thirdparty_link.is_symlink():
        if os.readlink(thirdparty_link) != "../thirdparty":
            raise ValueError("Bundle dependency link was modified")
    else:
        thirdparty_link.symlink_to("../thirdparty", target_is_directory=True)
    with (destination / "SOURCE_MANIFEST.csv").open("w" if args.refresh else "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "source", "bytes", "sha256"))
        writer.writeheader(); writer.writerows(records)
    verify(destination, records)
    with (destination / "SOURCE_SUMMARY.json").open("w" if args.refresh else "x") as stream:
        json.dump(dict(status="COPIED_AND_STATICALLY_VERIFIED", files=len(records), source_bytes=total,
                       data_included=False, results_included=False, binaries_included=False,
                       frozen_experiment_source=str(args.remora), newer_build_source=str(args.roargraph),
                       python_syntax_checked=True, internal_cmake_sources_checked=True,
                       original_directories_unchanged=True), stream, indent=2)
        stream.write("\n")
    print(f"[complete] {destination}; source manifest/checksums verified")


if __name__ == "__main__":
    main()
