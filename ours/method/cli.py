"""Public commands for the method, without the experiment catalog."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import ROOT

METHODS = ("opq-pq", "qaq", "leanvec-ood", "extended-rabitq", "saq")
VARIANTS = ("euclidean", "alignment", "balanced", "uncentered-fixed",
            "centered-fixed", "centered", "remora")
DEFAULTS = dict(bits=64, variant="remora", device="cpu", kmeans_device="auto",
                geometric_encoder="faiss", geometric_encoder_device="auto",
                trainer_threads=8, seed=20261001, residual_sample_size=100000,
                ji_iterations=8, ji_batch_size=8192, encode_batch_size=65536,
                encoding_audit_size=1024, centered_epochs=4, recode_nodes=300000,
                build_dir="ours/build-workflow-v1", pool_beams=[50, 100, 200],
                pool_train_queries=4096, pool_validation_queries=1024,
                build_threads=8, rabitq_config="exact")
PATHS = ("base", "primary_model", "primary_codes", "graph", "alignment_queries",
         "validation_queries", "train_pool", "validation_pool", "output_root", "build_dir")


def absolute(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path, args):
    supplied = json.loads(Path(path).read_text())
    unknown = set(supplied) - set(DEFAULTS) - set(PATHS) - {"method"}
    if unknown:
        raise ValueError(f"Unknown method configuration keys: {sorted(unknown)}")
    config = dict(DEFAULTS, **supplied)
    for key in ("method", "bits", "variant", "device"):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    for key in ("method", "base", "primary_model", "primary_codes", "alignment_queries", "output_root"):
        if not config.get(key):
            raise ValueError(f"Missing method configuration key: {key}")
    if config["method"] not in METHODS or config["variant"] not in VARIANTS:
        raise ValueError("Unsupported primary method or REMORA variant")
    bits = config["bits"]
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 < bits <= 256 or bits % 8:
        raise ValueError("Side bits must be a multiple of 8 in [8, 256]")
    for key in ("trainer_threads", "seed", "residual_sample_size", "ji_iterations", "ji_batch_size",
                "encode_batch_size", "encoding_audit_size", "centered_epochs", "recode_nodes",
                "pool_train_queries", "pool_validation_queries", "build_threads"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be a {'nonnegative' if key == 'seed' else 'positive'} integer")
    if config["residual_sample_size"] < 256:
        raise ValueError("residual_sample_size must be >=256 (256 codewords per group)")
    for key in ("device", "kmeans_device", "geometric_encoder_device"):
        value = config[key]
        if key != "device" and value == "auto":
            continue
        if not isinstance(value, str) or (value != "cpu" and not (
                value.startswith("cuda:") and value[5:].isdigit())):
            raise ValueError(f"Invalid {key}: {value}")
    if config["geometric_encoder"] not in ("faiss", "torch"):
        raise ValueError("geometric_encoder must be faiss or torch")
    if not config["pool_beams"] or any(isinstance(b, bool) or not isinstance(b, int) or b < 10
                                        for b in config["pool_beams"]):
        raise ValueError("pool_beams must contain integer beams >=10")
    for key in PATHS:
        if config.get(key):
            config[key] = str(absolute(config[key]))
    out = Path(config["output_root"])
    config.setdefault("train_pool", str(out / "pools/train/pools.npz"))
    config.setdefault("validation_pool", str(out / "pools/validation/pools.npz"))
    return config


def validate_inputs(config, stage):
    from protocol import matrix_header
    required = ["base", "primary_model", "primary_codes", "alignment_queries"]
    if stage != "alignment":
        required += ["validation_queries"]
    required += ["graph"] if stage == "prepare" else ["train_pool", "validation_pool"] if stage == "train" else []
    for key in required:
        if not config.get(key) or not Path(config[key]).is_file():
            raise ValueError(f"Missing {key}: {config.get(key, '(not configured)')}")
    nodes, dim = matrix_header(config["base"])
    if nodes < 256:
        raise ValueError("At least 256 base nodes are required by the product codebooks")
    for key in ("alignment_queries", "validation_queries"):
        if key in config and (stage != "alignment" or key == "alignment_queries"):
            count, qdim = matrix_header(config[key])
            if qdim != dim:
                raise ValueError(f"Dimension mismatch: {key}")
            if stage == "prepare":
                limit = config["pool_train_queries" if key == "alignment_queries" else "pool_validation_queries"]
                if limit > count:
                    raise ValueError(f"Requested {limit} pool queries but {key} only has {count}")
    if stage != "alignment" and config["alignment_queries"] == config["validation_queries"]:
        raise ValueError("Training and validation must use separate query files")


def main(argv=None):
    parser = argparse.ArgumentParser(description="REMORA: prepare candidate pools, run Alignment or full training")
    commands = parser.add_subparsers(dest="stage", required=True)
    for name, aliases, description in (
        ("prepare", [], "Collect Primary traversal candidate pools for train/validation"),
        ("alignment", ["align"], "Run the existing Alignment variant, including full encoding/export"),
        ("initialize", ["init"], "Only fit query-visible coordinates and initial product codebooks"),
        ("train", ["training"], "Adapt the refiner, select on validation, encode/export all nodes"),
    ):
        command = commands.add_parser(name, aliases=aliases, help=description, description=description)
        command.add_argument("--config", type=Path, required=True, help="Method JSON; relative data paths use repository root")
        command.add_argument("--method", choices=METHODS)
        command.add_argument("--bits", type=int)
        command.add_argument("--variant", choices=VARIANTS)
        command.add_argument("--device", help="cpu or process-visible cuda:N; Faiss auto backends follow it")
        if name != "prepare":
            command.add_argument("--output", type=Path, help="New checkpoint (.npz) or training directory")
        if name in ("train", "alignment"):
            command.add_argument("--alignment", type=Path, help="Load a matching checkpoint; if omitted, fit alignment first")
        command.add_argument("--execute", action="store_true", help="Run the stage; otherwise print the plan")
    args = parser.parse_args(argv)
    action = {"align": "alignment", "training": "train", "init": "initialize"}.get(args.stage, args.stage)
    if action == "alignment":
        if args.variant not in (None, "alignment"):
            raise ValueError("The alignment command runs --variant alignment; use train for other variants")
        args.variant = "alignment"
    stage = "alignment" if action == "initialize" else "train" if action == "alignment" else action
    config = load_config(args.config, args)
    root = Path(config["output_root"])
    kind = {"euclidean": "euclidean", "alignment": "ordered"}.get(config["variant"], "balanced")
    default = root / f"alignment-{kind}-b{config['bits']}.npz" if stage == "alignment" else root / config["variant"]
    output = absolute(args.output) if getattr(args, "output", None) else default
    plan = dict(command=action, stage=stage, method=config["method"], bits=config["bits"], variant=config["variant"],
                device=config["device"], output=str(root / "pools" if stage == "prepare" else output),
                inputs={key: config.get(key) for key in ("base", "primary_model", "primary_codes", "alignment_queries")})
    if stage != "alignment":
        plan["inputs"]["validation_queries"] = config.get("validation_queries")
    if stage == "prepare":
        from .pools import pool_commands
        plan["commands"] = pool_commands(config)
    if stage == "train":
        plan["inputs"].update(train_pool=config["train_pool"], validation_pool=config["validation_pool"],
                              alignment=str(absolute(args.alignment)) if args.alignment else "fit automatically")
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("Plan only; add --execute to run.")
        return 0
    validate_inputs(config, stage)
    # Native formats use the existing offline reader bridge for reconstruction.
    reader = Path(config["build_dir"]) / "roar_native_primary_reader"
    if reader.is_file():
        os.environ.setdefault("REMORA_NATIVE_PRIMARY_READER", str(reader))
    os.environ.setdefault("REMORA_PRIMARY_DECODE_THREADS", str(config["trainer_threads"]))
    if stage == "prepare":
        from .pools import prepare_pools
        prepare_pools(config)
    elif stage == "alignment":
        from .alignment import fit_alignment, save_alignment, signature
        from protocol import matrix
        from qvrr import Primary
        if output.exists():
            raise ValueError(f"Alignment checkpoint already exists: {output}; choose a new --output")
        metadata = signature(config, config["method"], config["primary_model"], config["primary_codes"],
                             config["alignment_queries"], config["bits"], config["variant"])
        primary = Primary(config["primary_model"], config["primary_codes"])
        state = fit_alignment(config, primary, matrix(config["base"]), matrix(config["alignment_queries"]),
                              config["bits"], config["variant"], config["method"])
        save_alignment(output, state, metadata)
        print(f"[done] alignment checkpoint: {output}", flush=True)
    else:
        if args.alignment and not absolute(args.alignment).is_file():
            raise ValueError(f"Missing alignment checkpoint: {absolute(args.alignment)}")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Training output is not empty: {output}; choose a new --output")
        from .training import train
        output.mkdir(parents=True, exist_ok=True)
        resolved = output / "trainer-config.json"
        resolved.write_text(json.dumps(config, indent=2) + "\n")
        train(resolved, config["method"], config["primary_model"], config["primary_codes"],
              config["train_pool"], config["validation_pool"], config["bits"], config["variant"],
              output, alignment_path=absolute(args.alignment) if args.alignment else None)
    return 0


def entry(argv=None):
    try:
        return main(argv)
    except (ValueError, RuntimeError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as error:
        print(f"REMORA method: {error}", file=sys.stderr)
        return 2
