"""Collect offline supervision on an existing Primary graph.

This stage has no dependency on test queries, test GT, or experiment catalogs.
"""
from pathlib import Path
import subprocess


def pool_commands(config):
    binary = Path(config["build_dir"])
    kind = config["method"]
    model = Path(config["primary_model"])
    if model.is_file():
        with model.open("rb") as stream:
            if stream.read(8) in (b"OPQPQ001", b"ERBQRG01", b"SAQRG001", b"LVOOD001"):
                kind = "qvrr"
    native = ["--primary-kind", kind, "--primary-model", config["primary_model"],
              "--primary-codes", config["primary_codes"], "--disable-ji"]
    if config["method"] == "extended-rabitq":
        native += ["--rabitq-config", config["rabitq_config"]]
    jobs = []
    for split, query_key, count_key in (("train", "alignment_queries", "pool_train_queries"),
                                       ("validation", "validation_queries", "pool_validation_queries")):
        dest = Path(config["output_root"]) / "pools" / split
        for beam in config["pool_beams"]:
            prefix = dest / f"b{beam}"
            jobs.append(dict(split=split, beam=beam, prefix=str(prefix), queries=config.get(query_key),
                trace=[str(binary / "roar_primary_ji_search"), *native,
                       "--graph", config.get("graph", "<set graph in config>"),
                       "--queries", config.get(query_key, "<set validation_queries in config>"),
                       "--query-limit", str(config[count_key]), "--beam", str(beam),
                       "--threads", str(config["build_threads"]), "--out-prefix", str(prefix),
                       "--retained-csr", str(prefix) + ".retained.csr", "--packed-graph"],
                audit=[str(binary / "roar_exp234_audit"), *native,
                       "--graph", config.get("graph", "<set graph in config>"),
                       "--queries", config.get(query_key, "<set validation_queries in config>"),
                       "--out-prefix", str(prefix) + ".audit", "--audit-ids", str(prefix) + ".audit.ids.ibin"]))
    return jobs


def prepare_pools(config):
    import numpy as np
    from collect_pools import read_trace, save_pools
    from protocol import write_matrix
    destination = Path(config["output_root"]) / "pools"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Candidate pool directory is not empty: {destination}; choose a new output_root")
    for key in ("train_pool", "validation_pool"):
        if Path(config[key]).exists():
            raise ValueError(f"Candidate pool output already exists: {config[key]}")
    jobs = pool_commands(config)
    for job in jobs:
        for name in ("trace", "audit"):
            if not Path(job[name][0]).is_file():
                raise ValueError(f"Missing native program: {job[name][0]}; compile the search/audit targets")
    traces = {"train": [], "validation": []}
    for job in jobs:
        prefix = Path(job["prefix"])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for stage in ("trace", "audit"):
            print(f"[pools] {job['split']} beam={job['beam']} {stage}", flush=True)
            with Path(str(prefix) + f".{stage}.log").open("w") as log:
                subprocess.run(job[stage], stdout=log, stderr=subprocess.STDOUT, check=True)
            if stage == "trace":
                offsets, nodes = read_trace(str(prefix) + ".retained.csr")
                ids = np.empty((len(offsets)-1, int(np.max(np.diff(offsets)))), np.uint32)
                for row in range(len(ids)):
                    lo, hi = offsets[row:row+2]
                    ids[row] = nodes[lo]
                    ids[row, :hi-lo] = nodes[lo:hi]
                write_matrix(str(prefix) + ".audit.ids.ibin", ids, "<u4")
        traces[job["split"]].append((job["beam"], str(prefix) + ".retained.csr",
                                     str(prefix) + ".audit.scores.fbin"))
    for split, queries, output in (("train", config["alignment_queries"], config["train_pool"]),
                                   ("validation", config["validation_queries"], config["validation_pool"])):
        if Path(output).exists():
            raise ValueError(f"Candidate pool output already exists: {output}")
        save_pools(config["base"], queries, traces[split], output)
