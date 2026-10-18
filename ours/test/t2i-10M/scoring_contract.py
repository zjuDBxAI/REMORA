"""Fail-closed scoring contract shared by all five T2I-10M experiments.

Primary/FP32 are reference branches. Every residual-code branch must refine
scores DURING graph search. Terminal-only results are not interchangeable.
"""


def require_traversal(config):
    if config.get("scoring_placement") != "traversal":
        raise ValueError(
            "All REMORA experiment branches require scoring_placement=traversal; "
            "terminal-only configuration is not permitted in this suite"
        )


def validate_native_summary(summary, variant):
    expected = variant not in ("primary", "fp32")
    if summary.get("ji_traversal_enabled") is not expected:
        raise ValueError(f"Native traversal scoring mismatch for {variant}")
    if summary.get("terminal_ji", {}).get("enabled") is not False:
        raise ValueError("Terminal-only correction is not permitted")
    if expected and summary.get("ji_query_space") != "raw":
        raise ValueError("REMORA must correct the complete raw-space score residual")
    if "primary-traversal-terminal-ji" in summary.get("mode", ""):
        raise ValueError("Terminal-only measurement cannot enter the traversal report")


def validate_command(command, variant):
    values = list(map(str, command))
    expected = variant not in ("primary", "fp32")
    if "--terminal-ji" in values:
        raise ValueError("Refusing to launch terminal-only search")
    if expected:
        for option in ("--ji-model", "--ji-side-codes", "--ji-query-space"):
            if option not in values:
                raise ValueError(f"Missing traversal refinement argument {option}")
        if values[values.index("--ji-query-space") + 1] != "raw" or "--disable-ji" in values:
            raise ValueError("Wrong REMORA traversal decoder settings")
    elif "--disable-ji" not in values:
        raise ValueError("Reference branch must disable residual correction")
