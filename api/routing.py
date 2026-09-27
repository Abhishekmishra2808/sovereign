"""Transparent structural routing heuristic, not a claim of measured speedup."""
import json
import os


def summarize(text, model_format):
    if model_format == "json":
        model = json.loads(text)
        variables = model.get("variables", [])
        constraints = model.get("constraints", [])
        return {"columns": len(variables), "rows": len(constraints),
                "nonzeros": sum(len(c.get("linear", {})) for c in constraints),
                "integer_variables": sum(v.get("type") in ("integer", "binary") for v in variables),
                "problem_type": model.get("problem_type", "LP"), "estimated": False}
    # Count structural entries without doing numerical optimization on Render.
    section, rows, columns, nonzeros, integers, integer_mode = "", set(), set(), 0, set(), False
    for line in text.splitlines():
        fields = line.split()
        if not fields or line.lstrip().startswith("*"):
            continue
        if fields[0] in ("NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA", "QUADOBJ", "QMATRIX", "OBJSENSE", "OBJNAME") and (len(fields) == 1 or fields[0] == "NAME"):
            section = fields[0]
            continue
        if section == "ROWS" and len(fields) >= 2 and fields[0] != "N":
            rows.add(fields[1])
        elif section == "COLUMNS":
            if any("MARKER" in f for f in fields):
                integer_mode = any("INTORG" in f for f in fields)
                continue
            if len(fields) >= 3:
                columns.add(fields[0])
                if integer_mode:
                    integers.add(fields[0])
                nonzeros += sum(fields[i] in rows for i in range(1, len(fields) - 1, 2))
        elif section == "BOUNDS" and len(fields) >= 3 and fields[0] in ("BV", "LI", "UI"):
            integers.add(fields[2])
    return {"columns": len(columns), "rows": len(rows), "nonzeros": nonzeros,
            "integer_variables": len(integers), "problem_type": "MILP" if integers else "LP", "estimated": True}


def route_request(request):
    shape = summarize(request["modelJson"], request["modelFormat"])
    gpu_eligible = request.get("algorithm", "auto") != "simplex"
    if shape["problem_type"].upper() == "QP":
        gpu_eligible = request.get("qpAlgorithm", "auto") != "frank_wolfe"
    large = shape["nonzeros"] >= int(os.environ.get("SOVEREIGN_GPU_MIN_NONZEROS", "50000")) or (
        shape["columns"] >= 2000 and shape["nonzeros"] >= 8000)
    requested = request.get("device", "auto")
    # Repeated whole-solver trials on 10k-40k-variable transport LPs have not
    # shown a CUDA speedup. Keep auto on CPU until a workload-specific crossover
    # is measured. The former structural heuristic can be enabled explicitly
    # for experiments without changing a user's manual CUDA selection.
    experimental_auto = os.environ.get("SOVEREIGN_GPU_AUTO_ENABLED", "0") == "1"
    preferred = requested if requested != "auto" else (
        "cuda" if experimental_auto and large and gpu_eligible else "cpu")
    if requested != "auto":
        reason = f"{requested.upper()} selected manually."
    elif not gpu_eligible:
        reason = "Selected algorithm primarily uses CPU operations."
    elif not experimental_auto:
        reason = "CPU selected by default: measured whole-solver CUDA runs have not shown a speedup for the tested workloads. Select CUDA explicitly to compare."
    elif large:
        reason = "Experimental size-based routing prefers CUDA; use CPU if no matching GPU is online."
    else:
        reason = "Small model: CPU avoids GPU transfer and launch overhead."
    return {"preferred_device": preferred, "reason": reason, "shape": shape,
            "policy": "experimental-structural-v1" if experimental_auto else "cpu-baseline-v2",
            "execution_device": None}
