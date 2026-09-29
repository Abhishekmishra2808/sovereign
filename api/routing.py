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
                "bounded_columns": sum(_finite_upper(v) for v in variables),
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
            "integer_variables": len(integers), "bounded_columns": 0,
            "problem_type": "MILP" if integers else "LP", "estimated": True}


def _finite_upper(variable):
    upper = variable.get("upper_bound", 1 if variable.get("type") == "binary" else None)
    return isinstance(upper, (int, float)) and upper < 1e29


def dense_order(shape):
    """Size of the dense system interior point factors every iteration.

    Upper bounds become extra rows; QP factors the full KKT system.
    """
    order = shape["rows"] + shape.get("bounded_columns", 0)
    if shape["problem_type"].upper() == "QP":
        order += shape["columns"]
    return order


def gpu_ineligible_reason(request, shape):
    kind = shape["problem_type"].upper()
    if kind == "MILP" or shape["integer_variables"]:
        return "Branch and bound solves node LPs with the dual simplex, which runs on CPU."
    if kind == "QP":
        if request.get("qpAlgorithm", "auto") == "frank_wolfe":
            return "Frank-Wolfe runs on CPU."
    elif request.get("algorithm", "auto") == "simplex":
        return "The simplex method runs on CPU."
    return None


def route_request(request):
    shape = summarize(request["modelJson"], request["modelFormat"])
    requested = request.get("device", "auto")
    ineligible = gpu_ineligible_reason(request, shape)
    order = dense_order(shape)
    # Below 400 rows the CUDA factorization never pays for its transfers. Above
    # it, a GPU machine runs the engine in auto mode, which still keeps sparse
    # models on its sparse CPU factorization: on an RTX 2050 the dense GPU path
    # won only on the 1,500-row dense planning LP (1.6x; gpu-dense-ipm.md).
    min_order = int(os.environ.get("SOVEREIGN_GPU_MIN_ROWS", "400"))
    auto_enabled = os.environ.get("SOVEREIGN_GPU_AUTO_ENABLED", "1") != "0"
    if requested != "auto":
        preferred, reason = requested, f"{requested.upper()} selected manually."
    elif ineligible:
        preferred, reason = "cpu", ineligible
    elif not auto_enabled:
        preferred, reason = "cpu", "Automatic GPU routing is turned off on this server."
    elif order >= min_order:
        preferred = "cuda"
        reason = (f"Large interior-point system ({order} rows): sent to a GPU machine, where the engine "
                  "uses CUDA only if the model is dense enough to benefit and otherwise stays on the CPU.")
    else:
        preferred = "cpu"
        reason = f"Small interior-point system ({order} rows): CPU avoids GPU launch and transfer overhead."
    return {"preferred_device": preferred, "reason": reason, "shape": shape, "dense_order": order,
            "policy": "measured-dense-v3", "execution_device": None}
