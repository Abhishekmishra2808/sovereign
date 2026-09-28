"""Generate the GPU showcase models: interior-point systems large enough for CUDA to win.

The CUDA path accelerates the dense factorization of the interior-point normal
equations, whose order is roughly the number of rows (plus columns for QP). Below
about 400 the CPU is faster, so the Lab's small public models never show a GPU
speed-up. These models are sized 600 to 1500 so the difference is visible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "datasets" / "gpu"


def production_plan(m: int, n: int, per_col: int = 6) -> dict:
    """max c'x s.t. Ax <= b, x >= 0 with a sparse positive A (per_col nonzeros per column)."""
    rng = random.Random(m * 7919 + n)
    rows = [dict() for _ in range(m)]
    for j in range(n):
        for i in rng.sample(range(m), min(per_col, m)):
            rows[i][f"x{j}"] = round(rng.uniform(0.5, 3.0), 3)
    return {
        "problem_type": "LP", "sense": "maximize",
        "variables": [{"name": f"x{j}", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30}
                      for j in range(n)],
        "objective": {"linear": {f"x{j}": round(rng.uniform(1.0, 10.0), 3) for j in range(n)}},
        "constraints": [{"name": f"r{i}", "linear": row, "sense": "<=", "rhs": round(rng.uniform(50, 150), 2)}
                        for i, row in enumerate(rows) if row],
    }


def portfolio_qp(n: int, sectors: int) -> dict:
    """min 0.5 x'Qx - mu'x with a budget row and sector caps; Q is diagonally dominant, so convex."""
    rng = random.Random(n)
    names = [f"w{j}" for j in range(n)]
    quad = {a: {a: round(rng.uniform(0.5, 2.0), 3)} for a in names}
    for j in range(n):
        for k in rng.sample(range(n), 3):
            if k != j:
                v = round(rng.uniform(-0.05, 0.05), 4)
                quad[names[j]][names[k]] = quad[names[j]].get(names[k], 0) + v
                quad[names[k]][names[j]] = quad[names[k]].get(names[j], 0) + v
    rows = [{"name": "budget", "linear": {a: 1.0 for a in names}, "sense": "=", "rhs": 1.0}]
    for s in range(sectors):
        rows.append({"name": f"sector{s}", "linear": {names[j]: 1.0 for j in range(s, n, sectors)},
                     "sense": "<=", "rhs": 0.4})
    return {
        "problem_type": "QP", "sense": "minimize",
        "variables": [{"name": a, "type": "continuous", "lower_bound": 0, "upper_bound": 1e30} for a in names],
        "objective": {"linear": {a: -round(rng.uniform(0.01, 0.2), 4) for a in names}, "quadratic": quad},
        "constraints": rows,
    }


MODELS = {
    "plan_600x900": lambda: production_plan(600, 900),
    "plan_1000x1500": lambda: production_plan(1000, 1500),
    "plan_1500x2200": lambda: production_plan(1500, 2200),
    "portfolio_qp_600": lambda: portfolio_qp(600, 40),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in MODELS.items():
        path = OUT / f"{name}.json"
        model = build()
        path.write_text(json.dumps(model), encoding="utf-8")
        print("wrote", path.relative_to(ROOT), f"vars={len(model['variables'])} cons={len(model['constraints'])}")


if __name__ == "__main__":
    main()
