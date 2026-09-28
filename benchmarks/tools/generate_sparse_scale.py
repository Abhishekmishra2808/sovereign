"""Generate the sparse-scale models: production planning LPs and portfolio QPs.

Each planning period has an inventory balance row per product and one shared
capacity row, so the interior-point normal equations are block banded. The
sparse LDL^T factors them with little fill, while a dense factorization needs
the full m x m matrix (10,200 rows is 800 MB of doubles).

The portfolio QPs have a sector-block covariance (each asset correlates with a
few assets of its own sector), so the QP interior point's KKT system of order
assets + 2 x sectors + 1 also factors sparsely; 20,000 assets is far beyond a
dense factorization.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "datasets" / "sparse"


def staircase(products: int, periods: int) -> dict:
    variables, objective, constraints = [], {}, []
    for t in range(periods):
        capacity = {}
        for p in range(products):
            make, stock = f"make{p}_{t}", f"stock{p}_{t}"
            variables += [{"name": make, "type": "continuous", "lower_bound": 0, "upper_bound": 1e30},
                          {"name": stock, "type": "continuous", "lower_bound": 0, "upper_bound": 1e30}]
            objective[make] = 1.0 + 0.1 * ((p * 3 + t * 7) % 11)
            objective[stock] = 0.05 + 0.01 * (p % 4)
            balance = {make: 1.0, stock: -1.0}
            if t > 0:
                balance[f"stock{p}_{t - 1}"] = 1.0
            constraints.append({"name": f"balance{p}_{t}", "linear": balance, "sense": "=",
                                "rhs": 5.0 + (p * 5 + t * 3) % 9})
            capacity[make] = 1.0 + 0.1 * (p % 3)
        constraints.append({"name": f"capacity{t}", "linear": capacity, "sense": "<=", "rhs": 12.0 * products})
    return {"problem_type": "LP", "sense": "minimize", "variables": variables,
            "objective": {"linear": objective}, "constraints": constraints}


def sector_portfolio(n: int, sector_size: int) -> dict:
    """min 0.5 x'Qx - mu'x, budget row and a cap per sector; Q is diagonally dominant, so convex."""
    rng = random.Random(n * 31 + sector_size)
    names = [f"w{j}" for j in range(n)]
    quad = {a: {a: round(rng.uniform(0.5, 2.0), 3)} for a in names}
    for j in range(n):
        start = j - j % sector_size
        members = range(start, min(start + sector_size, n))
        for k in rng.sample(members, min(3, len(members))):
            if k != j:
                v = round(rng.uniform(-0.05, 0.05), 4)
                quad[names[j]][names[k]] = quad[names[j]].get(names[k], 0) + v
                quad[names[k]][names[j]] = quad[names[k]].get(names[j], 0) + v
    rows = [{"name": "budget", "linear": {a: 1.0 for a in names}, "sense": "=", "rhs": 1.0}]
    for s, start in enumerate(range(0, n, sector_size)):
        rows.append({"name": f"sector{s}", "linear": {names[j]: 1.0 for j in range(start, min(start + sector_size, n))},
                     "sense": "<=", "rhs": 0.05})
    return {
        "problem_type": "QP", "sense": "minimize",
        "variables": [{"name": a, "type": "continuous", "lower_bound": 0, "upper_bound": 1e30} for a in names],
        "objective": {"linear": {a: -round(rng.uniform(0.01, 0.2), 4) for a in names}, "quadratic": quad},
        "constraints": rows,
    }


MODELS = {
    "staircase_20x100": lambda: staircase(20, 100),
    "staircase_50x200": lambda: staircase(50, 200),
    "portfolio_qp_5000": lambda: sector_portfolio(5000, 50),
    "portfolio_qp_20000": lambda: sector_portfolio(20000, 50),
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
