"""Generate the sparse-scale models: multi-period production planning LPs.

Each period has an inventory balance row per product and one shared capacity
row, so the interior-point normal equations are block banded. The sparse
LDL^T factors them with little fill, while a dense factorization needs the full
m x m matrix (10,200 rows is 800 MB of doubles).
"""
from __future__ import annotations

import json
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


MODELS = {"staircase_20x100": (20, 100), "staircase_50x200": (50, 200)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (products, periods) in MODELS.items():
        path = OUT / f"{name}.json"
        model = staircase(products, periods)
        path.write_text(json.dumps(model), encoding="utf-8")
        print("wrote", path.relative_to(ROOT), f"vars={len(model['variables'])} cons={len(model['constraints'])}")


if __name__ == "__main__":
    main()
