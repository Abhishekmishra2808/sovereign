"""Generate robustness demos, scale stress models, and a small MIPLIB-style MIP."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "benchmarks" / "datasets"


def write(path: Path, model: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print("wrote", path.relative_to(ROOT), f"vars={len(model['variables'])} cons={len(model['constraints'])}")


def kuhn_cycling_lp() -> dict:
    """Classic degenerate LP that cycles under naive Dantzig pivoting without
    anti-cycling (Kuhn-style / textbook degeneracy demo).

    Our revised simplex switches to Bland's rule after stagnant pivots.
    Expected optimal objective = -1.25 (minimize) at a degenerate vertex.
    Reference story: without Bland/lex, some pivot rules revisit bases.
    """
    return {
        "problem_type": "LP",
        "sense": "minimize",
        "variables": [
            {"name": "x1", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30},
            {"name": "x2", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30},
            {"name": "x3", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30},
            {"name": "x4", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30},
        ],
        "objective": {"linear": {"x1": -0.75, "x2": 20.0, "x3": -0.5, "x4": 6.0}},
        "constraints": [
            {
                "name": "r1",
                "linear": {"x1": 0.25, "x2": -8.0, "x3": -1.0, "x4": 9.0},
                "sense": "<=",
                "rhs": 0.0,
            },
            {
                "name": "r2",
                "linear": {"x1": 0.5, "x2": -12.0, "x3": -0.5, "x4": 3.0},
                "sense": "<=",
                "rhs": 0.0,
            },
            {
                "name": "r3",
                "linear": {"x1": 0.0, "x2": 0.0, "x3": 1.0, "x4": 0.0},
                "sense": "<=",
                "rhs": 1.0,
            },
        ],
        "notes": {
            "name": "robustness_kuhn_degeneracy",
            "purpose": "Degenerate LP; demonstrates Bland anti-cycling path in revised simplex",
            "expected_status": "OPTIMAL",
            "expected_objective_approx": -1.25,
        },
    }


def illconditioned_lp() -> dict:
    """Ill-conditioned constraint matrix: coefficients span ~1e-6 .. 1e6.

    Numerically equivalent (after scaling) to maximize 3x+2y s.t. x+y<=10, x,y>=0
    with known optimum 30 at (10,0). Without row/column scaling a naive solver
    can lose feasibility or report a wrong objective on the raw system.
    """
    return {
        "problem_type": "LP",
        "sense": "maximize",
        "variables": [
            {"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30},
            {"name": "y", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30},
        ],
        "objective": {"linear": {"x": 3.0e6, "y": 2.0e6}},
        "constraints": [
            {
                "name": "scaled_resource",
                "linear": {"x": 1.0e6, "y": 1.0e6},
                "sense": "<=",
                "rhs": 1.0e7,
            },
            {
                "name": "tiny_cut",
                "linear": {"x": 1.0e-6, "y": 0.0},
                "sense": ">=",
                "rhs": 0.0,
            },
        ],
        "notes": {
            "name": "robustness_illconditioned",
            "purpose": "Ill-conditioned coeffs; scaling + tolerances must recover obj≈30",
            "expected_status": "OPTIMAL",
            "expected_objective_approx": 30000000.0,
            "note": "objective is scaled 1e6× vs sample_lp; primal still x=10,y=0",
        },
    }


def weak_lp_relaxation_milp() -> dict:
    """MILP with weak LP relaxation (fractional root) — gap closed by B&B.

    max 100 x + 99 y  s.t. 100 x + 99 y <= 100, x,y binary
    LP relaxation opt = 100 at fractional points; integer opt = 100 at (1,0)
    or 99 at (0,1). Forces tree search / cuts usefulness story.
    """
    return {
        "problem_type": "MILP",
        "sense": "maximize",
        "variables": [
            {"name": "x", "type": "binary", "lower_bound": 0, "upper_bound": 1},
            {"name": "y", "type": "binary", "lower_bound": 0, "upper_bound": 1},
        ],
        "objective": {"linear": {"x": 100, "y": 99}},
        "constraints": [
            {"name": "weak", "linear": {"x": 100, "y": 99}, "sense": "<=", "rhs": 100}
        ],
        "notes": {
            "name": "robustness_weak_lp_relaxation",
            "purpose": "Weak LP relaxation; integer optimum via branch-and-bound",
            "expected_status": "OPTIMAL",
            "expected_objective_approx": 100.0,
        },
    }


def miplib_style_knapsack() -> dict:
    """Small 0-1 knapsack in the spirit of MIPLIB toy instances (not an official ID).

    Capacity knapsack with known optimum 34 (items with values 19+8+7).
    """
    values = [15, 10, 9, 19, 8, 7]
    weights = [8, 6, 5, 10, 4, 3]
    capacity = 17
    variables = [
        {"name": f"x{i}", "type": "binary", "lower_bound": 0, "upper_bound": 1}
        for i in range(len(values))
    ]
    return {
        "problem_type": "MILP",
        "sense": "maximize",
        "variables": variables,
        "objective": {"linear": {f"x{i}": values[i] for i in range(len(values))}},
        "constraints": [
            {
                "name": "capacity",
                "linear": {f"x{i}": weights[i] for i in range(len(weights))},
                "sense": "<=",
                "rhs": capacity,
            }
        ],
        "notes": {
            "name": "miplib_style_knapsack6",
            "family": "MIPLIB-style 0-1 knapsack (curated, not official MIPLIB ID)",
            "expected_status": "OPTIMAL",
            "expected_objective_approx": 34.0,
        },
    }


def multi_knapsack_milp(n_items: int = 20, n_knapsacks: int = 3, seed: int = 7) -> dict:
    """Harder multi-dimensional 0-1 knapsack (weak LP relaxation).

    Not an official MIPLIB ID — synthetic but large enough that plain B&B
    and branch-and-cut/strong branching diverge in node counts.
    """
    # Deterministic LCG
    state = seed

    def rnd() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    values = [1 + int(20 * rnd()) for _ in range(n_items)]
    weights = [[1 + int(10 * rnd()) for _ in range(n_items)] for _ in range(n_knapsacks)]
    capacities = []
    for k in range(n_knapsacks):
        total = sum(weights[k])
        capacities.append(max(5, int(0.35 * total)))

    variables = [
        {"name": f"x{i}", "type": "binary", "lower_bound": 0, "upper_bound": 1}
        for i in range(n_items)
    ]
    constraints = []
    for k in range(n_knapsacks):
        constraints.append(
            {
                "name": f"cap_{k}",
                "linear": {f"x{i}": weights[k][i] for i in range(n_items)},
                "sense": "<=",
                "rhs": capacities[k],
            }
        )
    return {
        "problem_type": "MILP",
        "sense": "maximize",
        "variables": variables,
        "objective": {"linear": {f"x{i}": values[i] for i in range(n_items)}},
        "constraints": constraints,
        "notes": {
            "name": f"multi_knapsack_{n_items}x{n_knapsacks}",
            "purpose": "Harder MILP for B&B vs branch-and-cut ablation",
            "n_items": n_items,
            "n_knapsacks": n_knapsacks,
        },
    }


def set_partition_milp(n_sets: int = 12, universe: int = 8, seed: int = 3) -> dict:
    """Set-partition / set-cover style 0-1 MILP with weak relaxations."""
    state = seed

    def rnd() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    # Each set covers a random subset of elements
    covers = []
    costs = []
    for s in range(n_sets):
        mask = [1 if rnd() < 0.35 else 0 for _ in range(universe)]
        if sum(mask) == 0:
            mask[int(rnd() * universe) % universe] = 1
        covers.append(mask)
        costs.append(1 + int(8 * rnd()))

    variables = [
        {"name": f"s{j}", "type": "binary", "lower_bound": 0, "upper_bound": 1}
        for j in range(n_sets)
    ]
    constraints = []
    for e in range(universe):
        linear = {f"s{j}": covers[j][e] for j in range(n_sets) if covers[j][e]}
        if not linear:
            continue
        constraints.append(
            {"name": f"cover_{e}", "linear": linear, "sense": ">=", "rhs": 1}
        )
    return {
        "problem_type": "MILP",
        "sense": "minimize",
        "variables": variables,
        "objective": {"linear": {f"s{j}": costs[j] for j in range(n_sets)}},
        "constraints": constraints,
        "notes": {
            "name": f"set_partition_{n_sets}x{universe}",
            "purpose": "Set-cover MILP for branching/cut ablation",
        },
    }


def strip_notes(model: dict) -> tuple[dict, dict]:
    notes = model.pop("notes", {})
    return model, notes


def scale_transport_lp(n_sources: int = 40, n_sinks: int = 40) -> dict:
    """Balanced transportation LP: n*m variables, n+m constraints.

    Default 40×40 → 1600 vars, 80 cons — mid-size stress beyond toy demos.
    """
    variables = []
    objective = {}
    for i in range(n_sources):
        for j in range(n_sinks):
            name = f"x_{i}_{j}"
            variables.append(
                {"name": name, "type": "continuous", "lower_bound": 0, "upper_bound": 1e30}
            )
            # mild cost variation
            objective[name] = 1.0 + 0.01 * ((i + 1) * (j + 1) % 17)

    supply = 10.0
    demand = (n_sources * supply) / n_sinks
    constraints = []
    for i in range(n_sources):
        constraints.append(
            {
                "name": f"supply_{i}",
                "linear": {f"x_{i}_{j}": 1.0 for j in range(n_sinks)},
                "sense": "=",
                "rhs": supply,
            }
        )
    for j in range(n_sinks):
        constraints.append(
            {
                "name": f"demand_{j}",
                "linear": {f"x_{i}_{j}": 1.0 for i in range(n_sources)},
                "sense": "=",
                "rhs": demand,
            }
        )
    return {
        "problem_type": "LP",
        "sense": "minimize",
        "variables": variables,
        "objective": {"linear": objective},
        "constraints": constraints,
        "notes": {
            "name": f"scale_transport_{n_sources}x{n_sinks}",
            "purpose": "Scale stress: thousands of variables transportation LP",
            "n_vars": n_sources * n_sinks,
            "n_cons": n_sources + n_sinks,
        },
    }


def main() -> None:
    catalog = []

    for builder, rel in [
        (kuhn_cycling_lp, DS / "robustness" / "kuhn_degeneracy.json"),
        (illconditioned_lp, DS / "robustness" / "illconditioned.json"),
        (weak_lp_relaxation_milp, DS / "robustness" / "weak_lp_relaxation.json"),
        (miplib_style_knapsack, DS / "miplib" / "knapsack6.json"),
        (lambda: multi_knapsack_milp(18, 3, 11), DS / "miplib" / "multi_knapsack_18x3.json"),
        (lambda: multi_knapsack_milp(24, 4, 19), DS / "miplib" / "multi_knapsack_24x4.json"),
        (lambda: set_partition_milp(14, 9, 5), DS / "miplib" / "set_partition_14x9.json"),
    ]:
        model = builder()
        model, notes = strip_notes(model)
        write(rel, model)
        catalog.append({"path": str(rel.relative_to(ROOT)).replace("\\", "/"), **notes})

    # Scale: 50x50 = 2500 vars (mid-size). Also a smaller 20x20 smoke.
    for n in (20, 50):
        model = scale_transport_lp(n, n)
        model, notes = strip_notes(model)
        rel = DS / "scale" / f"transport_{n}x{n}.json"
        write(rel, model)
        catalog.append({"path": str(rel.relative_to(ROOT)).replace("\\", "/"), **notes})

    cat_path = DS / "catalog.json"
    cat_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print("wrote", cat_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
