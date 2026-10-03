"""Cross-check the official LISWET1 output with CVXOPT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from cvxopt import matrix, solvers, spmatrix


def load(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def objective_qp(x: np.ndarray, c: np.ndarray) -> np.longdouble:
    xl = x.astype(np.longdouble)
    cl = c.astype(np.longdouble)
    return np.longdouble("0.5") * np.dot(xl, xl) - np.dot(cl, xl)


def objective_distance(x: np.ndarray, c: np.ndarray) -> np.longdouble:
    delta = x.astype(np.longdouble) - c.astype(np.longdouble)
    return np.longdouble("0.5") * np.dot(delta, delta)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-file", type=Path, required=True)
    parser.add_argument("--x-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-length", type=int, default=2002)
    args = parser.parse_args()

    c = load(args.c_file)
    x_gi = load(args.x_file)
    if len(c) != args.expected_length or len(x_gi) != len(c):
        raise SystemExit(
            f"expected c/x length {args.expected_length}, "
            f"got {len(c)}/{len(x_gi)}"
        )
    n = len(c) - 2

    rows, columns, values = [], [], []
    for i in range(n):
        for j, value in ((i, -1.0), (i + 1, 2.0), (i + 2, -1.0)):
            rows.append(i)
            columns.append(j)
            values.append(value)
    g_matrix = spmatrix(values, rows, columns, (n, n + 2))
    solvers.options.update({
        "show_progress": False,
        "abstol": 1e-9,
        "reltol": 1e-9,
        "feastol": 1e-9,
        "maxiters": 500,
    })
    reference = solvers.qp(
        spmatrix(1.0, range(n + 2), range(n + 2)),
        matrix(-c),
        g_matrix,
        matrix(np.zeros(n)),
    )
    x_ref = np.asarray(reference["x"], dtype=np.float64).reshape(-1)
    a_x_ref = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a_x_ref[i] = x_ref[i] - 2.0 * x_ref[i + 1] + x_ref[i + 2]

    qp_gi = objective_qp(x_gi, c)
    qp_ref = objective_qp(x_ref, c)
    distance_gi = objective_distance(x_gi, c)
    distance_ref = objective_distance(x_ref, c)
    objective_difference = abs(distance_gi - distance_ref)
    report = {
        "n": n,
        "variables": len(c),
        "reference": "CVXOPT sparse primal QP",
        "reference_status": str(reference["status"]),
        "reference_iterations": int(reference["iterations"]),
        "reference_primal_infeasibility": float(
            reference["primal infeasibility"]
        ),
        "reference_dual_infeasibility": float(
            reference["dual infeasibility"]
        ),
        "reference_min_slack": float(np.min(a_x_ref)),
        "x_difference_inf": float(np.max(np.abs(x_gi - x_ref))),
        "gi_objective_qp": str(qp_gi),
        "reference_objective_qp": str(qp_ref),
        "gi_objective_distance": str(distance_gi),
        "reference_objective_distance": str(distance_ref),
        "objective_difference_longdouble": str(objective_difference),
        "relative_objective_difference": str(
            objective_difference / max(np.longdouble(1), abs(distance_ref))
        ),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
