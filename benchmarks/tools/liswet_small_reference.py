"""Generate small LISWET-shaped QPs and trusted active-set references.

The CUTEst LISWET1 vector is intentionally not embedded here. Pass
``--c-file`` with one value per line when an official dump is available.
Without it, this tool creates only synthetic smooth-plus-noise test data.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def matrix(n: int) -> np.ndarray:
    a = np.zeros((n, n + 2))
    rows = np.arange(n)
    a[rows, rows] = 1.0
    a[rows, rows + 1] = -2.0
    a[rows, rows + 2] = 1.0
    return a


def reference(c: np.ndarray) -> tuple[np.ndarray, float]:
    a = matrix(len(c) - 2)
    best = None
    for mask in range(1 << a.shape[0]):
        active = [i for i in range(a.shape[0]) if mask & (1 << i)]
        if active:
            aw = a[active]
            kkt = np.block([[np.eye(len(c)), -aw.T], [aw, np.zeros((len(active), len(active)))]])
            rhs = np.r_[
                c,
                np.zeros(len(active)),
            ]
            try:
                sol = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                continue
            x, lam = sol[: len(c)], sol[len(c) :]
        else:
            x, lam = c.copy(), np.empty(0)
        slack = a @ x
        full_lam = np.zeros(a.shape[0])
        if active:
            full_lam[active] = lam
        if slack.min(initial=0.0) < -1e-8 or full_lam.min(initial=0.0) < -1e-8:
            continue
        value = 0.5 * float(np.dot(x - c, x - c))
        if best is None or value < best[1]:
            best = (x, value)
    if best is None:
        raise RuntimeError("no feasible active set found")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=(5, 10, 50, 200), default=10)
    parser.add_argument("--c-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.c_file:
        values = [float(line) for line in args.c_file.read_text().split()]
        c = np.asarray(values, dtype=float)
        if len(c) != args.n + 2:
            raise SystemExit(f"expected {args.n + 2} values, got {len(c)}")
        source = str(args.c_file)
    else:
        t = np.linspace(-1.0, 1.0, args.n + 2)
        c = 0.4 * t * t + 0.1 * np.sin(7.0 * t)
        c += np.random.default_rng(26119 + args.n).normal(0.0, 0.02, args.n + 2)
        source = "synthetic smooth-plus-noise"
    if args.n > 12:
        raise SystemExit("brute-force references are limited to n <= 12")
    x, distance_objective = reference(c)
    objective = 0.5 * float(np.dot(x, x)) - float(np.dot(c, x))
    a = matrix(args.n)
    model = {
        "problem_type": "QP",
        "sense": "minimize",
        "variables": [
            {"name": f"x{i}", "type": "continuous", "lower_bound": -1e30, "upper_bound": 1e30}
            for i in range(args.n + 2)
        ],
        "objective": {
            "linear": {f"x{i}": float(-c[i]) for i in range(args.n + 2)},
            "quadratic": {f"x{i}": {f"x{i}": 1.0} for i in range(args.n + 2)},
        },
        "constraints": [
            {
                "name": f"r{i}",
                "linear": {f"x{i}": 1.0, f"x{i + 1}": -2.0, f"x{i + 2}": 1.0},
                "sense": ">=",
                "rhs": 0.0,
            }
            for i in range(args.n)
        ],
    }
    args.out.write_text(json.dumps(model, indent=2))
    args.out.with_suffix(".reference.json").write_text(json.dumps({
        "source": source, "objective": objective,
        "distance_objective": distance_objective, "primal": x.tolist(),
    }, indent=2))
    print(f"wrote {args.out}: n={args.n} objective={objective:.12g} source={source}")


if __name__ == "__main__":
    main()
