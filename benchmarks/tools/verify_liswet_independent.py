"""Independent high-precision verifier for convex-sequence projections.

This module deliberately does not import Sovereign code.  It reconstructs the
second-difference operator, recovers multipliers from x-c with mpmath, and
reports both objective conventions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import mpmath as mp


mp.mp.dps = 50


def read_values(path: Path) -> list[mp.mpf]:
    values = [mp.mpf(token) for token in path.read_text(encoding="utf-8").split()]
    if not values:
        raise ValueError(f"{path} is empty")
    return values


def inf_norm(values: Iterable[mp.mpf]) -> mp.mpf:
    return max((abs(value) for value in values), default=mp.mpf("0"))


def verify(c: list[mp.mpf], x: list[mp.mpf]) -> dict[str, str | int]:
    if len(c) != len(x):
        raise ValueError(f"c and x lengths differ: {len(c)} != {len(x)}")
    if len(c) < 3:
        raise ValueError("expected at least three variables")
    n = len(c) - 2
    residual = [x[i] - c[i] for i in range(n + 2)]

    lam = [mp.mpf("0") for _ in range(n)]
    lam[0] = residual[0]
    if n > 1:
        lam[1] = residual[1] + 2 * lam[0]
    for i in range(2, n):
        lam[i] = residual[i] + 2 * lam[i - 1] - lam[i - 2]

    slack = [
        x[i] - 2 * x[i + 1] + x[i + 2]
        for i in range(n)
    ]
    at_lambda = [mp.mpf("0") for _ in range(n + 2)]
    for i, value in enumerate(lam):
        at_lambda[i] += value
        at_lambda[i + 1] -= 2 * value
        at_lambda[i + 2] += value

    stationarity = inf_norm(
        residual[i] - at_lambda[i] for i in range(n + 2)
    )
    primal_violation = max(
        (max(mp.mpf("0"), -value) for value in slack),
        default=mp.mpf("0"),
    )
    complementarity = max(
        (abs(lam[i] * slack[i]) for i in range(n)),
        default=mp.mpf("0"),
    )
    qp_objective = mp.fsum(
        [mp.mpf("0.5") * value * value for value in x]
    ) - mp.fsum(c[i] * x[i] for i in range(n + 2))
    distance_objective = mp.fsum(
        [mp.mpf("0.5") * value * value for value in residual]
    )

    return {
        "n": n,
        "variables": n + 2,
        "min_slack": mp.nstr(min(slack), 30),
        "primal_violation": mp.nstr(primal_violation, 30),
        "min_lambda": mp.nstr(min(lam), 30),
        "lambda_inf": mp.nstr(inf_norm(lam), 30),
        "stationarity_inf": mp.nstr(stationarity, 30),
        "complementarity_inf": mp.nstr(complementarity, 30),
        "objective_qp": mp.nstr(qp_objective, 30),
        "objective_distance": mp.nstr(distance_objective, 30),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-file", type=Path, required=True)
    parser.add_argument("--x-file", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = verify(read_values(args.c_file), read_values(args.x_file))
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
