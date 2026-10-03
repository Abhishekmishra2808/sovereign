"""Evaluate a saved LISWET1 primal with PyCUTEst."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pycutest


def main() -> int:
    x_path = Path(os.environ.get(
        "LISWET_X_FILE",
        "/workspace/benchmarks/reports/liswet_real/x.txt",
    ))
    c_path = Path(os.environ.get(
        "LISWET_C_FILE",
        "/workspace/benchmarks/reports/liswet_real/c.txt",
    ))
    x = np.loadtxt(
        x_path,
        dtype=float,
    )
    c = np.loadtxt(
        c_path,
        dtype=float,
    )
    problem = pycutest.import_problem(
        "LISWET1", sifParams={"N": 2000, "K": 2}, quiet=True
    )
    objective = problem.obj(x)
    objective_zero = problem.obj(np.zeros(len(c), dtype=float))
    gradient = problem.grad(x)
    gradient_zero = problem.grad(np.zeros(len(c), dtype=float))
    c_cutest = -np.asarray(gradient_zero, dtype=float)
    if os.environ.get("DUMP_CUTEST") == "1":
        dump_path = Path(os.environ.get(
            "LISWET_CUTEST_OUT",
            "/workspace/benchmarks/reports/liswet_real/c_cutest.txt",
        ))
        np.savetxt(dump_path, c_cutest, fmt="%.17g")
    constraints = np.asarray(problem.cons(x))
    print(json.dumps({
        "problem_n": int(problem.n),
        "problem_m": int(problem.m),
        "objective": float(objective),
        "objective_zero": float(objective_zero),
        "objective_zero_from_c": float(0.5 * np.dot(c, c)),
        "gradient_inf": float(np.max(np.abs(gradient))),
        "c_cutest_minus_c_file_inf": float(
            np.max(np.abs(c_cutest - c))
        ),
        "constraint_min": float(np.min(constraints)),
        "constraint_max": float(np.max(constraints)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
