"""Cross-check GI against an independent sparse conic/QP solve.

CVXOPT is the default because the badly conditioned second-difference
constraints make OSQP's first-order stopping test inconclusive at n=1000+
on this machine. Clarabel and ``--method osqp`` remain available as
additional diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import clarabel
from cvxopt import matrix, solvers, spmatrix
import numpy as np
import osqp
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]


def synthetic_c(n: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n + 2)
    c = 0.4 * t * t + 0.1 * np.sin(7.0 * t)
    return c + np.random.default_rng(26119 + n).normal(0.0, 0.02, n + 2)


def make_model(c: np.ndarray) -> dict:
    n = len(c) - 2
    return {
        "problem_type": "QP",
        "sense": "minimize",
        "variables": [
            {"name": f"x{i}", "type": "continuous",
             "lower_bound": -1e30, "upper_bound": 1e30}
            for i in range(n + 2)
        ],
        "objective": {
            "linear": {f"x{i}": float(-c[i]) for i in range(n + 2)},
            "quadratic": {f"x{i}": {f"x{i}": 1.0} for i in range(n + 2)},
        },
        "constraints": [
            {"name": f"r{i}",
             "linear": {f"x{i}": 1.0, f"x{i+1}": -2.0, f"x{i+2}": 1.0},
             "sense": ">=", "rhs": 0.0}
            for i in range(n)
        ],
    }


def objective(x: np.ndarray, c: np.ndarray) -> np.longdouble:
    xl = x.astype(np.longdouble)
    cl = c.astype(np.longdouble)
    return np.longdouble("0.5") * np.dot(xl, xl) - np.dot(cl, xl)


def run(binary: Path, n: int, out_dir: Path, method: str) -> dict:
    c = synthetic_c(n)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as handle:
        json.dump(make_model(c), handle)
        model_path = handle.name
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SOVEREIGN_")}
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(binary), "solve", model_path],
            capture_output=True, text=True, env=env, check=False,
            timeout=600,
        )
    finally:
        Path(model_path).unlink(missing_ok=True)
    elapsed = time.perf_counter() - start
    result = json.loads(completed.stdout)
    if result.get("status") != "OPTIMAL":
        raise RuntimeError(f"GI failed for n={n}: {result.get('message')}")
    x_gi = np.asarray(
        [result["primal"][f"x{i}"] for i in range(n + 2)],
        dtype=np.float64,
    )

    rows = np.ones(n)
    a = sparse.diags(
        [rows, -2.0 * rows, rows], [0, 1, 2],
        shape=(n, n + 2), format="csr",
    )
    a_csc = a.tocsc()
    if method == "cvxopt":
        p_matrix = spmatrix(1.0, range(n + 2), range(n + 2))
        row_indices = []
        column_indices = []
        values = []
        for i in range(n):
            for j, value in ((i, -1.0), (i + 1, 2.0), (i + 2, -1.0)):
                row_indices.append(i)
                column_indices.append(j)
                values.append(value)
        g_matrix = spmatrix(
            values, row_indices, column_indices, (n, n + 2)
        )
        solvers.options.update({
            "show_progress": False,
            "abstol": 1e-8,
            "reltol": 1e-8,
            "feastol": 1e-8,
            "maxiters": 500,
        })
        reference_result = solvers.qp(
            p_matrix, matrix(-c), g_matrix, matrix(np.zeros(n))
        )
        x_ref = np.asarray(reference_result["x"], dtype=np.float64).reshape(-1)
        reference_status = str(reference_result["status"])
        reference_status_val = None
        reference_iterations = int(reference_result["iterations"])
        reference_primal_residual = float(
            reference_result["primal infeasibility"]
        )
        reference_dual_residual = float(
            reference_result["dual infeasibility"]
        )
    elif method == "clarabel":
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.tol_gap_abs = 1e-9
        settings.tol_gap_rel = 1e-9
        settings.tol_feas = 1e-9
        settings.max_iter = 500
        reference = clarabel.DefaultSolver(
            sparse.eye(n + 2, format="csc"),
            -c,
            -a_csc,
            np.zeros(n),
            [clarabel.NonnegativeConeT(n)],
            settings,
        )
        reference.solve()
        reference_result = reference.get_solution()
        x_ref = np.asarray(reference_result.x, dtype=np.float64)
        reference_status = str(reference_result.status)
        reference_status_val = None
        reference_iterations = int(reference_result.iterations)
        reference_primal_residual = float(reference_result.r_prim)
        reference_dual_residual = float(reference_result.r_dual)
    else:
        reference = osqp.OSQP()
        reference.setup(
            P=sparse.eye(n + 2, format="csc"),
            q=-c,
            A=a,
            l=np.zeros(n),
            u=np.full(n, np.inf),
            eps_abs=1e-9,
            eps_rel=1e-9,
            max_iter=200000,
            polish=True,
            adaptive_rho=True,
            verbose=False,
        )
        reference_result = reference.solve()
        x_ref = np.asarray(reference_result.x, dtype=np.float64)
        reference_status = str(reference_result.info.status)
        reference_status_val = int(reference_result.info.status_val)
        reference_iterations = int(reference_result.info.iter)
        reference_primal_residual = float(reference_result.info.prim_res)
        reference_dual_residual = float(reference_result.info.dual_res)
    x_difference = np.max(
        np.abs(x_gi.astype(np.longdouble) - x_ref.astype(np.longdouble))
    )
    objective_difference = abs(objective(x_gi, c) - objective(x_ref, c))
    ref_objective = abs(objective(x_ref, c))
    report = {
        "n": n,
        "reference_method": method,
        "gi_runtime_seconds": elapsed,
        "reference_status": reference_status,
        "reference_status_val": reference_status_val,
        "reference_iterations": reference_iterations,
        "reference_primal_residual": reference_primal_residual,
        "reference_dual_residual": reference_dual_residual,
        "reference_objective": float(objective(x_ref, c)),
        "x_difference_inf_longdouble": str(x_difference),
        "objective_difference_longdouble": str(objective_difference),
        "objective_relative_difference": str(
            objective_difference / max(np.longdouble(1), ref_objective)
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"nnls_n{n}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "build64" / "solver" / "sovereign.exe")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "benchmarks" / "reports" / "liswet_step6")
    parser.add_argument("--n", nargs="+", type=int, default=[1000, 5000])
    parser.add_argument("--method", choices=("cvxopt", "clarabel", "osqp"),
                        default="cvxopt")
    args = parser.parse_args()
    for n in args.n:
        print(json.dumps(run(args.binary, n, args.out_dir, args.method)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
