"""Diagnostics-only synthetic LISWET ladder runner.

This tool does not alter solver code or invent CUTEst data. It records the
returned primal when available and recomputes feasibility/objectives in Python.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def make(n: int) -> tuple[dict, np.ndarray, np.ndarray]:
    t = np.linspace(-1.0, 1.0, n + 2)
    c = 0.4 * t * t + 0.1 * np.sin(7.0 * t)
    c += np.random.default_rng(26119 + n).normal(0.0, 0.02, n + 2)
    rows = []
    for i in range(n):
        rows.append({"name": f"r{i}", "linear": {f"x{i}": 1.0, f"x{i+1}": -2.0, f"x{i+2}": 1.0},
                     "sense": ">=", "rhs": 0.0})
    model = {
        "problem_type": "QP", "sense": "minimize",
        "variables": [{"name": f"x{i}", "type": "continuous",
                       "lower_bound": -1e30, "upper_bound": 1e30} for i in range(n + 2)],
        "objective": {"linear": {f"x{i}": float(-c[i]) for i in range(n + 2)},
                      "quadratic": {f"x{i}": {f"x{i}": 1.0} for i in range(n + 2)}},
        "constraints": rows,
    }
    a = np.zeros((n, n + 2))
    for i in range(n):
        a[i, i:i+3] = (1.0, -2.0, 1.0)
    return model, c, a


def run(binary: Path, model: dict, c: np.ndarray, a: np.ndarray, algo: str, limit: float) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(model, f)
        path = f.name
    try:
        env = {k: v for k, v in os.environ.items() if not k.startswith("SOVEREIGN_")}
        env.update({"SOVEREIGN_QP_ALGORITHM": algo, "SOVEREIGN_QP_MAX_ITERATIONS": "1000"})
        p = subprocess.run([str(binary), "solve", path], capture_output=True, text=True,
                           timeout=limit + 30, env=env)
        try:
            result = json.loads(p.stdout)
        except json.JSONDecodeError:
            return {"status": "PROCESS_ERROR", "stderr": p.stderr[-300:]}
    finally:
        Path(path).unlink(missing_ok=True)
    primal = result.get("primal") or {}
    if not primal:
        result["diagnostics"] = None
        return result
    x = np.array([primal.get(f"x{i}", np.nan) for i in range(len(c))])
    slack = a @ x
    qp = 0.5 * float(np.dot(x, x)) - float(np.dot(c, x))
    distance = 0.5 * float(np.dot(x - c, x - c))
    r = x - c
    lam = np.empty(len(c) - 2)
    lam[0] = r[0]
    if len(lam) > 1:
        lam[1] = r[1] + 2.0 * lam[0]
    for k in range(2, len(lam)):
        lam[k] = r[k] + 2.0 * lam[k - 1] - lam[k - 2]
    consistency = [
        r[-2] - (lam[-2] - 2.0 * lam[-1]),
        r[-1] - lam[-1],
    ]
    stationarity = float(np.max(np.abs(x - c - a.T @ lam)))
    complementarity = float(np.max(np.abs(lam * slack)))
    scale = max(1.0, float(np.max(np.abs(c))), float(np.max(np.abs(a.T @ lam))))
    result["diagnostics"] = {
        "max_primal_violation": float(max(0.0, -float(slack.min(initial=0.0)))),
        "min_lambda": float(lam.min(initial=0.0)),
        "lambda_inf": float(np.max(np.abs(lam))),
        "stationarity": stationarity,
        "complementarity": complementarity,
        "lambda_consistency_last_rows": consistency,
        "scaled_tolerance_scale": scale,
        "objective_qp": qp,
        "objective_distance": distance,
        "c_inf": float(np.max(np.abs(c))),
        "x_inf": float(np.max(np.abs(x))),
        "primal_feasibility_scale": f"absolute; tolerance 1e-6 * max(1, ||c||inf)",
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build64" / "solver" / "sovereign.exe"))
    ap.add_argument("--limit", type=float, default=120.0)
    ap.add_argument("--n", nargs="+", type=int, default=[10, 50, 200, 1000, 10000])
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "reports" / "liswet_diagnostics.jsonl")
    args = ap.parse_args()
    rows = []
    for n in args.n:
        model, c, a = make(n)
        for algo in ("auto", "ipm", "frank_wolfe"):
            rows.append({"n": n, "path": algo,
                         "result": run(Path(args.binary), model, c, a, algo, args.limit)})
            print(n, algo, rows[-1]["result"].get("status"),
                  rows[-1]["result"].get("runtime_seconds"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
