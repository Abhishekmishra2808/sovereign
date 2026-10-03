"""Randomized GI-vs-brute-force test for small convex-sequence projections."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def matrix(n: int) -> np.ndarray:
    a = np.zeros((n, n + 2))
    rows = np.arange(n)
    a[rows, rows] = 1.0
    a[rows, rows + 1] = -2.0
    a[rows, rows + 2] = 1.0
    return a


def brute_force(c: np.ndarray) -> np.ndarray:
    n = len(c) - 2
    a = matrix(n)
    best: tuple[float, np.ndarray] | None = None
    for mask in range(1 << n):
        active = [i for i in range(n) if mask & (1 << i)]
        if active:
            aw = a[active]
            kkt = np.block([
                [np.eye(n + 2), -aw.T],
                [aw, np.zeros((len(active), len(active)))],
            ])
            try:
                solution = np.linalg.solve(
                    kkt, np.r_[c, np.zeros(len(active))]
                )
            except np.linalg.LinAlgError:
                continue
            x = solution[: n + 2]
            lam_active = solution[n + 2:]
        else:
            x = c.copy()
            lam_active = np.empty(0)
        slack = a @ x
        if slack.min(initial=0.0) < -1e-9 or lam_active.min(initial=0.0) < -1e-9:
            continue
        distance = 0.5 * float(np.dot(x - c, x - c))
        if best is None or distance < best[0]:
            best = (distance, x)
    if best is None:
        raise RuntimeError("brute-force reference found no KKT point")
    return best[1]


def model(c: np.ndarray) -> dict:
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


def solve(binary: Path, c: np.ndarray) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as handle:
        json.dump(model(c), handle)
        path = handle.name
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SOVEREIGN_")}
    try:
        completed = subprocess.run(
            [str(binary), "solve", path], capture_output=True, text=True,
            env=env, check=False, timeout=60,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    result = json.loads(completed.stdout)
    if result.get("status") != "OPTIMAL":
        raise RuntimeError(result.get("message", "GI did not certify"))
    return np.asarray(
        [result["primal"][f"x{i}"] for i in range(len(c))],
        dtype=np.float64,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "build64" / "solver" / "sovereign.exe")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=26119)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    cases: list[np.ndarray] = []
    # Include explicit structural edge cases.
    t = np.linspace(-1.0, 1.0, 8)
    cases.append(0.3 + 0.2 * t + 0.4 * t * t)       # already convex
    cases.append(0.3 + 0.2 * t - 0.4 * t * t)       # strictly concave
    while len(cases) < args.count:
        n = int(rng.integers(5, 13))
        t = np.linspace(-1.0, 1.0, n + 2)
        scale = 10.0 ** rng.uniform(-2.0, 2.0)
        smooth = rng.uniform(-1.0, 1.0) + rng.uniform(-1.0, 1.0) * t
        noise = rng.normal(0.0, rng.uniform(0.0, 0.2), n + 2)
        cases.append(scale * (smooth + 0.3 * t * t + noise))

    max_error = 0.0
    failures = []
    for index, c in enumerate(cases):
        reference = brute_force(c)
        try:
            candidate = solve(args.binary, c)
            error = float(np.max(np.abs(candidate - reference)))
            max_error = max(max_error, error)
        except Exception as exc:
            failures.append({"case": index, "error": str(exc)})

    report = {
        "count": len(cases),
        "seed": args.seed,
        "max_x_inf_error": max_error,
        "failures": failures,
        "passed": not failures and max_error <= 1e-8,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
