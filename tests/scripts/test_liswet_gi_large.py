"""Pinned deterministic n=10000 synthetic-scalability GI regression."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


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


def field(message: str, name: str) -> str:
    match = re.search(rf"(?:^|[; ]){re.escape(name)}=([^; ]+)", message)
    if not match:
        raise AssertionError(f"missing {name} in solver message")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "build64" / "solver" / "sovereign.exe")
    args = parser.parse_args()
    n = 10000
    t = np.linspace(-1.0, 1.0, n + 2)
    c = 0.4 * t * t + 0.1 * np.sin(7.0 * t)
    c += np.random.default_rng(26119 + n).normal(0.0, 0.02, n + 2)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as handle:
        json.dump(make_model(c), handle)
        path = handle.name
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SOVEREIGN_")}
    try:
        completed = subprocess.run(
            [str(args.binary), "solve", path], capture_output=True, text=True,
            env=env, check=False, timeout=180,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    result = json.loads(completed.stdout)
    assert result["status"] == "OPTIMAL", result.get("message")
    message = result["message"]
    assert int(field(message, "gi_iterations").split("/", 1)[0]) == 26328
    assert int(field(message, "drops")) == 8167
    assert int(field(message, "refreshes")) == 0
    assert int(field(message, "guard_trips")) == 0
    assert float(field(message, "certifier primal")) < 1e-8
    assert float(field(message, "dual")) < 1e-8
    assert float(field(message, "stationarity")) < 1e-8
    assert float(field(message, "complementarity")) < 1e-8
    assert np.isfinite(float(field(message, "lambda_inf")))
    print(json.dumps({
        "status": result["status"],
        "iterations": int(field(message, "gi_iterations").split("/", 1)[0]),
        "drops": int(field(message, "drops")),
        "primal": float(field(message, "certifier primal")),
        "dual": float(field(message, "dual")),
        "stationarity": float(field(message, "stationarity")),
        "complementarity": float(field(message, "complementarity")),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
