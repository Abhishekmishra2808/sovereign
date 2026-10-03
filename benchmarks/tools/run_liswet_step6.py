"""Produce reproducible synthetic-scalability outputs and reports."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_liswet1 import make_model  # noqa: E402
from verify_liswet_independent import read_values, verify  # noqa: E402


def synthetic_c(n: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n + 2)
    c = 0.4 * t * t + 0.1 * np.sin(7.0 * t)
    return c + np.random.default_rng(26119 + n).normal(0.0, 0.02, n + 2)


def run_one(binary: Path, n: int, root: Path) -> dict:
    output_dir = root / f"n{n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    c = synthetic_c(n)
    c_path = output_dir / "input_c.txt"
    c_path.write_text("".join(f"{value:.17g}\n" for value in c), encoding="utf-8")
    model_path = output_dir / "model.json"
    model_path.write_text(
        json.dumps(make_model(c.tolist())), encoding="utf-8"
    )
    completed = subprocess.run(
        [str(binary), "solve", str(model_path)],
        capture_output=True, text=True,
        env={key: value for key, value in os.environ.items()
             if not key.startswith("SOVEREIGN_")},
        check=False,
    )
    (output_dir / "solver.stdout.json").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_dir / "solver.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"solver process failed for n={n}: {completed.stderr[-500:]}")
    result = json.loads(completed.stdout)
    if result.get("status") != "OPTIMAL":
        raise RuntimeError(f"solver failed n={n}: {result.get('message')}")
    x = [result["primal"][f"x{i}"] for i in range(n + 2)]
    x_path = output_dir / "x.txt"
    x_path.write_text("".join(f"{float(value):.17g}\n" for value in x), encoding="utf-8")
    independent = verify(read_values(c_path), read_values(x_path))
    report = {
        "problem_family": "synthetic scalability",
        "n": n,
        "solver_message": result.get("message"),
        "solver_status": result.get("status"),
        "independent": independent,
    }
    (output_dir / "independent.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "build64" / "solver" / "sovereign.exe")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "benchmarks" / "reports" /
                        "liswet_synthetic_scalability")
    parser.add_argument("--n", nargs="+", type=int, default=[200, 1000, 10000, 20000])
    args = parser.parse_args()
    reports = [run_one(args.binary, n, args.out_dir) for n in args.n]
    print("\n".join(json.dumps(report) for report in reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
