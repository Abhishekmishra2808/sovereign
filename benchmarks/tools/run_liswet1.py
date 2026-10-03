"""Run the real LISWET1 pipeline from an explicit c.txt file.

There is intentionally no synthetic fallback.  The input is converted to the
identity-Hessian convex-sequence QP, solved by Sovereign, and the returned
primal is written one value per line for independent verification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_c(path: Path, expected_length: int | None) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(
            f"LISWET1 c-file is missing: {path}; refusing synthetic fallback"
        )
    values = [float(token) for token in path.read_text(encoding="utf-8").split()]
    if len(values) < 3:
        raise ValueError("c-file must contain at least three values")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"expected {expected_length} values in {path}, got {len(values)}"
        )
    return values


def load_liswet1_sif(
    path: Path,
) -> tuple[list[float], str, dict[str, int]]:
    """Decode the LISWET1 data-section formula from a SIF file.

    LISWET1's SIF data section defines c_i explicitly as
    sqrt((i-1)/(N+K-1)) + 0.1*sin(i), i=1..N+K.  The SIF objective is
    0.5*sum(x_i^2) - sum(c_i*x_i), so no sign conversion is hidden here.
    """
    if not path.exists():
        raise FileNotFoundError(f"LISWET1 SIF is missing: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"(?im)^\s*NAME\s+LISWET1\s*$", text):
        raise ValueError(f"{path} is not a LISWET1 SIF file")
    upper = text.upper()
    groups_start = upper.find("GROUPS")
    constants_start = upper.find("CONSTANTS", groups_start + 1)
    elements_start = upper.find("ELEMENTS")
    if groups_start < 0 or constants_start < 0 or elements_start < 0:
        raise ValueError("LISWET1 SIF is missing GROUPS/CONSTANTS/ELEMENTS")
    groups_lines = [
        " ".join(line.split()).upper()
        for line in text[groups_start:constants_start].splitlines()
        if line.strip() and not line.lstrip().startswith("*")
    ]
    required_group_lines = (
        "R/ TI RI-1 RN+K-1",
        "R( GT SQRT TI",
        "R( RANDOM SIN RI",
        "R+ CI GT RANDOM",
        "RM -CI CI -1.0",
        "ZN OBJ X(I) -CI",
    )
    missing = [
        fragment for fragment in required_group_lines
        if not any(line.startswith(fragment) for line in groups_lines)
    ]
    if missing:
        raise ValueError(
            "LISWET1 SIF data formula does not match the supported "
            f"definition; missing lines: {missing}"
        )
    perturbation_match = next(
        (re.match(
            r"^RM\s+RANDOM\s+RANDOM\s+"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DE][+-]?\d+)?)\b",
            line,
        ) for line in groups_lines
         if line.startswith("RM RANDOM RANDOM ")),
        None,
    )
    if perturbation_match is None:
        raise ValueError("could not parse LISWET1 perturbation amplitude")
    perturbation = float(
        perturbation_match.group(1).replace("D", "E")
    )
    elements_lines = [
        " ".join(line.split()).upper()
        for line in text[elements_start:].splitlines()
        if line.strip() and not line.lstrip().startswith("*")
    ]
    quadratic_match = next(
        (re.match(
            r"^F\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DE][+-]?\d+)?)\s+\*\s+X\s+\*\s+X\b",
            line,
        ) for line in elements_lines if line.startswith("F ")),
        None,
    )
    if quadratic_match is None:
        raise ValueError("could not parse LISWET1 quadratic coefficient")
    quadratic_coefficient = float(
        quadratic_match.group(1).replace("D", "E")
    )
    active_lines = [
        line for line in text.splitlines()
        if not line.lstrip().startswith("*")
    ]
    n_match = None
    k_match = None
    for line in active_lines:
        if n_match is None:
            n_match = re.match(
                r"^\s*IE\s+N\s+(\d+)\b", line, re.IGNORECASE
            )
        if k_match is None:
            k_match = re.match(
                r"^\s*IE\s+K\s+(\d+)\b", line, re.IGNORECASE
            )
        if n_match is not None and k_match is not None:
            break
    if n_match is None or k_match is None:
        raise ValueError("LISWET1 SIF has no active IE N / IE K data lines")
    n = int(n_match.group(1))
    k = int(k_match.group(1))
    variables = n + k
    if variables < 3:
        raise ValueError(f"invalid LISWET1 variable count N+K={variables}")
    denominator = float(variables - 1)
    c = [
        math.sqrt(i / denominator) + perturbation * math.sin(i + 1)
        for i in range(variables)
    ]
    convention = (
        f"SIF: {quadratic_coefficient:g}*sum(x_i^2) - sum(c_i*x_i), "
        f"c_i=sqrt((i-1)/(N+K-1))+{perturbation:g}*sin(i)"
    )
    return c, convention, {"N": n, "K": k}


def crosscheck_pycutest(
    parameters: dict[str, int], c: list[float]
) -> dict[str, object]:
    """Compare -grad f(0) with c when a matching PyCUTEst install exists."""
    try:
        import numpy as np
        import pycutest
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}
    try:
        problem = pycutest.import_problem(
            "LISWET1", sifParams=parameters, quiet=True
        )
        if problem.n != len(c):
            return {
                "status": "mismatch",
                "reason": f"pycutest n={problem.n}, SIF c length={len(c)}",
            }
        _, gradient = problem.obj(
            np.zeros(len(c), dtype=float), gradient=True
        )
        difference = np.max(np.abs(np.asarray(gradient) + np.asarray(c)))
        return {
            "status": "matched" if difference <= 1e-10 else "mismatch",
            "gradient_plus_c_inf": float(difference),
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def make_model(c: list[float]) -> dict:
    n = len(c) - 2
    return {
        "problem_type": "QP",
        "sense": "minimize",
        "variables": [
            {
                "name": f"x{i}",
                "type": "continuous",
                "lower_bound": -1e30,
                "upper_bound": 1e30,
            }
            for i in range(n + 2)
        ],
        "objective": {
            "linear": {f"x{i}": -value for i, value in enumerate(c)},
            "quadratic": {f"x{i}": {f"x{i}": 1.0} for i in range(n + 2)},
        },
        "constraints": [
            {
                "name": f"r{i}",
                "linear": {
                    f"x{i}": 1.0,
                    f"x{i + 1}": -2.0,
                    f"x{i + 2}": 1.0,
                },
                "sense": ">=",
                "rhs": 0.0,
            }
            for i in range(n)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--c-file", type=Path)
    source.add_argument("--sif", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "build64" / "solver" / "sovereign.exe",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-length",
        type=int,
        help="assert c-file length; omit when --sif supplies N+K",
    )
    parser.add_argument("--high-precision", action="store_true")
    args = parser.parse_args()

    convention = "c-file values used as the linear coefficient -c"
    sif_parameters = None
    pycutest_report = None
    try:
        if args.sif:
            c, convention, sif_parameters = load_liswet1_sif(args.sif)
            pycutest_report = crosscheck_pycutest(sif_parameters, c)
        else:
            if args.expected_length is None:
                raise ValueError(
                    "--c-file requires --expected-length; use --sif to "
                    "derive N+K from LISWET1.SIF"
                )
            c = load_c(args.c_file, args.expected_length)
        model = make_model(c)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LISWET1 input error: {exc}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(model, handle)
        model_path = Path(handle.name)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SOVEREIGN_")
    }
    env["SOVEREIGN_QP_ALGORITHM"] = "auto"
    if args.high_precision:
        env["SOVEREIGN_QP_GI_HIGH_PRECISION"] = "1"
    try:
        completed = subprocess.run(
            [str(args.binary), "solve", str(model_path)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    finally:
        model_path.unlink(missing_ok=True)

    (args.out_dir / "solver.stdout.json").write_text(
        completed.stdout, encoding="utf-8"
    )
    (args.out_dir / "solver.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode or 1
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print(f"solver did not return JSON: {exc}", file=sys.stderr)
        return 1
    if result.get("status") != "OPTIMAL":
        print(
            f"LISWET1 solver did not certify OPTIMAL: "
            f"{result.get('status')}; {result.get('message', '')}",
            file=sys.stderr,
        )
        return 1

    primal = result.get("primal", {})
    try:
        x = [float(primal[f"x{i}"]) for i in range(len(c))]
    except (KeyError, TypeError, ValueError) as exc:
        print(f"solver response has incomplete primal output: {exc}", file=sys.stderr)
        return 1
    (args.out_dir / "c.txt").write_text(
        "".join(f"{value:.17g}\n" for value in c), encoding="utf-8"
    )
    (args.out_dir / "x.txt").write_text(
        "".join(f"{value:.17g}\n" for value in x), encoding="utf-8"
    )
    print(json.dumps({
        "status": result.get("status"),
        "n": len(c) - 2,
        "message": result.get("message"),
        "objective_convention": convention,
        "sif_parameters": sif_parameters,
        "pycutest_crosscheck": pycutest_report,
        "output_dir": str(args.out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
