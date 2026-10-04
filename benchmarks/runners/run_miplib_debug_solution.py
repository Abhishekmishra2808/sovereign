"""Run the Phase 1 known-solution soundness sweep.

This is a benchmark/debug harness only.  HiGHS-generated witnesses are inputs
to Sovereign's debug mode; they are never linked into the production solver.
The original MPS is independently parsed and checked before a row is accepted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "benchmarks" / "reports" / "miplib_stage2" / "corpus.json"
DEFAULT_OUT = ROOT / "benchmarks" / "reports" / "miplib_stage2" / "debug_sweep"
DEFAULT_BINARY = ROOT / "build-agent" / "solver" / "sovereign.exe"
# Match the solver's debug-solution bound check; this is not a benchmark
# acceptance relaxation.
REFERENCE_TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def dirty() -> bool | None:
    value = git_value("status", "--porcelain", "--untracked-files=all")
    return None if value == "unavailable" else bool(value)


def build_metadata() -> dict[str, Any]:
    cache = ROOT / "build-agent" / "CMakeCache.txt"
    values: dict[str, str] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.split(":", 1)[0]
            if key in {
                "CMAKE_BUILD_TYPE",
                "CMAKE_CXX_COMPILER",
                "CMAKE_CXX_FLAGS",
                "CMAKE_GENERATOR",
            }:
                values[key] = value
    return values


def load_entries() -> list[dict[str, Any]]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))["instances"]


def solver_environment(witness: Path, time_limit: float) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("SOVEREIGN_")}
    env.update(
        {
            "SOVEREIGN_DEBUG_SOLUTION": str(witness),
            "SOVEREIGN_TIME_LIMIT": str(time_limit),
            "SOVEREIGN_PARALLEL_WORKERS": "1",
            "SOVEREIGN_MAX_NODES": str(2**31 - 1),
            "SOVEREIGN_DISABLE_CUDA": "1",
        }
    )
    return env


def parse_result(stdout: str) -> dict[str, Any] | None:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def independent_row(entry: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    # Importing the reference parser is permitted in benchmark tooling.  It
    # checks the returned vector against the original MPS, not a transformed
    # model or the solver's own verifier.
    sys.path.insert(0, str(ROOT / "benchmarks" / "runners"))
    import run_coverage as coverage

    ref = coverage.ref_from_mps(ROOT / entry["file"])
    scored = coverage.score(
        {**entry, "suite": "miplib"},
        "MILP",
        dict(result),
        ref,
        float("inf"),
    )
    diagnostics = result.get("mip_diagnostics") or {}
    bound = diagnostics.get("best_bound")
    optimum = entry["reference"].get("value")
    bound_violation = None
    if bound is not None and optimum is not None:
        scale = max(1.0, abs(float(bound)), abs(float(optimum)))
        if ref.sense == 1.0:
            bound_violation = max(
                0.0, float(bound) - float(optimum) - REFERENCE_TOLERANCE * scale
            )
        else:
            bound_violation = max(
                0.0, float(optimum) - float(bound) - REFERENCE_TOLERANCE * scale
            )
    debug_enabled = bool(diagnostics.get("debug_solution_enabled"))
    debug_violation = diagnostics.get("debug_solution_violation")
    row = {
        "status": result.get("status", "ERROR"),
        "objective": result.get("objective_value"),
        "nodes": result.get("nodes"),
        "iterations": result.get("iterations"),
        "debug_solution_enabled": debug_enabled,
        "debug_solution_violation": debug_violation,
        "debug_solution_checks": diagnostics.get("debug_solution_checks"),
        "debug_solution_first_violation": diagnostics.get("debug_solution_first_violation"),
        "best_bound": bound,
        "best_bound_violation": bound_violation,
        "best_bound_verified": bound_violation is None or bound_violation == 0.0,
        "independent": {
            "outcome": scored.get("outcome"),
            "check": scored.get("check"),
            "missing_values": scored.get("missing_values"),
        },
        "raw_message": result.get("message", ""),
    }
    row["debug_trace_complete"] = debug_enabled and debug_violation is not None
    row["gate_ok"] = bool(
        row["debug_trace_complete"]
        and debug_violation is False
        and row["best_bound_verified"]
    )
    return row


def run_one(
    binary: Path, entry: dict[str, Any], witness_dir: Path, raw_dir: Path, limit: float
) -> dict[str, Any]:
    model = ROOT / entry["file"]
    witness = witness_dir / f"{entry['name']}.json"
    row: dict[str, Any] = {
        "name": entry["name"],
        "file": entry["file"],
        "source": entry.get("source"),
        "sha256": sha256(model) if model.exists() else None,
        "witness": witness.relative_to(ROOT).as_posix(),
    }
    if not model.exists():
        row.update({"status": "MISSING_INPUT", "gate_ok": False})
        return row
    if not witness.exists():
        row.update({"status": "MISSING_WITNESS", "gate_ok": False})
        return row
    witness_doc = json.loads(witness.read_text(encoding="utf-8"))
    if witness_doc.get("source_sha256") != row["sha256"]:
        row.update({"status": "WITNESS_HASH_MISMATCH", "gate_ok": False})
        return row

    raw_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    command = [str(binary), "solve", str(model)]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=solver_environment(witness, limit),
            capture_output=True,
            text=True,
            timeout=limit + 60.0,
            check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
        (raw_dir / f"{entry['name']}.stdout.json").write_text(stdout, encoding="utf-8")
        (raw_dir / f"{entry['name']}.stderr.log").write_text(stderr, encoding="utf-8")
        result = parse_result(stdout)
        row["returncode"] = completed.returncode
        row["wall_seconds"] = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        if result is None:
            row.update(
                {
                    "status": "ERROR",
                    "gate_ok": False,
                    "error": (stderr or stdout).strip()[-2000:],
                }
            )
        else:
            row.update(independent_row(entry, result))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        (raw_dir / f"{entry['name']}.stdout.json").write_text(
            stdout if isinstance(stdout, str) else stdout.decode(errors="replace"),
            encoding="utf-8",
        )
        (raw_dir / f"{entry['name']}.stderr.log").write_text(
            stderr if isinstance(stderr, str) else stderr.decode(errors="replace"),
            encoding="utf-8",
        )
        row.update(
            {
                "status": "TIMEOUT",
                "gate_ok": False,
                "wall_seconds": (dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
                "error": f"process exceeded harness timeout of {limit + 60.0:g} seconds",
            }
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if not args.binary.exists():
        raise SystemExit(f"missing solver binary: {args.binary}")
    entries = load_entries()
    selected = set(args.only or [entry["name"] for entry in entries])
    entries = [entry for entry in entries if entry["name"] in selected]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    witness_dir = ROOT / "benchmarks" / "reports" / "miplib_stage2" / "debug_solutions"
    metadata = {
        "suite": "miplib2017-stage2-debug-solution",
        "synthetic": False,
        "purpose": "Phase 1 soundness ablation; not a headline solver result",
        "commit": git_value("rev-parse", "HEAD"),
        "working_tree_dirty": dirty(),
        "binary": str(args.binary.resolve()),
        "binary_sha256": sha256(args.binary),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "compiler": build_metadata(),
        "threads": 1,
        "time_limit_seconds": args.time_limit,
        "environment": {
            "SOVEREIGN_TIME_LIMIT": str(args.time_limit),
            "SOVEREIGN_PARALLEL_WORKERS": "1",
            "SOVEREIGN_MAX_NODES": str(2**31 - 1),
            "SOVEREIGN_DISABLE_CUDA": "1",
            "SOVEREIGN_DEBUG_SOLUTION": "<per-instance witness path>",
        },
        "started": dt.datetime.now(dt.timezone.utc).isoformat(),
        "instances": [],
    }
    output = args.out_dir / "results.jsonl"
    if output.exists():
        output.unlink()
    for index, entry in enumerate(entries, 1):
        print(f"[{index}/{len(entries)}] {entry['name']}", flush=True)
        row = run_one(args.binary, entry, witness_dir, raw_dir, args.time_limit)
        metadata["instances"].append(row)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"  status={row.get('status')} gate_ok={row.get('gate_ok', False)} "
            f"violation={row.get('debug_solution_violation')}",
            flush=True,
        )
    metadata["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
    metadata["gate_passed"] = bool(metadata["instances"]) and all(
        row.get("gate_ok", False) for row in metadata["instances"]
    )
    metadata["counts"] = {
        "total": len(metadata["instances"]),
        "gate_ok": sum(row.get("gate_ok", False) for row in metadata["instances"]),
        "debug_violations": sum(
            row.get("debug_solution_violation") is True for row in metadata["instances"]
        ),
        "invalid_bounds": sum(
            row.get("best_bound_verified") is False for row in metadata["instances"]
        ),
        "incomplete_traces": sum(
            not row.get("debug_trace_complete", False) for row in metadata["instances"]
        ),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if metadata["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
