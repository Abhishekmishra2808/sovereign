"""Paired, verified CPU/CUDA whole-solver benchmark with raw response capture."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = [
    ROOT / "benchmarks/datasets/scale/transport_100x100.json",
    ROOT / "benchmarks/datasets/scale/transport_150x150.json",
    ROOT / "benchmarks/datasets/scale/transport_200x200.json",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_identity() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def invoke(engine: Path, model: Path, device: str, timeout: float):
    env = os.environ.copy()
    env.update(SOVEREIGN_DEVICE=device, SOVEREIGN_LP_ALGORITHM="ipm",
               SOVEREIGN_PRESOLVE="1")
    start = time.perf_counter()
    try:
        proc = subprocess.run([str(engine), "solve", str(model), "--verify"],
                              capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"device": device, "wall_seconds": time.perf_counter() - start,
                "solver_seconds": None, "status": "TIME_LIMIT", "objective": None,
                "gpu_operations": 0, "gpu_used": False, "verified": False,
                "return_code": None, "stdout_sha256": None,
                "response": {"error": f"Process exceeded {timeout}s wall limit",
                             "stdout_tail": str(exc.stdout or b"")[-500:]},
                "verification": {}}
    wall = time.perf_counter() - start
    head, marker, tail = proc.stdout.partition("\nverification:\n")
    try:
        if not marker:
            raise ValueError("Solver returned no verification block")
        response = json.loads(head)
        verification = json.loads(tail)
    except (ValueError, json.JSONDecodeError) as exc:
        response = {"status": "ERROR", "error": str(exc),
                    "stdout_tail": proc.stdout[-500:],
                    "stderr_tail": proc.stderr[-500:]}
        verification = {}
    return {
        "device": device, "wall_seconds": wall,
        "solver_seconds": response.get("runtime_seconds"),
        "status": response.get("status"),
        "objective": response.get("objective_value"),
        "gpu_operations": response.get("gpu_operations", 0),
        "gpu_used": response.get("gpu_used", False),
        "verified": verification.get("is_valid", False),
        "return_code": proc.returncode,
        "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
        "response": response, "verification": verification,
    }


def duration_stats(values):
    if not values:
        return None
    return {"median": statistics.median(values),
            "mad": statistics.median(abs(v - statistics.median(values)) for v in values),
            "minimum": min(values), "maximum": max(values)}


def summarize(runs):
    measured = [r for r in runs if not r["warmup"]]
    cpu = [r for r in measured if r["device"] == "cpu"]
    cuda = [r for r in measured if r["device"] == "cuda"]
    valid = all(r["status"] == "OPTIMAL" and r["verified"] and
                r["return_code"] == 0 for r in measured)
    real_gpu = all(r["gpu_used"] and r["gpu_operations"] > 0 for r in cuda)
    cpu_only = all(not r["gpu_used"] and r["gpu_operations"] == 0 for r in cpu)
    objectives_match = all(isinstance(c["objective"], (int, float)) and
                           isinstance(g["objective"], (int, float)) and
                           math.isclose(c["objective"], g["objective"],
                                        rel_tol=1e-7, abs_tol=1e-7)
                           for c, g in zip(cpu, cuda))
    wall_cpu = duration_stats([r["wall_seconds"] for r in cpu])
    wall_cuda = duration_stats([r["wall_seconds"] for r in cuda])
    solver_cpu = duration_stats([r["solver_seconds"] for r in cpu
                                 if isinstance(r["solver_seconds"], (int, float))])
    solver_cuda = duration_stats([r["solver_seconds"] for r in cuda
                                  if isinstance(r["solver_seconds"], (int, float))])
    return {"valid": valid and real_gpu and cpu_only and objectives_match,
            "all_verified_optimal": valid, "actual_cuda_used": real_gpu,
            "cpu_only": cpu_only, "objectives_match": objectives_match,
            "cpu_wall_seconds": wall_cpu, "cuda_wall_seconds": wall_cuda,
            "cpu_solver_seconds": solver_cpu, "cuda_solver_seconds": solver_cuda,
            "wall_speedup_cpu_over_cuda": wall_cpu["median"] / wall_cuda["median"],
            "solver_speedup_cpu_over_cuda": (solver_cpu["median"] / solver_cuda["median"]
                if solver_cpu and solver_cuda and solver_cuda["median"] > 0 else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path,
                        default=ROOT / "build-gpu-real/solver/Release/sovereign.exe")
    parser.add_argument("--models", type=Path, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/reports/gpu-paired.json")
    args = parser.parse_args()
    if args.trials < 2 or args.warmups < 0:
        parser.error("Use at least two trials and zero or more warmups.")
    engine = args.engine.resolve()
    if not engine.is_file():
        parser.error(f"Engine not found: {engine}")
    caps = json.loads(subprocess.check_output([str(engine), "capabilities"], text=True))
    if not caps.get("cuda_available"):
        parser.error("Engine reports no available CUDA GPU.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_name(args.output.stem + "-responses.jsonl.gz")
    report = {
        "machine": {"platform": platform.platform(), "cpu_threads": os.cpu_count(),
                    "gpu": gpu_identity()},
        "engine": str(engine), "engine_sha256": digest(engine), "capabilities": caps,
        "settings": {"trials": args.trials, "warmups": args.warmups,
                     "timeout_seconds": args.timeout, "lp_algorithm": "ipm",
                     "presolve": True, "pair_order": "alternating cpu/cuda then cuda/cpu"},
        "raw_responses": raw_path.name, "models": [],
    }
    with gzip.open(raw_path, "wt", encoding="utf-8") as raw:
        for model in args.models:
            model = model.resolve()
            entry = {"model": str(model), "sha256": digest(model), "runs": []}
            for index in range(args.warmups + args.trials):
                for device in (("cpu", "cuda") if index % 2 == 0 else ("cuda", "cpu")):
                    run = invoke(engine, model, device, args.timeout)
                    run["pair"] = index
                    run["warmup"] = index < args.warmups
                    raw.write(json.dumps({"model": str(model), **run}, allow_nan=False) + "\n")
                    entry["runs"].append({k: v for k, v in run.items()
                                          if k not in ("response", "verification")})
                    print(f"{model.name} pair={index} {device} "
                          f"status={run['status']} verified={run['verified']} "
                          f"gpu_ops={run['gpu_operations']} wall={run['wall_seconds']:.3f}s",
                          flush=True)
            entry["summary"] = summarize(entry["runs"])
            report["models"].append(entry)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                   encoding="utf-8")
    if not all(m["summary"]["valid"] for m in report["models"]):
        raise SystemExit("One or more CPU/CUDA comparison failed correctness; inspect report.")
    print(f"Wrote {args.output} and {raw_path}")


if __name__ == "__main__":
    main()
