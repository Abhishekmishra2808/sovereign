"""Scale ladder: verified 100k -> 1M variable LP and QP solves with time and peak memory.

Each model comes from benchmarks/tools/generate_scale_ladder.py (generated on
demand, checked against its SHA-256 sidecar). The engine runs once per model
with --verify; the report records wall time, solver time, JSON load time, peak
working set, the verifier verdict and, with --highs, a single-threaded HiGHS
reference objective (benchmark harness only, never inside solver/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.tools.generate_scale_ladder import MODELS, OUT, generate  # noqa: E402

DEFAULT_ENGINE = ROOT / "build64" / "solver" / ("sovereign.exe" if os.name == "nt" else "sovereign")
DEFAULT_LADDER = ["lp_staircase_100k", "qp_portfolio_100k", "lp_staircase_1m", "qp_portfolio_1m"]
REPORTS = ROOT / "benchmarks" / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(name: str) -> tuple[Path, dict]:
    path, meta_path = OUT / f"{name}.json", OUT / f"{name}.meta.json"
    if not path.exists() or not meta_path.exists():
        print(f"generating {name} ...", flush=True)
        generate(name)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = sha256(path)
    if digest != meta["sha256"]:
        raise SystemExit(f"{path} does not match its recorded SHA-256; regenerate it")
    return path, meta


def run_engine(engine: Path, path: Path, timeout: float) -> dict:
    # stdout carries every primal value (tens of MB at 1M variables), so it goes
    # to a temporary file instead of a pipe.
    with tempfile.TemporaryFile() as out:
        start = time.perf_counter()
        try:
            proc = subprocess.run([str(engine), "solve", str(path), "--verify"], stdout=out,
                                  stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "TIME_LIMIT", "wall_seconds": time.perf_counter() - start}
        wall = time.perf_counter() - start
        out.seek(0)
        text = out.read().decode("utf-8", errors="replace").replace("\r\n", "\n")
    head, marker, tail = text.partition("\nverification:\n")
    try:
        response = json.loads(head)
        verification = json.loads(tail) if marker else {}
    except json.JSONDecodeError:
        return {"status": "ERROR", "wall_seconds": wall, "return_code": proc.returncode,
                "stderr_tail": proc.stderr.decode(errors="replace")[-500:]}
    return {
        "status": response.get("status"),
        "objective": response.get("objective_value"),
        "iterations": response.get("iterations"),
        "wall_seconds": wall,
        "solver_seconds": response.get("runtime_seconds"),
        "load_seconds": response.get("load_seconds"),
        "peak_memory_mb": response.get("peak_memory_mb"),
        "algorithm": response.get("message", ""),
        "verified": bool(verification.get("is_valid")),
        "max_constraint_violation": verification.get("max_constraint_violation"),
        "max_bound_violation": verification.get("max_bound_violation"),
        "return_code": proc.returncode,
    }


def run_highs(path: Path, limit: float) -> dict:
    from worker.runner import highs_reference
    try:
        ref = highs_reference(path, limit)
    except Exception as exc:  # noqa: BLE001 - the reference is best effort
        return {"status": "ERROR", "error": str(exc)[:300]}
    return {"status": ref["status"], "objective": ref["objective"],
            "runtime_seconds": ref["runtime_seconds"], "version": ref["version"]}


def rel_gap(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(1.0, abs(b))


def machine() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "cpu_threads": os.cpu_count() or 1, "python": platform.python_version()}
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_ext", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            info["ram_gb"] = round(status.total_phys / 2**30, 1)
        except (OSError, AttributeError):
            pass
    return info


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Scale ladder",
        "",
        f"Engine `{report['engine']}` (64-bit, single-threaded), {report['machine']['platform']}, "
        f"{report['machine'].get('ram_gb', '?')} GB RAM. Generated {report['generated_at']}.",
        "",
        "Wall time includes reading the JSON model; peak memory is the process peak working set. "
        "HiGHS runs single-threaded as a reference objective only.",
        "",
        "| Model | Type | Variables | Rows | Nonzeros | Status | Objective | Verified | Wall s | Solver s | "
        "Load s | Peak MB | IPM iters | HiGHS status | HiGHS s | Rel. gap |",
        "|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in report["rows"]:
        m, e, h = row["model"], row["engine"], row.get("highs") or {}
        nnz = m["nonzeros"] + m.get("quadratic_terms", 0)
        lines.append(
            f"| {m['name']} | {m['problem_type']} | {m['variables']:,} | {m['constraints']:,} | {nnz:,} | "
            f"{e.get('status')} | {fmt(e.get('objective'), '.10g')} | {'yes' if e.get('verified') else 'no'} | "
            f"{fmt(e.get('wall_seconds'), '.1f')} | {fmt(e.get('solver_seconds'), '.1f')} | "
            f"{fmt(e.get('load_seconds'), '.1f')} | {fmt(e.get('peak_memory_mb'), '.0f')} | "
            f"{fmt(e.get('iterations'), 'd')} | {h.get('status', 'not run')} | "
            f"{fmt(h.get('runtime_seconds'), '.1f')} | {fmt(row.get('relative_gap'), '.1e')} |")
    lines += ["", "Model files (regenerate with `python benchmarks/tools/generate_scale_ladder.py <name>`):", ""]
    for row in report["rows"]:
        lines.append(f"- `{row['model']['name']}`: {row['model']['bytes'] / 1e6:.1f} MB, "
                     f"sha256 `{row['model']['sha256']}`")
    lines += ["", "Solver messages:", ""]
    for row in report["rows"]:
        lines.append(f"- `{row['model']['name']}`: {row['engine'].get('algorithm', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=DEFAULT_LADDER, metavar="NAME",
                        help=f"models to run, from {', '.join(MODELS)} (default: {' '.join(DEFAULT_LADDER)})")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--timeout", type=float, default=3600.0, help="engine wall limit per model (s)")
    parser.add_argument("--highs", nargs="*", metavar="NAME",
                        help="also solve with HiGHS: the named models, or every model if none are named")
    parser.add_argument("--highs-limit", type=float, default=1800.0)
    parser.add_argument("--out", type=Path, default=REPORTS / "scale-ladder")
    args = parser.parse_args()
    unknown = [name for name in args.names if name not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")

    rows = []
    for name in args.names:
        path, meta = ensure_model(name)
        print(f"{name}: {meta['variables']:,} vars, {meta['constraints']:,} rows ...", flush=True)
        engine = run_engine(args.engine, path, args.timeout)
        print(f"  engine {engine.get('status')} obj={engine.get('objective')} "
              f"wall={engine.get('wall_seconds', 0):.1f}s peak={engine.get('peak_memory_mb')} MB "
              f"verified={engine.get('verified')}", flush=True)
        row = {"model": meta, "engine": engine}
        if args.highs is not None and (not args.highs or name in args.highs):
            row["highs"] = run_highs(path, args.highs_limit)
            row["relative_gap"] = rel_gap(engine.get("objective"), row["highs"].get("objective"))
            print(f"  highs {row['highs'].get('status')} obj={row['highs'].get('objective')} "
                  f"t={row['highs'].get('runtime_seconds')}", flush=True)
        rows.append(row)

    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "engine": str(args.engine.relative_to(ROOT) if args.engine.is_relative_to(ROOT) else args.engine),
              "machine": machine(), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.out.with_suffix(".md"))
    print(f"wrote {args.out.with_suffix('.json')} and .md")


if __name__ == "__main__":
    main()
