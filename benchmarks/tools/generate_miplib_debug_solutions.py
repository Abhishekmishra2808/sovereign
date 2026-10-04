"""Generate benchmark-only known optimal witnesses for soundness debugging.

HiGHS is used here only as a reference solver. The generated files are
debug-solution inputs for Sovereign's ``SOVEREIGN_DEBUG_SOLUTION`` mode and
must never be shipped with the solver or npm package.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import highspy

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks" / "manifests" / "miplib_stage2.json"
CORPUS = ROOT / "benchmarks" / "reports" / "miplib_stage2" / "corpus.json"
DEFAULT_OUT = ROOT / "benchmarks" / "reports" / "miplib_stage2" / "debug_solutions"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    if CORPUS.exists():
        entries = json.loads(CORPUS.read_text(encoding="utf-8")).get("instances", [])
    else:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        entries = [
            {
                "name": name,
                "file": f"benchmarks/datasets/coverage/miplib_stage2/{name}.mps",
                "source": manifest["instance_url"].format(name=name),
            }
            for name in manifest["instances"]
        ]
    selected = set(args.only or [entry["name"] for entry in entries])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    started = dt.datetime.now(dt.timezone.utc).isoformat()

    for entry in entries:
        name = entry["name"]
        if name not in selected:
            continue
        path = ROOT / entry["file"]
        if not path.exists():
            missing.append(
                {
                    "name": name,
                    "file": entry["file"],
                    "source": entry.get("source"),
                    "reason": "missing input; run run_miplib_stage2.py --fetch",
                }
            )
            continue

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("threads", 1)
        highs.setOptionValue("time_limit", float(args.time_limit))
        highs.setOptionValue("large_matrix_value", 1e30)
        read_status = highs.readModel(str(path))
        run_status = None
        status = "ERROR"
        error = None
        if read_status in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
            try:
                highs.run()
                run_status = highs.modelStatusToString(highs.getModelStatus()).upper()
                status = run_status.replace(" ", "_")
            except Exception as exc:  # preserve every reference failure
                error = str(exc)
        else:
            error = f"HiGHS readModel returned {read_status}"

        solution = highs.getSolution()
        valid = bool(solution.value_valid)
        lp = highs.getLp()
        row = {
            "name": name,
            "source": entry.get("source"),
            "file": entry["file"],
            "sha256": sha256(path),
            "status": status,
            "read_status": str(read_status),
            "run_status": run_status,
            "value_valid": valid,
            "objective": None,
            "output": None,
            "error": error,
        }
        if status == "OPTIMAL" and valid:
            info = highs.getInfo()
            values = {
                str(lp.col_names_[i]): float(solution.col_value[i])
                for i in range(lp.num_col_)
            }
            witness = {
                "source": "HiGHS benchmark reference witness; not production data",
                "source_instance": entry["file"],
                "source_sha256": row["sha256"],
                "reference_solver": highs.version(),
                "objective": float(info.objective_function_value),
                "primal": values,
            }
            output = args.out_dir / f"{name}.json"
            output.write_text(json.dumps(witness, indent=2) + "\n", encoding="utf-8")
            row["objective"] = witness["objective"]
            row["output"] = output.relative_to(ROOT).as_posix()
        else:
            row["error"] = row["error"] or "no optimal, value-valid HiGHS witness"
        rows.append(row)
        print(json.dumps(row), flush=True)

    metadata = {
        "suite": "miplib2017-stage2",
        "synthetic": False,
        "purpose": "Phase 1 debug-solution witnesses",
        "commit": commit(),
        "working_tree_must_be_clean": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "highs": highspy.Highs().version(),
        "threads": 1,
        "time_limit_seconds": args.time_limit,
        "started": started,
        "instances": rows,
        "missing_or_failed": missing
        + [row for row in rows if row["status"] != "OPTIMAL" or not row["value_valid"]],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return 2 if metadata["missing_or_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
