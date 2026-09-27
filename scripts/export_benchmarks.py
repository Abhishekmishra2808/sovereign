"""Bundle the recorded CSV verbatim in a browser-friendly schema; no invented runs."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "benchmarks/reports/latest.csv"


def numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def main():
    rows = []
    with source.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "suite": row["suite"], "problem": row["problem"],
                "solver": row["solver"], "status": row["status"],
                "objective": numeric(row.get("objective")),
                "runtimeSeconds": numeric(row.get("runtime_s")),
                "gap": numeric(row.get("gap")), "nodes": numeric(row.get("nodes")),
                "iterations": numeric(row.get("iterations")), "milpStats": row.get("milp_stats") or None,
            })
    target = ROOT.parent / "sovereign-frontend/public/data/benchmarks.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"source": "benchmarks/reports/latest.csv",
                                 "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                 "rows": rows}, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Bundled {len(rows)} recorded benchmark rows")


if __name__ == "__main__":
    main()
