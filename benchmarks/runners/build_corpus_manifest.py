"""Write/check an immutable-input manifest for the current SIH corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from api.datasets import CASES, dataset  # noqa: E402

REPORT = ROOT / "benchmarks/reports/sih-online.json"
OUTPUT = ROOT / "benchmarks/reports/corpus-manifest.json"


def build_manifest():
    previous = json.loads(REPORT.read_text(encoding="utf-8"))
    references = {}
    for row in previous["rows"]:
        name = row["dataset"]
        ref = row["reference"]
        if name in references and references[name] != ref:
            raise ValueError(f"Conflicting reference results for {name}")
        references[name] = ref
    entries = []
    for name, suite, relative, source in CASES:
        current = dataset(name)
        ref = references.get(name)
        entries.append({
            "id": name, "suite": suite, "path": relative.replace("\\", "/"),
            "sha256": current["sha256"], "format": current["modelFormat"],
            "shape": current["shape"], "source_page_or_origin": source,
            "license_status": "not yet recorded",
            "reference_solver": "HiGHS 1.15.1" if ref else None,
            "reference_time_limit_seconds": previous["time_limit_seconds"] if ref else None,
            "reference_status": ref["status"] if ref else None,
            "reference_objective": ref["objective"] if ref else None,
            "reference_mip_gap": ref["mip_gap"] if ref else None,
        })
    return {"schema_version": 1,
            "description": "Current repository corpus; reference values are measured HiGHS runs, not certified known optima when HiGHS timed out.",
            "reference_report": "benchmarks/reports/sih-online.json",
            "reference_report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
            "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if files or metadata differ from the saved manifest")
    args = parser.parse_args()
    fresh = build_manifest()
    if args.check:
        saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if saved != fresh:
            raise SystemExit("Corpus manifest differs from current inputs or source metadata.")
        print(f"Verified {len(fresh['entries'])} corpus entries and SHA-256 hashes.")
    else:
        OUTPUT.write_text(json.dumps(fresh, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote {len(fresh['entries'])} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
