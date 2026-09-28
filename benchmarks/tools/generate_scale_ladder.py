"""Generate the scale-ladder models: structured LPs and convex QPs from 100k to 1M variables.

The LPs are the multi-period planning staircase of generate_sparse_scale.py and
the QPs its sector portfolio, scaled up. Files are hundreds of MB at 1M
variables, so they are written to benchmarks/datasets/ladder/ (not committed)
with a sidecar <name>.meta.json holding dimensions and the SHA-256 that the
scale-ladder report cites. Rerunning this script reproduces them byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.tools.generate_sparse_scale import sector_portfolio, staircase  # noqa: E402

OUT = ROOT / "benchmarks" / "datasets" / "ladder"

MODELS = {
    "lp_staircase_100k": lambda: staircase(100, 500),
    "lp_staircase_1m": lambda: staircase(500, 1000),
    "lp_staircase_1m_rows": lambda: staircase(1000, 1000),
    "qp_portfolio_100k": lambda: sector_portfolio(100_000, 50),
    "qp_portfolio_1m": lambda: sector_portfolio(1_000_000, 50),
}


def generate(name: str) -> dict:
    model = MODELS[name]()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    data = json.dumps(model, separators=(",", ":")).encode("utf-8")
    path.write_bytes(data)
    meta = {
        "name": name,
        "problem_type": model["problem_type"],
        "variables": len(model["variables"]),
        "constraints": len(model["constraints"]),
        "nonzeros": sum(len(row["linear"]) for row in model["constraints"]),
        "quadratic_terms": sum(len(row) for row in model["objective"].get("quadratic", {}).values()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "generator": "benchmarks/tools/generate_scale_ladder.py",
    }
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=list(MODELS), help="models to generate (default: all)")
    for name in parser.parse_args().names:
        meta = generate(name)
        print(f"wrote {name}: vars={meta['variables']} cons={meta['constraints']} "
              f"nnz={meta['nonzeros']} {meta['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
