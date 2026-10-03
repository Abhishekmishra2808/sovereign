"""Compare parsed and PyCUTEst-derived LISWET1 c vectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", type=Path, required=True)
    parser.add_argument("--cutest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    parsed = np.loadtxt(args.parsed, dtype=np.float64).reshape(-1)
    cutest = np.loadtxt(args.cutest, dtype=np.float64).reshape(-1)
    if len(parsed) != len(cutest):
        raise SystemExit(f"length mismatch: {len(parsed)} != {len(cutest)}")
    difference = cutest - parsed
    order = np.argsort(-np.abs(difference))[:args.top]
    report = {
        "variables": len(parsed),
        "difference_inf": float(np.max(np.abs(difference))),
        "difference_l2": float(np.linalg.norm(difference)),
        "nonzero_count": int(np.count_nonzero(difference)),
        "largest_indices_zero_based": [int(i) for i in order],
        "largest_indices_one_based": [int(i + 1) for i in order],
        "largest": [
            {
                "index_zero_based": int(i),
                "index_one_based": int(i + 1),
                "c_parsed": float(parsed[i]),
                "c_cutest": float(cutest[i]),
                "difference": float(difference[i]),
                "absolute_difference": float(abs(difference[i])),
            }
            for i in order
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
