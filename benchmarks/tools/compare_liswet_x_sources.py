"""Compare parsed-c and CUTEst-c LISWET1 primal outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-x", type=Path, required=True)
    parser.add_argument("--cutest-x", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parsed = np.loadtxt(args.parsed_x, dtype=np.float64).reshape(-1)
    cutest = np.loadtxt(args.cutest_x, dtype=np.float64).reshape(-1)
    if len(parsed) != len(cutest):
        raise SystemExit(f"length mismatch: {len(parsed)} != {len(cutest)}")
    difference = cutest - parsed
    index = int(np.argmax(np.abs(difference)))
    report = {
        "variables": len(parsed),
        "x_difference_inf": float(np.max(np.abs(difference))),
        "largest_index_zero_based": index,
        "largest_index_one_based": index + 1,
        "x_parsed": float(parsed[index]),
        "x_cutest": float(cutest[index]),
        "difference": float(difference[index]),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
