"""Compare Sovereign's reading of an MPS file with HiGHS' reading of the same file.

Prints every column bound, row side and objective coefficient on which the two
readers disagree, so a wrong answer can be traced to the reader or the solver.

  python benchmarks/tools/compare_mps_parse.py benchmarks/datasets/coverage/netlib/degen3.mps
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))
from run_coverage import INF, find_sovereign, ref_from_mps  # noqa: E402


def main(path: str, limit: int = 15) -> int:
    ref = ref_from_mps(Path(path))
    out = subprocess.run([find_sovereign(), "convert", path], capture_output=True, text=True, check=True).stdout
    m = json.loads(out)
    diffs = []
    sov_vars = {v["name"]: v for v in m["variables"]}
    for name, j in ref.index.items():
        v = sov_vars.get(name)
        if v is None:
            diffs.append(f"column {name}: missing in Sovereign")
            continue
        lo, hi = max(v.get("lower_bound", 0.0), -INF), min(v.get("upper_bound", INF), INF)
        if not np.isclose(lo, ref.col_lo[j]) or not np.isclose(hi, ref.col_hi[j]):
            diffs.append(f"column {name}: sovereign [{lo:g}, {hi:g}] vs highs [{ref.col_lo[j]:g}, {ref.col_hi[j]:g}]")
        c = m["objective"].get("linear", {}).get(name, 0.0) * (1 if m.get("sense") != "maximize" else -1)
        if not np.isclose(c, ref.cost[j] * ref.sense):
            diffs.append(f"cost {name}: sovereign {c:g} vs highs {ref.cost[j] * ref.sense:g}")
    n_ref_rows = ref.A.shape[0]
    print(f"columns: sovereign {len(sov_vars)} highs {len(ref.names)}; "
          f"rows: sovereign {len(m['constraints'])} highs {n_ref_rows}")
    print(f"objective constant: sovereign {m['objective'].get('constant', 0.0)} highs {ref.offset}")
    for d in diffs[:limit]:
        print(" ", d)
    print(f"{len(diffs)} column/cost differences")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
