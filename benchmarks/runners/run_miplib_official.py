"""Convert and benchmark official MIPLIB instances vs HiGHS.

Reports both:
  - full MILP solves (may TIMEOUT / node-limit on harder instances)
  - LP relaxations of the same official files (converter + IPM credibility)

Synthetic multi-knapsack / set-partition models under datasets/miplib/*.json
are NOT official MIPLIB IDs — those stay labeled separately in EVIDENCE.md.
"""

from __future__ import annotations

import copy
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "runners"))
from run_benchmarks import find_sovereign, obj_close, run_highs_on_json  # noqa: E402

OFFICIAL = ROOT / "benchmarks" / "datasets" / "miplib" / "official"

# Modest official MIPLIB 2017 instances (from miplib.zib.de WebData/instances/).
MILP_CHOSEN = ["flugpl", "gt2", "b-ball", "pk1", "gen-ip016"]
LP_RELAX_CHOSEN = ["flugpl", "gt2", "pk1", "b-ball", "gen-ip016"]
MILP_TIMEOUT_S = 30.0
MILP_MAX_NODES = 2000


def ensure_json(name: str) -> Path:
    gz = OFFICIAL / f"{name}.mps.gz"
    mps = OFFICIAL / f"{name}.mps"
    js = OFFICIAL / f"{name}.json"
    if not mps.exists():
        if not gz.exists():
            raise FileNotFoundError(f"Missing {gz}")
        mps.write_bytes(gzip.decompress(gz.read_bytes()))
    if not js.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks" / "tools" / "mps_to_json.py"),
                str(mps),
                "-o",
                str(js),
            ],
            check=True,
        )
    return js


def clean_env(lp_algo: str = "ipm", branch: str = "most_fractional", max_nodes: int | None = None) -> dict:
    env = os.environ.copy()
    for k in [k for k in env if k.startswith("SOVEREIGN_")]:
        env.pop(k)
    env["SOVEREIGN_LP_ALGORITHM"] = lp_algo
    env["SOVEREIGN_BRANCH_RULE"] = branch
    env["SOVEREIGN_PARALLEL_WORKERS"] = "1"
    if max_nodes is not None:
        env["SOVEREIGN_MAX_NODES"] = str(max_nodes)
    return env


def run_milp(binp: str, name: str) -> dict:
    js = ensure_json(name)
    m = json.loads(js.read_text(encoding="utf-8"))
    print(
        f"=== MILP {name} vars={len(m['variables'])} cons={len(m['constraints'])}",
        flush=True,
    )
    h = run_highs_on_json(js, timeout=30) or {}
    print(
        f"  HiGHS {h.get('status')} obj={h.get('objective')} t={h.get('runtime_s')}",
        flush=True,
    )
    env = clean_env(lp_algo="ipm", branch="most_fractional", max_nodes=MILP_MAX_NODES)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            [binp, "solve", str(js)],
            capture_output=True,
            text=True,
            timeout=MILP_TIMEOUT_S,
            env=env,
        )
        dt = time.perf_counter() - t0
        p = json.loads(r.stdout)
        match = obj_close(p.get("objective_value"), h.get("objective"))
        print(
            f"  Ours {p.get('status')} obj={p.get('objective_value')} "
            f"nodes={p.get('nodes')} t={dt:.3f}s match={match}",
            flush=True,
        )
        return {
            "instance": name,
            "source": "MIPLIB 2017 (official)",
            "kind": "MILP",
            "n_vars": len(m["variables"]),
            "n_cons": len(m["constraints"]),
            "highs_status": h.get("status"),
            "highs_obj": h.get("objective"),
            "highs_s": h.get("runtime_s"),
            "ours_status": p.get("status"),
            "ours_obj": p.get("objective_value"),
            "ours_nodes": p.get("nodes"),
            "ours_s": round(dt, 6),
            "match": match,
            "lp_algo": "ipm",
            "branch": "most_fractional",
            "max_nodes": MILP_MAX_NODES,
        }
    except subprocess.TimeoutExpired:
        print(f"  Ours TIMEOUT@{int(MILP_TIMEOUT_S)}s", flush=True)
        return {
            "instance": name,
            "source": "MIPLIB 2017 (official)",
            "kind": "MILP",
            "n_vars": len(m["variables"]),
            "n_cons": len(m["constraints"]),
            "highs_status": h.get("status"),
            "highs_obj": h.get("objective"),
            "highs_s": h.get("runtime_s"),
            "ours_status": "TIMEOUT",
            "ours_obj": None,
            "ours_nodes": None,
            "ours_s": MILP_TIMEOUT_S,
            "match": None,
            "lp_algo": "ipm",
            "branch": "most_fractional",
            "max_nodes": MILP_MAX_NODES,
        }


def run_lp_relax(binp: str, name: str) -> dict:
    js = ensure_json(name)
    m = json.loads(js.read_text(encoding="utf-8"))
    lp = copy.deepcopy(m)
    lp["problem_type"] = "LP"
    for v in lp["variables"]:
        v["type"] = "continuous"
    tmp = ROOT / "benchmarks" / "reports" / f"_tmp_{name}_lp.json"
    tmp.write_text(json.dumps(lp), encoding="utf-8")
    print(f"=== LP-relax {name}", flush=True)
    h = run_highs_on_json(tmp, timeout=30) or {}
    env = clean_env(lp_algo="ipm")
    t0 = time.perf_counter()
    r = subprocess.run(
        [binp, "solve", str(tmp)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    dt = time.perf_counter() - t0
    p = json.loads(r.stdout)
    match = obj_close(p.get("objective_value"), h.get("objective"))
    print(
        f"  IPM {p.get('status')} obj={p.get('objective_value')} "
        f"H={h.get('objective')} match={match} t={dt:.3f}s",
        flush=True,
    )
    return {
        "instance": name,
        "source": "MIPLIB 2017 (official)",
        "kind": "LP_relaxation",
        "n_vars": len(m["variables"]),
        "n_cons": len(m["constraints"]),
        "highs_status": h.get("status"),
        "highs_obj": h.get("objective"),
        "highs_s": h.get("runtime_s"),
        "ours_status": p.get("status"),
        "ours_obj": p.get("objective_value"),
        "ours_nodes": 0,
        "ours_s": round(dt, 6),
        "match": match,
        "lp_algo": "ipm",
    }


def main() -> int:
    binp = find_sovereign()
    milp_rows = [run_milp(binp, n) for n in MILP_CHOSEN]
    lp_rows = [run_lp_relax(binp, n) for n in LP_RELAX_CHOSEN]
    out = ROOT / "benchmarks" / "reports" / "miplib_official.json"
    payload = {
        "note": (
            "Official MIPLIB 2017 instances downloaded from miplib.zib.de "
            "(WebData/instances/<name>.mps.gz), converted with mps_to_json "
            "exactly like Netlib AFIRO. Synthetic multi-knapsack / set-partition "
            "files under datasets/miplib/ are NOT official MIPLIB IDs."
        ),
        "milp": milp_rows,
        "lp_relaxations": lp_rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
