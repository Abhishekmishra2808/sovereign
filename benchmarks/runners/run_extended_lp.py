"""Run the downloaded Mittelmann and Kennington LP showcase."""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_coverage as rc

ROOT = rc.ROOT
EXT = ROOT / "benchmarks" / "datasets" / "coverage" / "extended"
OUT = ROOT / "benchmarks" / "reports" / "showcase" / "extended_lp.jsonl"
CORES = [[0, 1], [2, 3], [4, 5], [6, 7]]


def one(item):
    i, path = item
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOVEREIGN_")}
    env.update(SOVEREIGN_PARALLEL_WORKERS="1", SOVEREIGN_DISABLE_CUDA="1",
               SOVEREIGN_TIME_LIMIT="300")
    p = subprocess.Popen(
        [str(rc.find_sovereign()), "solve", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env,
    )
    psutil.Process(p.pid).cpu_affinity(CORES[i % len(CORES)])
    started = time.perf_counter()
    try:
        raw = p.communicate(timeout=360)[0]
        ours = json.loads(raw) if raw.strip() else {
            "status": "ERROR",
            "message": "Sovereign returned no JSON",
        }
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        ours = {"status": "TIMEOUT", "runtime_seconds": 300.0}
    except json.JSONDecodeError:
        ours = {"status": "ERROR", "message": "Sovereign returned malformed JSON"}
    ours_time = float(ours.get("runtime_seconds") or (time.perf_counter() - started))

    # HiGHS reads the original MPS file, with the same coefficient acceptance
    # setting as the main coverage harness.
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", 300.0)
    h.setOptionValue("large_matrix_value", rc.LARGE_MATRIX_VALUE)
    h.readModel(str(path))
    ht = time.perf_counter()
    h.run()
    highs_time = time.perf_counter() - ht
    hs = h.modelStatusToString(h.getModelStatus()).upper().replace(" ", "_")
    result = {
        "name": path.stem,
        "family": path.parent.name,
        "file": str(path.relative_to(ROOT)),
        "sovereign": {"status": ours.get("status"), "time": ours_time,
                      "objective": ours.get("objective_value")},
        "highs": {"status": hs, "time": highs_time,
                  "objective": h.getInfo().objective_function_value},
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    paths = sorted([p for p in (EXT / "mittelmann").glob("*.mps")]
                   + [p for p in (EXT / "kennington").glob("*.mps")])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(len(CORES)) as pool:
        rows = list(pool.map(one, enumerate(paths)))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {OUT}; {len(rows)} instances")


if __name__ == "__main__":
    main()
