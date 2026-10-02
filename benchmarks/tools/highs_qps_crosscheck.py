"""Re-run every Maros-Meszaros QP that HiGHS missed, with HiGHS reading the original QPS file.

The coverage harness hands HiGHS the model converted from the MAT files, exactly as
Sovereign gets it. This rules out the conversion as the cause of a HiGHS miss: each
instance whose HiGHS outcome in reports/coverage/qp.jsonl is not SOLVED is solved again
from the upstream QPS file with default options, one thread and a 300 s limit. Four
instances run at a time, pinned to cores 8-11 so a concurrent benchmark is not disturbed.

    python benchmarks/tools/highs_qps_crosscheck.py

Writes reports/coverage/qp_highs_qps_crosscheck.jsonl (resumable), which
coverage_report.py renders as a table in COVERAGE.md.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
QPS_DIR = ROOT / "datasets" / "coverage" / "qps_original"
RESULTS = ROOT / "reports" / "coverage" / "qp.jsonl"
OUT = ROOT / "reports" / "coverage" / "qp_highs_qps_crosscheck.jsonl"
URL = "https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/QPS_Files/{}.QPS"
CORES = [[8], [9], [10], [11]]
LIMIT = 300.0
# HiGHS can overrun its own time limit (BOYD1); past this the run is killed.
KILL_AFTER = 420

WORKER = r'''
import sys, time, json, highspy
h = highspy.Highs()
h.setOptionValue("output_flag", False)
h.setOptionValue("threads", 1)
h.setOptionValue("time_limit", float(sys.argv[2]))
st = h.readModel(sys.argv[1])
t = time.perf_counter()
h.run()
print(json.dumps({"read": str(st), "status": h.modelStatusToString(h.getModelStatus()),
                  "obj": h.getInfo().objective_function_value, "time": time.perf_counter() - t}))
'''

lock = threading.Lock()


def run_one(slot: int, row: dict) -> dict:
    name = row["name"]
    # HiGHS picks the reader by extension and does not recognise ".QPS".
    path = QPS_DIR / f"{name}.mps"
    if not path.exists():
        urllib.request.urlretrieve(URL.format(name), path)
    p = subprocess.Popen([sys.executable, "-c", WORKER, str(path), str(LIMIT)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    psutil.Process(p.pid).cpu_affinity(CORES[slot % len(CORES)])
    out = ""
    try:
        out = p.communicate(timeout=KILL_AFTER)[0]
        res = json.loads(out.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        res = {"status": f"Killed at {KILL_AFTER} s (time limit {LIMIT:g} s not honoured)"}
    except (ValueError, IndexError):
        res = {"error": out[-300:]}
    res.update(name=name, harness=row["highs"].get("outcome"), harness_obj=row["highs"].get("objective"),
               reference=(row.get("reference") or {}).get("value"))
    print(json.dumps(res), flush=True)
    with lock, OUT.open("a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def main() -> None:
    QPS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    done = set()
    if OUT.exists():
        done = {json.loads(line)["name"] for line in OUT.read_text().splitlines() if line.strip()}
    todo = [r for r in rows if r["highs"].get("outcome") != "SOLVED" and r["name"] not in done]
    print(f"{len(todo)} HiGHS misses to re-run ({len(done)} already done)", flush=True)
    with cf.ThreadPoolExecutor(len(CORES)) as pool:
        list(pool.map(run_one, range(len(todo)), todo))


if __name__ == "__main__":
    main()
