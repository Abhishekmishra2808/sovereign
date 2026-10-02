"""Quick run over the problem types the SIH 26119 statement names, one instance list per category.

    python benchmarks/runners/run_showcase.py [--time-limit 120] [--only NAME ...] [--category CAT]

Sovereign only (four single-threaded runs at a time), every answer checked by the coverage
benchmark's independent verifier; HiGHS's result is read from reports/coverage/*.jsonl. The
category of each instance comes from its measured property in reports/robustness/properties.jsonl
(primal degeneracy, optimal-basis condition, LP-relaxation gap). Results go to
reports/showcase/showcase.jsonl, replacing earlier rows for the same instance.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_coverage as rc  # noqa: E402

OUT = rc.ROOT / "benchmarks" / "reports" / "showcase" / "showcase.jsonl"
LANES = [[0, 1], [2, 3], [4, 5], [6, 7]]

SHOWCASE = {
    "degenerate LP": ("netlib", ["cycle", "d6cube", "wood1p", "greenbea", "greenbeb", "degen3", "dfl001"]),
    "ill-conditioned LP": ("netlib", ["pilotnov", "pilot4", "perold", "pilot", "pilot87", "d2q06c", "fit2p"]),
    "ill-conditioned QP": ("qp", ["QPILOTNO", "LISWET1", "CVXQP3_L"]),
    "weak LP relaxation (MILP)": ("miplib", ["pk1", "qiu", "misc07", "fiber", "sp150x300d", "p200x1188c",
                                             "exp-1-500-5-5", "neos17", "enlight_hard", "b-ball", "10teams",
                                             "cap6000", "qnet1", "pg", "markshare_4_0"]),
    "infeasible, feasible relaxation": ("infeasible", ["enlight9", "neos-3135526-osun", "stein45inf"]),
}


def run(entry: dict, limit: float, lane, binary: str) -> dict:
    if entry["kind"] == "QP":
        model, ref = rc.ensure_qp_json(entry)
    else:
        model, ref = rc.ROOT / entry["file"], rc.ref_from_mps(rc.ROOT / entry["file"])
    res = rc.run_sovereign(binary, model, entry["kind"], limit, lane)
    return rc.score(entry, entry["kind"], res, ref, limit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--category", choices=list(SHOWCASE))
    args = ap.parse_args()
    corpus = rc.load_corpus()
    by_name = {(s, e["name"]): e for s in rc.SUITES for e in corpus.get(s, [])}
    disproved, highs = {}, {}
    for s in rc.SUITES:
        path = rc.OUT_DIR / f"{s}.jsonl"
        for line in path.read_text().splitlines() if path.exists() else []:
            r = json.loads(line)
            highs[r["name"]] = r["highs"]
            if r.get("reference_disproved"):
                disproved[r["name"]] = r["reference_disproved"]["best_verified"]
    jobs = []
    for cat, (suite, names) in SHOWCASE.items():
        if args.category and cat != args.category:
            continue
        for n in names:
            if args.only and n not in args.only:
                continue
            e = dict(by_name[(suite, n)])
            if n in disproved:
                e["reference"] = {**(e.get("reference") or {}), "value": disproved[n]}
            jobs.append((cat, e))
    binary = rc.find_sovereign()
    with cf.ThreadPoolExecutor(len(LANES)) as pool:
        futs = [pool.submit(run, e, args.time_limit, LANES[k % len(LANES)], binary) for k, (_, e) in enumerate(jobs)]
        results = [f.result() for f in futs]

    old = []
    if OUT.exists():
        names = {e["name"] for _, e in jobs}
        old = [line for line in OUT.read_text().splitlines() if line.strip() and json.loads(line)["name"] not in names]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (cat, e), res in zip(jobs, results):
        rows.append({"category": cat, "name": e["name"], "outcome": res["outcome"], "status": res.get("status"),
                     "time": res.get("time"), "why": res.get("why"), "message": (res.get("message") or "")[:200],
                     "limit": args.time_limit})
    OUT.write_text("".join(line + "\n" for line in old) + "".join(json.dumps(r) + "\n" for r in rows))

    solved = 0
    for r in rows:
        h = highs.get(r["name"], {})
        hs = f"{h.get('outcome')} {h.get('time', 0):.1f}s" if h else "-"
        solved += r["outcome"] == "SOLVED"
        print(f"{r['category']:<32} {r['name']:<20} {r['outcome']:<8} {r['time'] or 0:8.2f}s   HiGHS {hs}"
              + ("" if r["outcome"] == "SOLVED" else f"   | {r['why'] or r['message'][:90]}"))
    print(f"solved {solved} of {len(rows)} (limit {args.time_limit:g} s)")


if __name__ == "__main__":
    main()
