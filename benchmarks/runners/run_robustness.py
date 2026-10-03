"""Robustness evidence: measured hardness of every benchmark model, and a feature ablation.

    python benchmarks/runners/run_robustness.py --measure   # properties.jsonl (HiGHS + SciPy only)
    python benchmarks/runners/run_robustness.py --ablate    # ablation.jsonl (Sovereign, features off)
    python benchmarks/runners/run_robustness.py --report    # benchmarks/reports/ROBUSTNESS.md

--measure characterises each model independently of Sovereign: at the optimal basis HiGHS
returns, the share of basic variables sitting on a bound (primal degeneracy), the share of
nonbasic variables with a zero reduced cost (dual degeneracy) and a 1-norm condition estimate
of the basis matrix; for MILPs, the gap between the LP relaxation and the published optimum.

--ablate re-solves the suites with Sovereign's robustness features switched off one at a time
(SOVEREIGN_DUAL_DISABLE, SOVEREIGN_QP_DISABLE, SOVEREIGN_PRESOLVE, ...), scored by the same
independent checker as the coverage benchmark. Four single-threaded runs share the machine,
so the times are for comparing configurations, not for comparing with HiGHS.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_coverage as rc  # noqa: E402

OUT_DIR = rc.ROOT / "benchmarks" / "reports" / "robustness"
PROPS = OUT_DIR / "properties.jsonl"
ABLATION = OUT_DIR / "ablation.jsonl"
MD = rc.ROOT / "benchmarks" / "reports" / "ROBUSTNESS.md"
LANES = [[0, 1], [2, 3], [4, 5], [6, 7]]

DUAL = {"SOVEREIGN_LP_ALGORITHM": "dual"}
CONFIGS = {
    "netlib": [
        ("Full solver (default)", {}),
        ("Dual simplex only (no fallback)", DUAL),
        ("  minus presolve", {**DUAL, "SOVEREIGN_PRESOLVE": "0"}),
        ("  minus scaling", {**DUAL, "SOVEREIGN_DUAL_DISABLE": "scaling"}),
        ("  minus Harris ratio test", {**DUAL, "SOVEREIGN_DUAL_DISABLE": "harris"}),
        ("  minus steepest-edge pricing", {**DUAL, "SOVEREIGN_DUAL_PRICING": "dantzig"}),
        ("  minus bound widening", {**DUAL, "SOVEREIGN_DUAL_DISABLE": "widening"}),
        ("Textbook dual simplex (all of the above off)",
         {**DUAL, "SOVEREIGN_PRESOLVE": "0", "SOVEREIGN_DUAL_PRICING": "dantzig",
          "SOVEREIGN_DUAL_DISABLE": "scaling,harris,widening"}),
    ],
    "qp": [
        ("Full solver (default)", {}),
        ("  minus iterative refinement", {"SOVEREIGN_QP_DISABLE": "refinement"}),
        ("  minus common primal/dual step", {"SOVEREIGN_QP_DISABLE": "common_step"}),
        ("  minus Mehrotra-start retry", {"SOVEREIGN_QP_DISABLE": "retry"}),
        ("  minus Frank-Wolfe verification guard", {"SOVEREIGN_QP_DISABLE": "fw_guard"}),
        ("Textbook IPM (all of the above off)",
         {"SOVEREIGN_QP_DISABLE": "refinement,common_step,retry,fw_guard"}),
    ],
    "milp": [
        ("Full solver (default)", {}),
        ("  minus cutting planes", {"SOVEREIGN_ENABLE_CUTS": "0"}),
        ("  minus primal heuristics", {"SOVEREIGN_ENABLE_HEURISTICS": "0"}),
        ("  minus presolve", {"SOVEREIGN_PRESOLVE": "0"}),
    ],
}
LIMITS = {"netlib": 60.0, "qp": 60.0, "milp": 300.0}


# ---------------------------------------------------------------- measure
def basis_condition(A, col_basic, row_basic) -> float | None:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    m = A.shape[0]
    B = sp.hstack([A[:, col_basic], sp.identity(m, format="csc")[:, row_basic]]).tocsc().astype(float)
    if B.shape[1] != m or m == 0:
        return None
    try:
        lu = spla.splu(B)
    except RuntimeError:
        return math.inf
    inv = spla.LinearOperator((m, m), matvec=lu.solve, rmatvec=lambda v: lu.solve(v, trans="T"), dtype=float)
    return float(abs(B).sum(axis=0).max() * spla.onenormest(inv))


def coef_range(values) -> float | None:
    v = np.abs(np.asarray(values, float))
    v = v[v > 0]
    return float(v.max() / v.min()) if len(v) else None


def measure_lp(path: Path) -> dict:
    import highspy
    import scipy.sparse as sp
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", 300.0)
    h.setOptionValue("large_matrix_value", rc.LARGE_MATRIX_VALUE)
    h.readModel(str(path))
    lp = h.getLp()
    a = lp.a_matrix_
    A = sp.csc_matrix((np.asarray(a.value_), np.asarray(a.index_), np.asarray(a.start_)),
                      shape=(lp.num_row_, lp.num_col_))
    out = {"coef_range": coef_range(a.value_)}
    h.run()
    if h.modelStatusToString(h.getModelStatus()) != "Optimal":
        return out
    sol, basis = h.getSolution(), h.getBasis()
    lo = np.concatenate([lp.col_lower_, lp.row_lower_])
    hi = np.concatenate([lp.col_upper_, lp.row_upper_])
    val = np.concatenate([sol.col_value, sol.row_value])
    dual = np.concatenate([sol.col_dual, sol.row_dual])
    status = list(basis.col_status) + list(basis.row_status)
    basic = np.array([s == highspy.HighsBasisStatus.kBasic for s in status])

    def near(b):
        return np.isfinite(b) & (np.abs(b) < 1e20) & (np.abs(val - b) <= 1e-9 * (1 + np.abs(b)))
    on_bound = near(lo) | near(hi)
    free_nonbasic = ~basic & (lo != hi)
    m = lp.num_row_
    out.update(
        primal_degeneracy=float((basic & on_bound).sum() / max(m, 1)),
        dual_degeneracy=float((free_nonbasic & (np.abs(dual) <= 1e-9)).sum() / max(free_nonbasic.sum(), 1)),
        basis_condition=basis_condition(A, basic[:lp.num_col_], basic[lp.num_col_:]),
    )
    return out


def measure_mip(path: Path, reference: dict | None) -> dict:
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", 120.0)
    h.setOptionValue("large_matrix_value", rc.LARGE_MATRIX_VALUE)
    h.setOptionValue("solve_relaxation", True)
    h.readModel(str(path))
    out = {"coef_range": coef_range(h.getLp().a_matrix_.value_)}
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    out["relaxation_status"] = status
    if status == "Optimal":
        z_lp = h.getInfo().objective_function_value
        out["relaxation_objective"] = z_lp
        if reference and reference.get("value") is not None:
            z = reference["value"]
            out["root_gap"] = abs(z - z_lp) / max(abs(z), abs(z_lp), 1e-9)
    return out


def measure_qp(entry: dict) -> dict:
    c = rc.qp_canonical(rc.ROOT / entry["file"])
    return {"coef_range": coef_range(c["A"].data) if c["A"].nnz else None,
            "hessian_range": coef_range(c["P"].data) if c["P"].nnz else None}


def measure(corpus: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {(r["suite"], r["name"]) for r in load_jsonl(PROPS)}
    for suite in rc.SUITES:
        for e in corpus.get(suite, []):
            if (suite, e["name"]) in done:
                continue
            try:
                if suite == "netlib":
                    p = measure_lp(rc.ROOT / e["file"])
                elif suite == "qp":
                    p = measure_qp(e)
                else:
                    p = measure_mip(rc.ROOT / e["file"], e.get("reference"))
            except Exception as exc:  # one unreadable model must not stop the sweep
                p = {"error": str(exc)[:200]}
            row = {"suite": suite, "name": e["name"], **p}
            with PROPS.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


# ---------------------------------------------------------------- ablate
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def coverage_rows(suite: str) -> dict:
    return {r["name"]: r for r in load_jsonl(rc.OUT_DIR / f"{suite}.jsonl")}


def ablation_entries(corpus: dict, suite: str) -> list[dict]:
    if suite != "milp":
        return list(corpus[suite])
    # Only models the full solver closes within the limit can show what a feature costs.
    solved = {n for s in ("miplib", "infeasible") for n, r in coverage_rows(s).items()
              if r["sovereign"].get("outcome") == "SOLVED"}
    return [e for s in ("miplib", "infeasible") for e in corpus[s] if e["name"] in solved]


def ablate(corpus: dict, suites: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    binary = rc.find_sovereign()
    done = {(r["suite"], r["config"], r["name"]) for r in load_jsonl(ABLATION)}
    lock = threading.Lock()
    for suite in suites:
        entries = ablation_entries(corpus, suite)
        disproved = {}
        for s in ("netlib", "qp", "miplib"):
            for n, r in coverage_rows(s).items():
                if r.get("reference_disproved"):
                    disproved[n] = r["reference_disproved"]["best_verified"]
        refs = {}
        for e in entries:
            if e["kind"] == "QP":
                refs[e["name"]] = rc.ensure_qp_json(e)
            else:
                refs[e["name"]] = (rc.ROOT / e["file"], rc.ref_from_mps(rc.ROOT / e["file"]))
        jobs = [(cfg, env, e) for cfg, env in CONFIGS[suite] for e in entries
                if (suite, cfg, e["name"]) not in done]
        print(f"[{suite}] {len(jobs)} runs", flush=True)
        limit = LIMITS[suite]

        def one(k_job):
            k, (cfg, env, e) = k_job
            model, ref = refs[e["name"]]
            entry = dict(e)
            if e["name"] in disproved:
                entry["reference"] = {**(e.get("reference") or {}), "value": disproved[e["name"]]}
            res = rc.run_sovereign(binary, model, e["kind"], limit, LANES[k % len(LANES)], env)
            res = rc.score(entry, e["kind"], res, ref, limit)
            row = {"suite": suite, "config": cfg, "name": e["name"], "outcome": res["outcome"],
                   "status": res.get("status"), "time": res.get("time"), "why": res.get("why"),
                   "max_row_violation": (res.get("check") or {}).get("max_row_violation")}
            with lock:
                with ABLATION.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{suite}] {cfg.strip():<45} {e['name']:<20} {row['outcome']:<8} {row['time'] or 0:7.2f}s",
                      flush=True)

        with cf.ThreadPoolExecutor(len(LANES)) as pool:
            list(pool.map(one, enumerate(jobs)))


# ---------------------------------------------------------------- report
def sgm(times, shift):
    return math.exp(sum(math.log(t + shift) for t in times) / len(times)) - shift if times else float("nan")


def fmt(v, spec=".1e"):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def outcome_cell(r: dict) -> str:
    o = r.get("outcome", "?")
    return f"{o} {r.get('time', 0):.2f} s" if o == "SOLVED" else o


def report() -> None:
    props = {(r["suite"], r["name"]): r for r in load_jsonl(PROPS)}
    cov = {s: coverage_rows(s) for s in rc.SUITES}
    ab = load_jsonl(ABLATION)
    L = ["# Numerical robustness: evidence", "",
         "Generated by `benchmarks/runners/run_robustness.py`. The problem statement asks for a clear",
         "demonstration on degeneracy, weak LP relaxations and ill-conditioned constraint matrices. This",
         "report measures those properties on every public benchmark model *independently of Sovereign*",
         "(HiGHS optimal basis plus SciPy), then lists how Sovereign and HiGHS fared on the hardest ones",
         "(results from `COVERAGE.md`: one thread each, 300 s limit, every answer independently verified).",
         "The last section switches Sovereign's robustness features off one at a time to show what each",
         "one is responsible for.", ""]

    def solver_counts(rows):
        s = sum(r["sovereign"].get("outcome") == "SOLVED" for r in rows)
        h = sum(r["highs"].get("outcome") == "SOLVED" for r in rows)
        ws = sum(r["sovereign"].get("outcome") == "WRONG" for r in rows)
        wh = sum(r["highs"].get("outcome") == "WRONG" for r in rows)
        return s, h, ws, wh

    def hard_table(suite, key, title, intro, threshold, cols, top=25):
        rows = [(props[(suite, n)], r) for n, r in cov[suite].items()
                if (suite, n) in props and props[(suite, n)].get(key) is not None]
        rows.sort(key=lambda pr: -pr[0][key])
        hard = [pr for pr in rows if pr[0][key] >= threshold]
        s, h, ws, wh = solver_counts([r for _, r in hard])
        out = [f"### {title}", "", *intro, "",
               f"**{len(hard)} models meet the threshold. Sovereign solved {s}, HiGHS {h}; wrong answers: "
               f"Sovereign {ws}, HiGHS {wh}.** The {min(top, len(hard))} hardest:", "",
               "| Instance | Rows×Cols | " + " | ".join(c[0] for c in cols) + " | Sovereign | HiGHS |",
               "|---|---|" + "---:|" * len(cols) + "---|---|"]
        for p, r in hard[:top]:
            out.append(f"| {r['name']} | {r.get('rows')}×{r.get('cols')} | " +
                       " | ".join(c[1](p) for c in cols) +
                       f" | {outcome_cell(r['sovereign'])} | {outcome_cell(r['highs'])} |")
        return out + [""]

    pct = lambda k: (lambda p: f"{100 * p[k]:.0f}%")  # noqa: E731
    sci = lambda k: (lambda p: fmt(p.get(k)))  # noqa: E731
    L += ["## 1. Degeneracy (Netlib LP)", ""]
    L += hard_table("netlib", "primal_degeneracy", "Primal degeneracy at the optimum",
                    ["Share of basic variables sitting exactly on a bound in the optimal basis. Degenerate",
                     "vertices make the simplex method stall or cycle (zero-length pivots); threshold 20%."],
                    0.20, [("Primal degenerate", pct("primal_degeneracy")), ("Dual degenerate", pct("dual_degeneracy"))])
    L += hard_table("netlib", "dual_degeneracy", "Dual degeneracy (multiple optima)",
                    ["Share of nonbasic variables with a zero reduced cost: the optimum is not unique and",
                     "reduced costs carry no direction; threshold 30%."],
                    0.30, [("Dual degenerate", pct("dual_degeneracy")), ("Primal degenerate", pct("primal_degeneracy"))],
                    top=15)

    L += ["## 2. Ill-conditioned constraint matrices", ""]
    L += hard_table("netlib", "basis_condition", "Condition of the optimal basis (Netlib LP)",
                    ["1-norm condition estimate of the optimal basis matrix B (‖B‖₁·‖B⁻¹‖₁, SciPy `onenormest`).",
                     "Above about 1e8 a double-precision simplex loses half its digits in every solve; threshold 1e8."],
                    1e8, [("cond₁(B)", sci("basis_condition")), ("Coefficient range", sci("coef_range"))])
    L += hard_table("qp", "coef_range", "Coefficient range (Maros-Meszaros QP)",
                    ["max|a_ij| / min|a_ij| over the constraint matrix; threshold 1e6."],
                    1e6, [("A range", sci("coef_range")), ("Hessian range", sci("hessian_range"))], top=20)
    L += hard_table("miplib", "coef_range", "Coefficient range (MIPLIB 2017)",
                    ["max|a_ij| / min|a_ij|; big-M constraints show up here. Threshold 1e6."],
                    1e6, [("A range", sci("coef_range"))], top=15)

    L += ["## 3. Weak LP relaxations (MILP)", ""]
    L += hard_table("miplib", "root_gap", "Gap between the LP relaxation and the optimum (MIPLIB 2017)",
                    ["|z* − z_LP| / max(|z*|, |z_LP|), with z* the published optimum and z_LP the LP relaxation",
                     "(HiGHS, no cuts). Branch-and-bound has to close this gap; threshold 10%."],
                    0.10, [("Root gap", lambda p: f"{100 * p['root_gap']:.1f}%")])
    inf_rows = [(props.get(("infeasible", n), {}), r) for n, r in cov["infeasible"].items()]
    weak = [(p, r) for p, r in inf_rows if p.get("relaxation_status") == "Optimal"]
    s, h, ws, wh = solver_counts([r for _, r in weak])
    L += ["### Integer-infeasible models whose LP relaxation is feasible (MIPLIB infeasible set)", "",
          "The weakest relaxation possible: the LP relaxation has a solution but no integer point exists, so",
          "infeasibility can only be proven by search and cuts.", "",
          f"**{len(weak)} of {len(inf_rows)} models. Sovereign proved {s} infeasible, HiGHS {h}; wrong: "
          f"Sovereign {ws}, HiGHS {wh}.**", "",
          "| Instance | Rows×Cols | Sovereign | HiGHS |", "|---|---|---|---|"]
    L += [f"| {r['name']} | {r.get('rows')}×{r.get('cols')} | {outcome_cell(r['sovereign'])} | "
          f"{outcome_cell(r['highs'])} |" for _, r in sorted(weak, key=lambda pr: pr[1]["name"])] + [""]

    L += ["## 4. Ablation: what each robustness feature is responsible for", "",
          "Each suite re-solved with features switched off (environment switches listed in the script),",
          "scored by the same independent checker. Four single-threaded runs share the machine, so compare",
          "times between rows only. A feature that only rescues a few models still matters: those are",
          "exactly the degenerate and ill-conditioned ones above.", ""]
    for suite, title in (("netlib", "Netlib LP (91 models, 60 s limit)"),
                         ("qp", "Maros-Meszaros QP (138 models, 60 s limit)"),
                         ("milp", "MILP: MIPLIB models the full solver closes (300 s limit)")):
        rows = [r for r in ab if r["suite"] == suite]
        if not rows:
            continue
        full = {r["name"]: r for r in rows if r["config"] == CONFIGS[suite][0][0]}
        L += [f"### {title}", "",
              "| Configuration | Solved | Wrong | Timeout | Failed | SGM time (s) | Lost vs. full solver |",
              "|---|---:|---:|---:|---:|---:|---|"]
        for cfg, _ in CONFIGS[suite]:
            rs = [r for r in rows if r["config"] == cfg]
            if not rs:
                continue
            cnt = {o: sum(r["outcome"] == o for r in rs) for o in ("SOLVED", "WRONG", "TIMEOUT", "FAILED")}
            times = [r["time"] if r["outcome"] == "SOLVED" else LIMITS[suite] for r in rs]
            lost = sorted(r["name"] for r in rs if r["outcome"] != "SOLVED"
                          and full.get(r["name"], {}).get("outcome") == "SOLVED")
            lost_s = ", ".join(lost[:12]) + (f" (+{len(lost) - 12} more)" if len(lost) > 12 else "")
            name = cfg.replace("  minus", "&nbsp;&nbsp;minus")
            L.append(f"| {name} | {cnt['SOLVED']} | {cnt['WRONG']} | {cnt['TIMEOUT']} | {cnt['FAILED']} | "
                     f"{sgm(times, rc.SGM_SHIFT.get(suite, 10.0)):.3g} | {lost_s} |")
        wrong = [r for r in rows if r["outcome"] == "WRONG"]
        if wrong:
            L += ["", "Wrong answers in the ablation (the reason the corresponding feature exists):", ""]
            L += [f"- {r['config'].strip()}: **{r['name']}**, {r['why']}" for r in wrong]
        L.append("")
    MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {MD}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--ablate", nargs="*", choices=list(CONFIGS), help="suites to ablate (default: all)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    corpus = rc.load_corpus()
    if args.measure:
        measure(corpus)
    if args.ablate is not None:
        ablate(corpus, args.ablate or list(CONFIGS))
    if args.report:
        report()


if __name__ == "__main__":
    main()
