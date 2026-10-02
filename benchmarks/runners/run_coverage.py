"""Coverage benchmark: Sovereign vs HiGHS on public benchmark libraries.

Suites (download first with benchmarks/tools/fetch_coverage_corpus.py):
  netlib      full feasible Netlib LP set, scored against Netlib's published optima
  miplib      62 official MIPLIB 2017 instances, scored against miplib2017-v37.solu
  infeasible  28 MIPLIB 2017 instances proven infeasible (correct answer: INFEASIBLE)
  qp          138 Maros-Meszaros convex QPs, scored against 00README.QP optima

Protocol
  * Both solvers read the same input file, run single-threaded, with the same
    wall-clock limit, and (MILP) the same relative gap, 1e-6 (Sovereign's
    built-in default; HiGHS is set to match).
  * Reported times are solve times: Sovereign's engine time and the time of
    HiGHS' run() call. Both include presolve; both exclude reading the file.
  * Each returned solution is checked here, independently of both solvers:
    bounds, rows, integrality and objective are recomputed from the original
    model. An answer counts as SOLVED only if it is claimed optimal, passes that
    check, and its recomputed objective agrees with the published reference
    (or, where none exists, is not beaten by the other solver's verified point).
  * Results are appended per instance to reports/coverage/<suite>.jsonl, so a
    long run can be interrupted and resumed.

Usage
  python benchmarks/runners/run_coverage.py --suite netlib --time-limit 300
  python benchmarks/runners/run_coverage.py --report
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "benchmarks" / "datasets" / "coverage"
OUT_DIR = ROOT / "benchmarks" / "reports" / "coverage"
SUITES = ["netlib", "miplib", "infeasible", "qp"]
SUITE_TITLES = {"netlib": "Netlib LP", "miplib": "MIPLIB 2017 (MILP)",
                "infeasible": "MIPLIB 2017 infeasible set", "qp": "Maros-Meszaros QP"}
INF = 1e30
FEAS_TOL = 1e-6          # bounds: relative to 1 + |bound|; rows: to 1 + max(|rhs|, sum|a_ij x_j|)
INT_TOL = 1e-5
EXACT_FEAS_TOL = 1e-9    # lets a single solver's point disprove a published optimum
# HiGHS rejects |a_ij| > 1e15 by default; control30-5-10-4 has entries up to 6e23.
LARGE_MATRIX_VALUE = 1e30
MIP_GAP = 1e-6
# Several optima in the Netlib readme are off in the trailing digits (Koch,
# "The final NETLIB-LP results", Oper. Res. Letters 32, 2004): pilot87's
# published 301.71072827 is 1.3e-6 away from HiGHS' verified 301.71034733.
LP_OBJ_RTOL = 1e-5
QP_OBJ_RTOL = 1e-5
MILP_OBJ_RTOL = 1e-4
SGM_SHIFT = {"netlib": 1.0, "qp": 1.0, "miplib": 10.0, "infeasible": 10.0}

# Netlib models the LP literature singles out as degenerate or numerically hard.
HARD_NETLIB = ["cycle", "degen2", "degen3", "d2q06c", "dfl001", "greenbea", "greenbeb",
               "perold", "pilot", "pilot4", "pilot87", "pilotnov", "stair"]


# --------------------------------------------------------------------------- models

class RefModel:
    """Column-wise model used to check solutions independently of both solvers."""

    def __init__(self, names, cost, col_lo, col_hi, row_lo, row_hi, A, integer,
                 offset=0.0, sense=1.0, P=None):
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.cost = np.asarray(cost, float)
        self.col_lo, self.col_hi = np.asarray(col_lo, float), np.asarray(col_hi, float)
        self.row_lo, self.row_hi = np.asarray(row_lo, float), np.asarray(row_hi, float)
        self.A = A.tocsr()
        self.integer = np.asarray(integer, bool)
        self.offset, self.sense, self.P = float(offset), float(sense), P

    @property
    def shape(self) -> dict:
        return {"cols": len(self.names), "rows": int(self.A.shape[0]), "nnz": int(self.A.nnz),
                "integers": int(self.integer.sum()),
                "q_nnz": int(self.P.nnz) if self.P is not None else 0}

    def objective(self, x: np.ndarray) -> float:
        val = float(self.cost @ x) + self.offset
        if self.P is not None:
            val += 0.5 * float(x @ (self.P @ x))
        return val

    def check(self, x: np.ndarray) -> dict:
        def rel(viol, bound):
            b = np.where(np.abs(bound) >= INF, 0.0, np.abs(bound))
            return viol / (1.0 + b)

        bound_v = np.maximum(rel(np.maximum(self.col_lo - x, 0), self.col_lo),
                             rel(np.maximum(x - self.col_hi, 0), self.col_hi))
        ax = self.A @ x
        # Row residuals are scaled like OSQP's eps_rel * max(|Ax|, |b|): a row whose
        # terms sum to 1e6 in magnitude cannot be held to 1e-6 absolute in double.
        activity = abs(self.A) @ np.abs(x)
        row_v = np.maximum(
            np.maximum(self.row_lo - ax, 0) / (1.0 + np.maximum(np.where(np.abs(self.row_lo) >= INF, 0.0, np.abs(self.row_lo)), activity)),
            np.maximum(ax - self.row_hi, 0) / (1.0 + np.maximum(np.where(np.abs(self.row_hi) >= INF, 0.0, np.abs(self.row_hi)), activity)),
        ) if len(ax) else np.zeros(1)
        row_abs = np.maximum(rel(np.maximum(self.row_lo - ax, 0), self.row_lo),
                             rel(np.maximum(ax - self.row_hi, 0), self.row_hi)) if len(ax) else np.zeros(1)
        int_v = np.abs(x[self.integer] - np.round(x[self.integer])) if self.integer.any() else np.zeros(1)
        mb, mr, mi = float(bound_v.max(initial=0)), float(row_v.max(initial=0)), float(int_v.max(initial=0))
        ok = mb <= FEAS_TOL and mr <= FEAS_TOL and mi <= INT_TOL
        # Scaling rows by their activity lets an astronomically large x pass, so
        # disproving a published optimum also needs rows within 1e-6 of 1 + |rhs|.
        strict = ok and float(row_abs.max(initial=0)) <= FEAS_TOL and bool(np.all(np.abs(x) < 1e15))
        return {"ok": ok, "strict_ok": strict,
                "max_bound_violation": mb, "max_row_violation": mr,
                "max_row_violation_vs_rhs": float(row_abs.max(initial=0)),
                "max_integrality_violation": mi, "objective": self.objective(x)}


def _clean_inf(v):
    # The Maros-Meszaros MAT files encode infinity as 1e20, and in 11 of them
    # (PRIMALC*, POWELL20, Q* Netlib-derived) it has drifted to -9.9999999999997e19.
    v = np.asarray(v, float).ravel()
    v[v >= 1e19] = INF
    v[v <= -1e19] = -INF
    return v


def ref_from_mps(path: Path) -> RefModel:
    import highspy
    import scipy.sparse as sp
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("large_matrix_value", LARGE_MATRIX_VALUE)
    if h.readModel(str(path)) not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        raise RuntimeError(f"HiGHS could not read {path.name}")
    lp = h.getLp()
    a = lp.a_matrix_
    A = sp.csc_matrix((np.asarray(a.value_), np.asarray(a.index_), np.asarray(a.start_)),
                      shape=(lp.num_row_, lp.num_col_))
    integ = [int(t) != 0 for t in lp.integrality_] if len(lp.integrality_) else [False] * lp.num_col_
    sense = -1.0 if lp.sense_ == highspy.ObjSense.kMaximize else 1.0
    return RefModel(lp.col_names_, lp.col_cost_, _clean_inf(lp.col_lower_), _clean_inf(lp.col_upper_),
                    _clean_inf(lp.row_lower_), _clean_inf(lp.row_upper_), A, integ, lp.offset_, sense)


def qp_canonical(mat_path: Path) -> dict:
    """Maros-Meszaros MAT file -> bounds + remaining rows; singleton rows become bounds."""
    import scipy.io
    import scipy.sparse as sp
    d = scipy.io.loadmat(str(mat_path))
    P = sp.csc_matrix(d["P"]).astype(float)
    A = sp.csr_matrix(d["A"]).astype(float)
    n = P.shape[0]
    q = np.asarray(d["q"], float).ravel()
    r = float(np.asarray(d.get("r", 0.0)).ravel()[0]) if "r" in d else 0.0
    l, u = _clean_inf(d["l"]), _clean_inf(d["u"])
    if sp.tril(P, -1).nnz == 0 and sp.triu(P, 1).nnz > 0:
        P = (P + sp.triu(P, 1).T).tocsc()
    elif sp.triu(P, 1).nnz == 0 and sp.tril(P, -1).nnz > 0:
        P = (P + sp.tril(P, -1).T).tocsc()
    lo, hi = np.full(n, -INF), np.full(n, INF)
    counts = np.diff(A.indptr)
    keep = []
    for i in range(A.shape[0]):
        if counts[i] == 0:
            continue
        if counts[i] == 1:
            j, a = A.indices[A.indptr[i]], A.data[A.indptr[i]]
            if a != 0:
                bl, bu = (l[i] / a if l[i] > -INF else -INF), (u[i] / a if u[i] < INF else INF)
                if a < 0:
                    bl, bu = (u[i] / a if u[i] < INF else -INF), (l[i] / a if l[i] > -INF else INF)
                lo[j], hi[j] = max(lo[j], bl), min(hi[j], bu)
                continue
        keep.append(i)
    A = A[keep]
    return {"n": n, "P": P, "q": q, "r": r, "A": A, "l": l[keep], "u": u[keep], "lo": lo, "hi": hi}


def qp_json_path(name: str) -> Path:
    return CORPUS_DIR / "qp_json" / f"{name}.json"


def ensure_qp_json(entry: dict) -> tuple[Path, RefModel]:
    c = qp_canonical(ROOT / entry["file"])
    n, names = c["n"], [f"x{j}" for j in range(c["n"])]
    ref = RefModel(names, c["q"], c["lo"], c["hi"], c["l"], c["u"], c["A"], [False] * n,
                   offset=c["r"], P=c["P"])
    path = qp_json_path(entry["name"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        variables = [{"name": names[j], "type": "continuous",
                      "lower_bound": max(c["lo"][j], -INF), "upper_bound": min(c["hi"][j], INF)}
                     for j in range(n)]
        quad: dict[str, dict[str, float]] = {}
        Pc = c["P"].tocoo()
        for i, j, v in zip(Pc.row, Pc.col, Pc.data):
            if v != 0.0:
                quad.setdefault(names[i], {})[names[j]] = float(v)
        cons = []
        A = c["A"].tocsr()
        for i in range(A.shape[0]):
            lin = {names[A.indices[k]]: float(A.data[k]) for k in range(A.indptr[i], A.indptr[i + 1])}
            lo_i, hi_i = c["l"][i], c["u"][i]
            if lo_i > -INF and hi_i < INF and lo_i == hi_i:
                cons.append({"name": f"r{i}", "linear": lin, "sense": "=", "rhs": float(lo_i)})
                continue
            if hi_i < INF:
                cons.append({"name": f"r{i}u", "linear": lin, "sense": "<=", "rhs": float(hi_i)})
            if lo_i > -INF:
                cons.append({"name": f"r{i}l", "linear": lin, "sense": ">=", "rhs": float(lo_i)})
        model = {"problem_type": "QP", "sense": "minimize", "variables": variables,
                 "objective": {"constant": c["r"],
                               "linear": {names[j]: float(c["q"][j]) for j in range(n) if c["q"][j] != 0},
                               "quadratic": quad},
                 "constraints": cons}
        path.write_text(json.dumps(model), encoding="utf-8")
    return path, ref


# --------------------------------------------------------------------------- solvers

def find_sovereign() -> str:
    for p in [ROOT / "build64" / "solver" / "sovereign.exe", ROOT / "build" / "solver" / "sovereign.exe",
              ROOT / "build" / "solver" / "sovereign"]:
        if p.exists():
            return str(p)
    raise FileNotFoundError("sovereign binary not found; build it first")


def _pin(proc: subprocess.Popen, cpus: list[int] | None) -> None:
    if not cpus:
        return
    try:
        import psutil
        psutil.Process(proc.pid).cpu_affinity(cpus)
    except Exception:
        pass


def _run(cmd: list[str], env: dict, timeout: float, cpus: list[int] | None) -> tuple[str, str, float, bool]:
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    _pin(proc, cpus)
    try:
        out, err = proc.communicate(timeout=timeout)
        return out, err, time.perf_counter() - t0, False
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return out or "", err or "", time.perf_counter() - t0, True


def run_sovereign(binary: str, model: Path, kind: str, limit: float, cpus, extra_env: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOVEREIGN_")}
    env.update(extra_env or {})
    env["SOVEREIGN_PARALLEL_WORKERS"] = "1"
    env["SOVEREIGN_DISABLE_CUDA"] = "1"
    if kind == "MILP":
        env["SOVEREIGN_TIME_LIMIT"] = str(limit)
        # HiGHS has no node limit; the default 100000 would stop Sovereign
        # long before the shared time limit (enlight9: 71 s of 300 s).
        env["SOVEREIGN_MAX_NODES"] = str(2**31 - 1)
    out, err, wall, killed = _run([binary, "solve", str(model)], env, limit + 60, cpus)
    if killed:
        return {"status": "TIME_LIMIT", "time": limit, "wall": wall, "killed": True, "x": None}
    try:
        p = json.loads(out)
    except json.JSONDecodeError:
        return {"status": "ERROR", "time": limit, "wall": wall, "x": None,
                "message": (err or out).strip()[-300:]}
    status = p.get("status", "ERROR")
    t = float(p.get("runtime_seconds") or wall)
    if t > limit and status == "OPTIMAL" and kind != "MILP":
        status = "TIME_LIMIT"
    return {"status": status, "time": t, "wall": wall, "load": p.get("load_seconds"),
            "objective": p.get("objective_value"), "x": p.get("primal"),
            "iterations": p.get("iterations"), "nodes": p.get("nodes"),
            "gap": p.get("optimality_gap"), "message": (p.get("message") or "")[:200]}


def highs_worker(model: str, limit: float, qp: bool) -> dict:
    """Runs inside a child process so a hang or crash cannot take down the harness."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(limit))
    h.setOptionValue("mip_rel_gap", MIP_GAP)
    h.setOptionValue("large_matrix_value", LARGE_MATRIX_VALUE)
    if qp:
        c = qp_canonical(ROOT / model)
        import scipy.sparse as sp
        n = c["n"]
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = n, c["A"].shape[0]
        lp.col_cost_, lp.offset_ = c["q"], c["r"]
        lp.col_lower_ = np.where(c["lo"] <= -INF, -highspy.kHighsInf, c["lo"])
        lp.col_upper_ = np.where(c["hi"] >= INF, highspy.kHighsInf, c["hi"])
        lp.row_lower_ = np.where(c["l"] <= -INF, -highspy.kHighsInf, c["l"])
        lp.row_upper_ = np.where(c["u"] >= INF, highspy.kHighsInf, c["u"])
        Ac = c["A"].tocsc()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = Ac.indptr, Ac.indices, Ac.data
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, c["A"].shape[0]
        L = sp.tril(c["P"]).tocsc()
        hess = highspy.HighsHessian()
        hess.dim_, hess.format_ = n, highspy.HessianFormat.kTriangular
        hess.start_, hess.index_, hess.value_ = L.indptr, L.indices, L.data
        m = highspy.HighsModel()
        m.lp_, m.hessian_ = lp, hess
        # The active-set QP solver refuses models whose null space exceeds this
        # (default 4000); lift it so large Maros-Meszaros models get attempted.
        h.setOptionValue("qp_nullspace_limit", max(4000, n))
        h.passModel(m)
    else:
        h.readModel(model)
    t0 = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus()).upper().replace(" ", "_")
    status = {"TIME_LIMIT_REACHED": "TIME_LIMIT", "PRIMAL_INFEASIBLE_OR_UNBOUNDED": "INFEASIBLE_OR_UNBOUNDED",
              "ITERATION_LIMIT_REACHED": "ITERATION_LIMIT"}.get(status, status)
    sol = h.getSolution()
    info = h.getInfo()
    x = list(sol.col_value) if sol.value_valid else None
    return {"status": status, "time": elapsed, "objective": info.objective_function_value if x else None,
            "x": x, "gap": info.mip_gap if math.isfinite(info.mip_gap) else None,
            "iterations": info.simplex_iteration_count + info.ipm_iteration_count,
            "nodes": info.mip_node_count if info.mip_node_count >= 0 else None, "version": h.version()}


def run_highs(model: str, kind: str, limit: float, cpus) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--highs-worker", model,
           "--time-limit", str(limit)] + (["--qp"] if kind == "QP" else [])
    out, err, wall, killed = _run(cmd, os.environ.copy(), limit + 60, cpus)
    if killed:
        return {"status": "TIME_LIMIT", "time": limit, "wall": wall, "killed": True, "x": None}
    try:
        r = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"status": "ERROR", "time": limit, "wall": wall, "x": None, "message": (err or out)[-300:]}
    r["wall"] = wall
    return r


# --------------------------------------------------------------------------- scoring

def to_vector(x: Any, ref: RefModel) -> tuple[np.ndarray | None, int]:
    if x is None:
        return None, 0
    if isinstance(x, dict):
        v = np.zeros(len(ref.names))
        missing = 0
        for name, i in ref.index.items():
            if name in x:
                v[i] = x[name]
            else:
                missing += 1
        return v, missing
    return np.asarray(x, float), 0


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


def reference_tolerance(entry: dict, kind: str) -> float:
    ref = entry.get("reference") or {}
    if kind in ("LP", "QP") and ref.get("significant_digits"):
        base = LP_OBJ_RTOL if kind == "LP" else QP_OBJ_RTOL
        return max(base, 0.5 * 10.0 ** (1 - ref["significant_digits"]))
    return {"LP": LP_OBJ_RTOL, "QP": QP_OBJ_RTOL}.get(kind, MILP_OBJ_RTOL)


def score(entry: dict, kind: str, res: dict, ref: RefModel, limit: float) -> dict:
    """Attach verification and an outcome label to one solver result."""
    x, missing = to_vector(res.pop("x", None), ref)
    res["missing_values"] = missing
    if x is not None:
        res["check"] = ref.check(x)
    status = res["status"]
    reference = entry.get("reference") or {}
    if entry["suite"] == "infeasible":
        res["outcome"] = ("SOLVED" if status == "INFEASIBLE" else
                          "WRONG" if status in ("OPTIMAL", "FEASIBLE") else
                          "TIMEOUT" if status == "TIME_LIMIT" else "FAILED")
        return res
    chk = res.get("check")
    if status == "OPTIMAL":
        if not chk or not chk["ok"] or missing:
            res["outcome"] = "WRONG"
            res["why"] = "claimed optimal but failed the independent feasibility check"
            return res
        ref_val = reference.get("value") if reference.get("kind", "opt") == "opt" else None
        if ref_val is not None:
            res["ref_rel_error"] = rel_diff(chk["objective"], ref_val)
            if entry["suite"] in ("netlib", "qp") and ref.offset:
                # Some readme optima (e.g. Netlib E226) omit the objective constant.
                res["ref_rel_error"] = min(res["ref_rel_error"], rel_diff(chk["objective"] - ref.offset, ref_val))
            ok = res["ref_rel_error"] <= reference_tolerance(entry, kind)
            res["outcome"] = "SOLVED" if ok else "WRONG"
            if not ok:
                res["why"] = "objective differs from the published optimum"
        else:
            res["outcome"] = "SOLVED"
        return res
    if status in ("INFEASIBLE", "UNBOUNDED", "INFEASIBLE_OR_UNBOUNDED"):
        res["outcome"] = "WRONG"
        res["why"] = f"claimed {status} on a model with a known optimum"
        return res
    if status in ("TIME_LIMIT", "FEASIBLE", "ITERATION_LIMIT"):
        # A FEASIBLE point returned well inside the budget is a solver giving
        # up (YAO: Frank-Wolfe after 0.8 s), not a time limit.
        gave_up = status == "FEASIBLE" and res.get("time", limit) < 0.95 * limit
        res["outcome"] = "FAILED" if gave_up else "TIMEOUT"
        if gave_up:
            res["why"] = "returned a feasible point without proving optimality before the time limit"
        if chk and chk["ok"] and reference.get("value") is not None:
            res["primal_gap"] = rel_diff(chk["objective"], reference["value"])
        return res
    res["outcome"] = "FAILED"
    return res


def check_reference(row: dict, kind: str) -> None:
    """A verified point that beats the published optimum disproves it.

    The Netlib readme has known stale optima (greenbea, greenbeb, pilot; see
    Koch 2004). When such a point beats the reference by more than the
    tolerance (see the acceptance rule below), the reference cannot be optimal,
    so optimality claims are scored against the best verified objective instead.
    """
    ref_val = row["reference"]["value"]
    tol = reference_tolerance(row, kind)
    sense = row.get("sense", 1.0)
    offset = row.get("offset", 0.0)
    verified = {k: row[k]["check"]["objective"] for k in ("sovereign", "highs")
                if (row[k].get("check") or {}).get("ok") and not row[k].get("missing_values")}

    def beats(obj: float, target: float) -> bool:
        return sense * (target - obj) > tol * max(1.0, abs(obj), abs(target))

    disproof = {k: v for k, v in verified.items()
                if row[k]["check"].get("strict_ok")
                and beats(v, ref_val) and not (offset and not beats(v - offset, ref_val))}
    # A point within 1e-6 on every row is not enough on its own: on
    # ill-conditioned models (LISWET) it can still sit far from the true
    # optimum. Either both solvers beat the published value and agree, or one
    # point is feasible to EXACT_FEAS_TOL (LISWET8's 00README.QP entry is a
    # typo, 7144.7006 for 714.47006, beaten by a point feasible to 2e-15).
    exact = {k: v for k, v in disproof.items()
             if row[k]["check"]["max_row_violation_vs_rhs"] <= EXACT_FEAS_TOL
             and row[k]["check"]["max_bound_violation"] <= EXACT_FEAS_TOL}
    if len(disproof) == 2 and rel_diff(disproof["sovereign"], disproof["highs"]) <= tol:
        accepted = disproof
    elif exact:
        accepted = exact
    else:
        return
    best = min(accepted.values(), key=lambda v: sense * v)
    row["reference_disproved"] = {"published": ref_val, "best_verified": best, "by": sorted(accepted)}
    for k in ("sovereign", "highs"):
        r = row[k]
        if r.get("status") != "OPTIMAL" or k not in verified:
            continue
        r["ref_rel_error"] = rel_diff(verified[k], best)
        ok = r["ref_rel_error"] <= tol
        r["outcome"] = "SOLVED" if ok else "WRONG"
        r["why"] = ("matches the best verified objective (published optimum disproved)" if ok else
                    "other solver found a verified point with a better objective")


def cross_check(row: dict, kind: str) -> None:
    """Without a published optimum, a verified better point disproves an optimality claim."""
    s, h = row["sovereign"], row["highs"]
    if row["suite"] != "infeasible" and s.get("check") and h.get("check"):
        row["solver_rel_diff"] = rel_diff(s["check"]["objective"], h["check"]["objective"])
    if row["suite"] == "infeasible":
        return
    if (row.get("reference") or {}).get("value") is not None:
        check_reference(row, kind)
        return
    if s.get("outcome") == "SOLVED" and h.get("outcome") == "SOLVED":
        a, b = s["check"]["objective"], h["check"]["objective"]
        tol = QP_OBJ_RTOL if kind == "QP" else LP_OBJ_RTOL
        if rel_diff(a, b) > tol:
            worse = s if a > b else h
            worse["outcome"] = "WRONG"
            worse["why"] = "other solver found a verified point with a better objective"
        row["agree"] = rel_diff(a, b) <= tol


# --------------------------------------------------------------------------- driver

def load_corpus() -> dict:
    path = CORPUS_DIR / "corpus.json"
    if not path.exists():
        sys.exit("corpus missing: run python benchmarks/tools/fetch_coverage_corpus.py")
    return json.loads(path.read_text())


def done_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["name"] for line in path.read_text().splitlines() if line.strip()}


def run_instance(binary: str, entry: dict, limit: float, repeats: int, repeat_under: float,
                 pin_ours, pin_highs) -> dict:
    kind = entry["kind"]
    if kind == "QP":
        model_path, ref = ensure_qp_json(entry)
        highs_model = entry["file"]
    else:
        model_path, ref = ROOT / entry["file"], ref_from_mps(ROOT / entry["file"])
        highs_model = str(model_path)

    def lane_ours():
        return run_sovereign(binary, model_path, kind, limit, pin_ours)

    def lane_highs():
        return run_highs(highs_model, kind, limit, pin_highs)

    with cf.ThreadPoolExecutor(2) as pool:
        fo, fh = pool.submit(lane_ours), pool.submit(lane_highs)
        ours, highs = fo.result(), fh.result()
    row = {"name": entry["name"], "suite": entry["suite"], "kind": kind, **ref.shape,
           "reference": entry.get("reference"), "application": entry.get("application"),
           "sense": ref.sense, "offset": ref.offset,
           "sovereign": score(entry, kind, ours, ref, limit), "highs": score(entry, kind, highs, ref, limit)}
    cross_check(row, kind)

    # Re-time quick, solved runs and keep the median, to damp timer noise.
    for key, lane in (("sovereign", lane_ours), ("highs", lane_highs)):
        r = row[key]
        r["times"] = [r["time"]]
        if repeats > 1 and r["outcome"] == "SOLVED" and r["time"] < repeat_under:
            for _ in range(repeats - 1):
                again = lane()
                if again["status"] == r["status"]:
                    r["times"].append(again["time"])
            r["time"] = statistics.median(r["times"])
    return row


def machine_info() -> dict:
    import highspy
    try:
        import psutil
        cpu = f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads"
    except Exception:
        cpu = f"{os.cpu_count()} threads"
    proc = platform.processor()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        proc = out or proc
    except Exception:
        pass
    return {"platform": platform.platform(), "cpu": proc, "cpu_counts": cpu,
            "python": platform.python_version(), "highs": highspy.Highs().version()}


def run_suite(args, suite: str, corpus: dict) -> None:
    binary = find_sovereign()
    entries = corpus.get(suite, [])
    if args.only:
        entries = [e for e in entries if e["name"] in set(args.only)]
    out = OUT_DIR / f"{suite}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.force and out.exists() and not args.only:
        out.unlink()
    skip = set() if args.force else done_names(out)
    if args.force and args.only and out.exists():
        keep = [l for l in out.read_text().splitlines() if l.strip() and json.loads(l)["name"] not in set(args.only)]
        out.write_text("".join(k + "\n" for k in keep))
    meta = {"suite": suite, "time_limit": args.time_limit, "repeats": args.repeats,
            "repeat_under_s": args.repeat_under, "pin_sovereign": args.pin_ours, "pin_highs": args.pin_highs,
            "started": dt.datetime.now().isoformat(timespec="seconds"), "binary": binary, **machine_info()}
    (OUT_DIR / f"{suite}.meta.json").write_text(json.dumps(meta, indent=1))
    todo = [e for e in entries if e["name"] not in skip]
    print(f"[{suite}] {len(todo)} to run ({len(entries) - len(todo)} already done), "
          f"limit {args.time_limit}s", flush=True)
    for k, entry in enumerate(todo, 1):
        try:
            row = run_instance(binary, entry, args.time_limit, args.repeats, args.repeat_under,
                               args.pin_ours, args.pin_highs)
        except Exception as exc:  # a broken input must not stop the sweep
            row = {"name": entry["name"], "suite": suite, "kind": entry["kind"], "harness_error": str(exc)[:300],
                   "sovereign": {"outcome": "FAILED", "status": "ERROR", "time": args.time_limit},
                   "highs": {"outcome": "FAILED", "status": "ERROR", "time": args.time_limit}}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        s, h = row["sovereign"], row["highs"]
        print(f"[{suite} {k}/{len(todo)}] {entry['name']:<22} "
              f"sovereign {s['outcome']:<7} {s.get('time', 0):8.3f}s | "
              f"highs {h['outcome']:<7} {h.get('time', 0):8.3f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", nargs="*", choices=SUITES, default=[])
    ap.add_argument("--only", nargs="*", help="run just these instance names")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--repeats", type=int, default=3, help="timings per solved run under --repeat-under")
    ap.add_argument("--repeat-under", type=float, default=20.0)
    ap.add_argument("--pin-ours", type=lambda s: [int(c) for c in s.split(",")], default=[2, 3])
    ap.add_argument("--pin-highs", type=lambda s: [int(c) for c in s.split(",")], default=[4, 5])
    ap.add_argument("--force", action="store_true", help="re-run instances that already have results")
    ap.add_argument("--report", action="store_true", help="write COVERAGE.md from existing results")
    ap.add_argument("--highs-worker", help=argparse.SUPPRESS)
    ap.add_argument("--qp", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.highs_worker:
        print(json.dumps(highs_worker(args.highs_worker, args.time_limit, args.qp)))
        return 0
    corpus = load_corpus()
    for suite in args.suite:
        run_suite(args, suite, corpus)
    if args.report:
        from coverage_report import write_report
        write_report(OUT_DIR, ROOT / "benchmarks" / "reports" / "COVERAGE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
