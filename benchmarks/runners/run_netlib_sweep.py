"""Reproducible Netlib LP sweep with an original-model verifier.

This runner is intentionally separate from the broad coverage harness. It
records every locally available feasible Netlib model and every locally
available infeasible-corpus model, including failures and timeouts.

The Sovereign binary is the only production executable invoked here. HiGHS is
started only through the benchmark worker in ``run_coverage.py``. The
independent verifier recomputes primal feasibility, dual feasibility and
complementarity from the original MPS-derived model and the exported LP
certificate; it never trusts the solver's status or residual scalars.

Typical usage:

    python benchmarks/runners/run_netlib_sweep.py --time-limit 300

If the ignored coverage corpus is absent, fetch it with:

    python benchmarks/tools/fetch_coverage_corpus.py --suite netlib infeasible
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "benchmarks" / "datasets" / "coverage"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "netlib-raw"
CSV_PATH = RESULTS_DIR / "netlib.csv"
MD_PATH = RESULTS_DIR / "netlib.md"
PLOT_PATH = RESULTS_DIR / "netlib-performance-profile.png"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage import (  # noqa: E402
    INF,
    RefModel,
    ref_from_mps,
    run_highs,
)


VERIFY_TOL = 1e-6
OBJECTIVE_TOL = 1e-6
SGM_SHIFT = 10.0
DEFAULT_LIMIT = 300.0
NETLIB_REFERENCE_SOURCE = "https://www.netlib.org/lp/data/readme"
SUITES = ("netlib",)
SUITE_LABELS = {
    "netlib": "feasible Netlib LP",
}


CSV_FIELDS = [
    "suite",
    "instance",
    "source_path",
    "sha256",
    "bytes",
    "rows",
    "cols",
    "nnz",
    "reference_objective",
    "reference_offset",
    "reference_kind",
    "reference_source",
    "reference_significant_digits",
    "reference_approximate",
    "reference_note",
    "sovereign_status",
    "sovereign_verified_status",
    "sovereign_objective",
    "sovereign_lp_diagnostics",
    "certificate_lp_diagnostics",
    "sovereign_time_s",
    "sovereign_wall_s",
    "sovereign_iterations",
    "sovereign_presolve_fixed_variables",
    "sovereign_presolve_substituted_variables",
    "sovereign_presolve_removed_constraints",
    "sovereign_presolve_tightened_bounds",
    "sovereign_presolve_passes",
    "certificate_status",
    "certificate_time_s",
    "certificate_wall_s",
    "certificate_iterations",
    "verifier_pass",
    "verifier_primal_infeasibility",
    "verifier_dual_infeasibility",
    "verifier_complementarity",
    "verifier_objective",
    "verifier_objective_reference_rel_diff",
    "verifier_objective_highs_rel_diff",
    "highs_status",
    "highs_objective",
    "highs_time_s",
    "highs_iterations",
    "highs_objective_reference_rel_diff",
    "failure_cause",
    "failure_detail",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return None
    return abs(a - b) / max(1.0, abs(a), abs(b))


def published_rel_diff(value: float | None, reference: float | None, offset: float = 0.0) -> float | None:
    candidates = [rel_diff(value, reference)]
    if value is not None and reference is not None and offset:
        candidates.append(rel_diff(value - offset, reference))
    usable = [candidate for candidate in candidates if candidate is not None]
    return min(usable) if usable else None


def finite_bound(value: float) -> bool:
    return math.isfinite(value) and abs(value) < INF / 2


def load_jsonl_names(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(row.get("name", "")).casefold()
        if name:
            out[name] = row
    return out


def load_corpus() -> dict[str, list[dict]]:
    path = CORPUS_DIR / "corpus.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def corpus_reference_maps(corpus: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    maps: dict[str, dict[str, dict]] = {}
    for suite in SUITES:
        maps[suite] = {}
        for entry in corpus.get(suite, []):
            name = str(entry.get("name", "")).casefold()
            if name:
                maps[suite][name] = entry
    # The older checked-in report is useful when only AFIRO is present and the
    # ignored fetched corpus has not been materialized locally.
    report_map = load_jsonl_names(ROOT / "benchmarks" / "reports" / "coverage" / "netlib.jsonl")
    for name, row in report_map.items():
        if name not in maps["netlib"]:
            maps["netlib"][name] = {
                "name": row.get("name", name),
                "suite": "netlib",
                "kind": "LP",
                "reference": row.get("reference"),
            }
    return maps


def discover_files() -> tuple[dict[str, dict[str, Path]], dict[str, dict[str, dict]], dict]:
    corpus = load_corpus()
    references = corpus_reference_maps(corpus)
    found: dict[str, dict[str, Path]] = {suite: {} for suite in SUITES}
    # Prefer the fetched coverage copy because its path/hash is pinned by
    # corpus.json. Fall back to the tracked AFIRO copy.
    roots = {
        "netlib": [CORPUS_DIR / "netlib", ROOT / "benchmarks" / "datasets" / "netlib"],
    }
    for suite, directories in roots.items():
        for directory in directories:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.mps")):
                found[suite].setdefault(path.stem.casefold(), path)
    return found, references, corpus


def git_value(args: list[str], fallback: str = "unavailable") -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        value = proc.stdout.strip()
        return value or fallback
    except (OSError, subprocess.SubprocessError):
        return fallback


def build_metadata(binary: str | None = None) -> dict[str, str]:
    candidates: list[Path] = []
    if binary:
        binary_path = Path(binary)
        if not binary_path.is_absolute():
            binary_path = ROOT / binary_path
        candidates.extend(parent / "CMakeCache.txt" for parent in binary_path.resolve().parents)
    candidates.extend(
        [
            ROOT / "build64" / "CMakeCache.txt",
            ROOT / "build" / "CMakeCache.txt",
        ]
    )
    cache = next((path for path in candidates if path.exists()), candidates[-1])
    values: dict[str, str] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            name = key.split(":", 1)[0]
            if name in {
                "CMAKE_BUILD_TYPE",
                "CMAKE_CXX_COMPILER",
                "CMAKE_CXX_FLAGS",
                "CMAKE_CXX_FLAGS_RELEASE",
            }:
                values[name] = value
    values["build_cache"] = str(cache.relative_to(ROOT)) if cache.exists() else "unavailable"
    return values


def environment_metadata(highs_version: str | None, binary: str | None) -> dict[str, str]:
    cpu = platform.processor() or "unknown"
    return {
        "commit": git_value(["rev-parse", "HEAD"]),
        "working_tree": "clean" if not git_value(["status", "--porcelain"], "").strip() else "dirty",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu,
        "python": platform.python_version(),
        "threads_available": str(os.cpu_count() or 1),
        "fixed_threads": "1",
        "highspy": highs_version or "unavailable",
        "binary": binary or "unavailable",
        **build_metadata(binary),
    }


def parse_highs_version() -> str | None:
    try:
        import highspy

        return str(highspy.Highs().version())
    except Exception:
        return None


def solver_environment(presolve: bool) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("SOVEREIGN_"):
            del env[key]
    env.update(
        {
            "SOVEREIGN_DISABLE_CUDA": "1",
            "SOVEREIGN_PARALLEL_WORKERS": "1",
            "SOVEREIGN_PRESOLVE": "1" if presolve else "0",
        }
    )
    return env


def original_ref_from_mps(path: Path):
    """Build the verifier model with the C++ MPS reader's range expansion.

    HiGHS keeps an MPS RANGES record as one row with both bounds. Sovereign's
    from-scratch MPS reader expands that same record into an upper inequality
    followed by a lower inequality. Splitting bounded non-equality rows here
    keeps the independently parsed row order aligned with exported
    ``row_<index>`` dual diagnostics without using solver code.
    """
    parsed = ref_from_mps(path)
    row_indices: list[int] = []
    row_lo: list[float] = []
    row_hi: list[float] = []
    for i, (lo, hi) in enumerate(zip(parsed.row_lo, parsed.row_hi)):
        if finite_bound(lo) and finite_bound(hi) and lo != hi:
            row_indices.extend([i, i])
            row_lo.extend([-INF, lo])
            row_hi.extend([hi, INF])
        else:
            row_indices.append(i)
            row_lo.append(lo)
            row_hi.append(hi)
    A = parsed.A.tocsr()[row_indices, :]
    return RefModel(
        parsed.names,
        parsed.cost,
        parsed.col_lo,
        parsed.col_hi,
        np.asarray(row_lo, dtype=float),
        np.asarray(row_hi, dtype=float),
        A,
        parsed.integer,
        offset=parsed.offset,
        sense=parsed.sense,
        P=parsed.P,
    )


def parse_solver_payload(stdout: str, output_path: Path) -> dict | None:
    if output_path.exists():
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def run_sovereign(binary: str, model: Path, output_path: Path, limit: float, presolve: bool) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = output_path.with_suffix(".stderr.log")
    command = [binary, "solve", str(model), "--out", str(output_path)]
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=solver_environment(presolve),
    )
    try:
        stdout, stderr = proc.communicate(timeout=limit)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    wall = time.perf_counter() - started
    stderr_path.write_text(stderr or "", encoding="utf-8")
    payload = parse_solver_payload(stdout or "", output_path)
    if timed_out:
        return {
            "status": "TIME_LIMIT",
            "time": limit,
            "wall": wall,
            "payload": payload,
            "returncode": proc.returncode,
            "message": f"external timeout after {limit:g}s",
        }
    if payload is None:
        return {
            "status": "ERROR",
            "time": wall,
            "wall": wall,
            "payload": None,
            "returncode": proc.returncode,
            "message": (stderr or stdout or "solver produced no JSON")[-500:],
        }
    return {
        "status": str(payload.get("status", "ERROR")).upper(),
        "time": float(payload.get("runtime_seconds") or wall),
        "wall": wall,
        "payload": payload,
        "returncode": proc.returncode,
        "message": str(payload.get("message") or "")[:500],
    }


def vector_from_payload(payload: dict | None, ref) -> np.ndarray | None:
    if not payload or not isinstance(payload.get("primal"), dict):
        return None
    primal = payload["primal"]
    values = []
    for name in ref.names:
        try:
            value = float(primal[name])
        except (KeyError, TypeError, ValueError):
            return None
        values.append(value)
    return np.asarray(values, dtype=float)


def independent_verify(ref, payload: dict | None) -> dict:
    """Verify an LP KKT certificate against the original MPS-derived model."""
    result = {
        "status": "UNKNOWN",
        "pass": False,
        "primal_infeasibility": None,
        "dual_infeasibility": None,
        "complementarity": None,
        "objective": None,
        "detail": "",
    }
    if not payload:
        result["detail"] = "missing certificate payload"
        return result
    x = vector_from_payload(payload, ref)
    if x is None or not np.all(np.isfinite(x)):
        result["detail"] = "missing or non-finite primal vector"
        return result
    objective = ref.objective(x)
    result["objective"] = objective
    if not isinstance(payload.get("dual"), dict):
        result["detail"] = "solver did not export an LP dual certificate"
        return result
    dual_map = payload["dual"]
    row_dual = np.empty(ref.A.shape[0], dtype=float)
    missing = []
    for i in range(ref.A.shape[0]):
        key = f"row_{i}"
        if key not in dual_map:
            missing.append(key)
        else:
            try:
                row_dual[i] = float(dual_map[key])
            except (TypeError, ValueError):
                missing.append(key)
    if missing:
        result["detail"] = f"missing row dual certificate entries: {len(missing)}"
        return result
    if not np.all(np.isfinite(row_dual)):
        result["detail"] = "non-finite row dual certificate"
        return result

    ax = np.asarray(ref.A @ x, dtype=float)
    activity_scale = np.asarray(abs(ref.A) @ np.abs(x), dtype=float)
    bound_violation = np.zeros_like(x)
    lower_mask = np.array([finite_bound(v) for v in ref.col_lo])
    upper_mask = np.array([finite_bound(v) for v in ref.col_hi])
    bound_violation[lower_mask] = np.maximum(
        (ref.col_lo[lower_mask] - x[lower_mask])
        / (1.0 + np.abs(ref.col_lo[lower_mask])),
        0.0,
    )
    bound_violation[upper_mask] = np.maximum(
        bound_violation[upper_mask],
        (x[upper_mask] - ref.col_hi[upper_mask])
        / (1.0 + np.abs(ref.col_hi[upper_mask])),
    )

    row_violation = np.zeros_like(ax)
    row_lo_mask = np.array([finite_bound(v) for v in ref.row_lo])
    row_hi_mask = np.array([finite_bound(v) for v in ref.row_hi])
    if np.any(row_lo_mask):
        denominator = 1.0 + np.maximum(np.abs(ref.row_lo[row_lo_mask]), activity_scale[row_lo_mask])
        row_violation[row_lo_mask] = np.maximum(
            (ref.row_lo[row_lo_mask] - ax[row_lo_mask]) / denominator,
            0.0,
        )
    if np.any(row_hi_mask):
        denominator = 1.0 + np.maximum(np.abs(ref.row_hi[row_hi_mask]), activity_scale[row_hi_mask])
        row_violation[row_hi_mask] = np.maximum(
            row_violation[row_hi_mask],
            (ax[row_hi_mask] - ref.row_hi[row_hi_mask]) / denominator,
        )
    primal = float(max(bound_violation.max(initial=0.0), row_violation.max(initial=0.0)))

    # Sovereign's LP dual convention is c - A^T y. It uses y <= 0 for
    # upper-bounded (<=) rows and y >= 0 for lower-bounded (>=) rows.
    sense = float(getattr(ref, "sense", 1.0))
    effective_cost = sense * ref.cost
    gradient = effective_cost - np.asarray(ref.A.T @ row_dual, dtype=float)
    dual_sign_violation = np.zeros_like(row_dual)
    le_mask = (~row_lo_mask) & row_hi_mask
    ge_mask = row_lo_mask & (~row_hi_mask)
    if np.any(le_mask):
        dual_sign_violation[le_mask] = np.maximum(row_dual[le_mask], 0.0)
    if np.any(ge_mask):
        dual_sign_violation[ge_mask] = np.maximum(-row_dual[ge_mask], 0.0)

    stationarity = np.zeros_like(gradient)
    fixed_mask = lower_mask & upper_mask & (
        np.abs(ref.col_hi - ref.col_lo) <= VERIFY_TOL * (1.0 + np.abs(ref.col_lo))
    )
    for j in range(len(x)):
        if fixed_mask[j]:
            stationarity[j] = 0.0
        elif lower_mask[j] and x[j] <= ref.col_lo[j] + VERIFY_TOL * (1.0 + abs(ref.col_lo[j])):
            stationarity[j] = max(0.0, -gradient[j])
        elif upper_mask[j] and x[j] >= ref.col_hi[j] - VERIFY_TOL * (1.0 + abs(ref.col_hi[j])):
            stationarity[j] = max(0.0, gradient[j])
        else:
            stationarity[j] = abs(gradient[j])

    # Revised simplex may export generated upper-bound row duals. Check them
    # when available, but do not require them: the original-space gradient
    # above is sufficient to certify bound dual feasibility.
    upper_consistency = np.zeros(len(x), dtype=float)
    for key, value in dual_map.items():
        match = re.fullmatch(r"upper_(\d+)", str(key))
        if not match:
            continue
        j = int(match.group(1))
        if j >= len(x):
            continue
        try:
            upper_consistency[j] = abs(gradient[j] - float(value))
        except (TypeError, ValueError):
            upper_consistency[j] = math.inf

    dual_scale = 1.0
    if len(effective_cost):
        dual_scale = max(dual_scale, float(np.max(np.abs(effective_cost))))
    if len(gradient):
        dual_scale = max(dual_scale, float(np.max(np.abs(gradient))))
    dual_scale = max(dual_scale, float(np.max(np.abs(row_dual), initial=0.0)))
    dual = float(
        max(
            dual_sign_violation.max(initial=0.0),
            stationarity.max(initial=0.0),
            upper_consistency.max(initial=0.0),
        )
        / dual_scale
    )

    row_comp = np.zeros_like(row_dual)
    if np.any(le_mask):
        row_comp[le_mask] = np.abs(row_dual[le_mask]) * np.maximum(
            ref.row_hi[le_mask] - ax[le_mask], 0.0
        )
    if np.any(ge_mask):
        row_comp[ge_mask] = np.abs(row_dual[ge_mask]) * np.maximum(
            ax[ge_mask] - ref.row_lo[ge_mask], 0.0
        )
    lower_multiplier = np.maximum(gradient, 0.0)
    upper_multiplier = np.maximum(-gradient, 0.0)
    bound_comp = np.zeros_like(x)
    if np.any(lower_mask):
        bound_comp[lower_mask] = lower_multiplier[lower_mask] * np.maximum(
            x[lower_mask] - ref.col_lo[lower_mask], 0.0
        )
    if np.any(upper_mask):
        bound_comp[upper_mask] = np.maximum(
            bound_comp[upper_mask],
            upper_multiplier[upper_mask]
            * np.maximum(ref.col_hi[upper_mask] - x[upper_mask], 0.0),
        )
    comp_scale = 1.0
    comp_scale = max(comp_scale, abs(float(effective_cost @ x)))
    if len(ref.row_hi):
        finite_rows = row_lo_mask | row_hi_mask
        comp_scale = max(
            comp_scale,
            float(np.max(np.abs(row_dual[finite_rows] * np.where(
                row_hi_mask[finite_rows], ref.row_hi[finite_rows], ref.row_lo[finite_rows]
            )), initial=0.0)),
        )
    if len(x):
        comp_scale = max(comp_scale, float(np.max(np.abs(x * gradient))))
    complementarity = float(
        max(row_comp.max(initial=0.0), bound_comp.max(initial=0.0)) / comp_scale
    )

    result.update(
        {
            "status": "OPTIMAL" if primal <= VERIFY_TOL and dual <= VERIFY_TOL and complementarity <= VERIFY_TOL else "UNKNOWN",
            "pass": primal <= VERIFY_TOL and dual <= VERIFY_TOL and complementarity <= VERIFY_TOL,
            "primal_infeasibility": primal,
            "dual_infeasibility": dual,
            "complementarity": complementarity,
            "objective": objective,
            "detail": "original-model KKT verification passed"
            if primal <= VERIFY_TOL and dual <= VERIFY_TOL and complementarity <= VERIFY_TOL
            else "original-model KKT residual exceeded fixed tolerance",
        }
    )
    return result


def classify_failure(primary: dict, certificate: dict, verification: dict, suite: str) -> tuple[str, str]:
    status = primary.get("status", "ERROR")
    cert_status = certificate.get("status", "ERROR")
    text = " ".join(
        str(value)
        for value in (
            primary.get("message", ""),
            (primary.get("payload") or {}).get("message", ""),
            (primary.get("payload") or {}).get("warnings", ""),
        )
    ).lower()
    if suite == "infeasible":
        if status == "INFEASIBLE":
            return "missing certificate", "infeasible status has no independently checkable Farkas certificate"
        if status in {"OPTIMAL", "FEASIBLE"}:
            return "wrong infeasible/unbounded detection", f"infeasible corpus returned {status}"
    if (
        status in {"TIME_LIMIT", "ITERATION_LIMIT"}
        or cert_status in {"TIME_LIMIT", "ITERATION_LIMIT"}
        or "iteration limit" in text
        or "time limit" in text
    ):
        return "iteration or time limit", primary.get("message", "time or iteration limit")
    if "presolve" in text:
        return "presolve", primary.get("message", "presolve-related failure")
    if "cycle" in text or "degener" in text or "bland" in text:
        return "degeneracy/cycling", primary.get("message", "degeneracy/cycling diagnostic")
    if status in {"NUMERICAL_ERROR", "ERROR"} or cert_status in {"NUMERICAL_ERROR", "ERROR"}:
        return "numerical", primary.get("message", "numerical or harness error")
    if not verification.get("pass"):
        return "numerical", verification.get("detail", "independent verifier failed")
    return "other", verification.get("detail", "objective/reference criterion failed")


def reference_details(entry: dict | None) -> tuple[float | None, str, int | None, bool | None]:
    reference = (entry or {}).get("reference") or {}
    value = reference.get("value")
    try:
        value = float(value) if value is not None else None
    except (TypeError, ValueError):
        value = None
    digits = reference.get("significant_digits")
    try:
        digits = int(digits) if digits is not None else None
    except (TypeError, ValueError):
        digits = None
    approximate = reference.get("approximate")
    if approximate is not None:
        approximate = bool(approximate)
    return (
        value,
        str(reference.get("kind", "opt")) if reference else "",
        digits,
        approximate,
    )

def refresh_record(row: dict, path: Path, entry: dict | None) -> dict:
    """Recompute verifier fields from raw output without rerunning a case."""
    ref = original_ref_from_mps(path)
    primary_payload = None
    primary_path = RAW_DIR / str(row["instance"]) / "primary.json"
    if primary_path.exists():
        try:
            primary_payload = json.loads(primary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            primary_payload = None
    certificate_path = RAW_DIR / str(row["instance"]) / "certificate.json"
    payload = None
    if certificate_path.exists():
        try:
            payload = json.loads(certificate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
    verification = independent_verify(ref, payload)
    reference, reference_kind, reference_digits, reference_approximate = reference_details(entry)
    if reference is None:
        reference = row.get("reference_objective")
        reference_kind = row.get("reference_kind") or ""
        reference_digits = row.get("reference_significant_digits")
        reference_approximate = row.get("reference_approximate")
    cert_obj = verification.get("objective")
    highs_obj = row.get("highs_objective")
    ref_diff = published_rel_diff(cert_obj, reference, ref.offset)
    highs_ref_diff = published_rel_diff(highs_obj, reference, ref.offset)
    highs_diff = rel_diff(cert_obj, highs_obj)
    primary_status = str(row.get("sovereign_status") or "ERROR")
    certificate_status = str(row.get("certificate_status") or "ERROR")
    passed = (
        bool(verification.get("pass"))
        and str(row.get("highs_status") or "ERROR") == "OPTIMAL"
        and highs_obj is not None
        and highs_diff is not None
        and highs_diff <= OBJECTIVE_TOL
    )
    reference_note = (
        "reference value differs from readme; certificate + HiGHS agree"
        if reference is not None and ref_diff is not None and ref_diff > OBJECTIVE_TOL
        else ""
    )
    row.update(
        {
            "rows": ref.shape["rows"],
            "cols": ref.shape["cols"],
            "nnz": ref.shape["nnz"],
            "reference_objective": reference,
            "reference_offset": ref.offset,
            "reference_kind": reference_kind,
            "reference_source": NETLIB_REFERENCE_SOURCE if reference is not None else "",
            "reference_significant_digits": reference_digits,
            "reference_approximate": reference_approximate,
            "reference_note": reference_note,
            "sovereign_verified_status": "OPTIMAL"
            if passed
            else ("FEASIBLE" if verification.get("pass") else "UNKNOWN"),
            "verifier_pass": bool(verification.get("pass")),
            "sovereign_lp_diagnostics": json.dumps(
                (primary_payload or {}).get("lp_diagnostics", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "certificate_lp_diagnostics": json.dumps(
                (payload or {}).get("lp_diagnostics", {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "verifier_primal_infeasibility": verification.get("primal_infeasibility"),
            "verifier_dual_infeasibility": verification.get("dual_infeasibility"),
            "verifier_complementarity": verification.get("complementarity"),
            "verifier_objective": cert_obj,
            "verifier_objective_reference_rel_diff": ref_diff,
            "verifier_objective_highs_rel_diff": highs_diff,
            "highs_objective_reference_rel_diff": highs_ref_diff,
        }
    )
    if passed:
        row["failure_cause"] = ""
        row["failure_detail"] = (
            reference_note
            or "verified original-model KKT certificate and HiGHS objective comparison passed"
        )
    elif verification.get("pass") and (
        str(row.get("highs_status") or "ERROR") != "OPTIMAL"
        or highs_obj is None
        or highs_diff is None
        or highs_diff > OBJECTIVE_TOL
    ):
        row["failure_cause"] = "HiGHS comparison mismatch"
        row["failure_detail"] = (
            f"highs_status={row.get('highs_status')}; highs_rel_diff={highs_diff}; "
            f"pass threshold={OBJECTIVE_TOL:.1e}"
        )
    else:
        primary = {
            "status": primary_status,
            "message": (primary_payload or {}).get("message", ""),
            "payload": primary_payload,
        }
        certificate = {"status": certificate_status}
        row["failure_cause"], row["failure_detail"] = classify_failure(
            primary, certificate, verification, str(row.get("suite", "netlib"))
        )
        row["failure_detail"] = f"{row['failure_detail']}; verifier={verification.get('detail', '')}"
    (RAW_DIR / str(row["instance"]) / "record.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8"
    )
    return row


def run_one(
    binary: str,
    suite: str,
    name: str,
    path: Path,
    entry: dict | None,
    limit: float,
    pin_highs: list[int],
) -> dict:
    ref = original_ref_from_mps(path)
    sha = sha256_file(path)
    instance_dir = RAW_DIR / name
    primary = run_sovereign(binary, path, instance_dir / "primary.json", limit, presolve=True)
    # A second no-presolve lane provides an original-space dual certificate.
    # Its time is reported separately and never replaces the primary timing.
    if primary.get("status") in {"OPTIMAL", "FEASIBLE"}:
        certificate = run_sovereign(
            binary, path, instance_dir / "certificate.json", limit, presolve=False
        )
    else:
        certificate = {
            "status": "NOT_RUN",
            "time": 0.0,
            "wall": 0.0,
            "payload": None,
            "message": "certificate lane skipped because primary solve was not a candidate",
        }
    verification = independent_verify(ref, certificate.get("payload"))
    reference, reference_kind, reference_digits, reference_approximate = reference_details(entry)
    primary_payload = primary.get("payload") or {}
    cert_payload = certificate.get("payload") or {}
    highs = run_highs(str(path), "LP", limit, pin_highs)
    primary_obj = primary_payload.get("objective_value")
    cert_obj = verification.get("objective")
    highs_obj = highs.get("objective")
    ref_diff = published_rel_diff(cert_obj, reference, ref.offset)
    highs_ref_diff = published_rel_diff(highs_obj, reference, ref.offset)
    highs_diff = rel_diff(cert_obj, highs_obj)

    if suite == "netlib":
        passed = (
            bool(verification.get("pass"))
            and highs.get("status") == "OPTIMAL"
            and highs_obj is not None
            and highs_diff is not None
            and highs_diff <= OBJECTIVE_TOL
        )
        verified_status = "OPTIMAL" if passed else ("FEASIBLE" if verification.get("pass") else "UNKNOWN")
    else:
        # No Farkas certificate is currently serialized by the LP result. A
        # raw INFEASIBLE label is therefore retained as evidence but never
        # promoted to an independently verified status.
        passed = False
        verified_status = "UNKNOWN"

    if passed:
        failure_cause, failure_detail = (
            "",
            "verified original-model KKT certificate and HiGHS objective comparison passed",
        )
    elif suite == "netlib" and verification.get("pass"):
        failure_cause = "HiGHS comparison mismatch"
        failure_detail = (
            f"highs_status={highs.get('status')}; highs_rel_diff={highs_diff}; "
            f"pass threshold={OBJECTIVE_TOL:.1e}"
        )
    else:
        failure_cause, failure_detail = classify_failure(primary, certificate, verification, suite)
    reference_note = (
        "reference value differs from readme; certificate + HiGHS agree"
        if reference is not None and ref_diff is not None and ref_diff > OBJECTIVE_TOL and passed
        else ""
    )
    presolve = primary_payload.get("presolve") or {}
    row = {
        "suite": suite,
        "instance": name,
        "source_path": path.relative_to(ROOT).as_posix(),
        "sha256": sha,
        "bytes": path.stat().st_size,
        "rows": ref.shape["rows"],
        "cols": ref.shape["cols"],
        "nnz": ref.shape["nnz"],
        "reference_objective": reference,
        "reference_offset": ref.offset,
        "reference_kind": reference_kind,
        "reference_source": NETLIB_REFERENCE_SOURCE if reference is not None else "",
        "reference_significant_digits": reference_digits,
        "reference_approximate": reference_approximate,
        "reference_note": reference_note,
        "sovereign_status": primary.get("status"),
        "sovereign_verified_status": verified_status,
        "sovereign_objective": primary_obj,
        "sovereign_lp_diagnostics": json.dumps(
            primary_payload.get("lp_diagnostics", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "certificate_lp_diagnostics": json.dumps(
            (certificate.get("payload") or {}).get("lp_diagnostics", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "sovereign_time_s": primary.get("time"),
        "sovereign_wall_s": primary.get("wall"),
        "sovereign_iterations": primary_payload.get("iterations"),
        "sovereign_presolve_fixed_variables": presolve.get("fixed_variables", 0),
        "sovereign_presolve_substituted_variables": presolve.get("substituted_variables", 0),
        "sovereign_presolve_removed_constraints": presolve.get("removed_constraints", 0),
        "sovereign_presolve_tightened_bounds": presolve.get("tightened_bounds", 0),
        "sovereign_presolve_passes": presolve.get("passes", 0),
        "certificate_status": certificate.get("status"),
        "certificate_time_s": certificate.get("time"),
        "certificate_wall_s": certificate.get("wall"),
        "certificate_iterations": cert_payload.get("iterations"),
        "verifier_pass": bool(verification.get("pass")),
        "verifier_primal_infeasibility": verification.get("primal_infeasibility"),
        "verifier_dual_infeasibility": verification.get("dual_infeasibility"),
        "verifier_complementarity": verification.get("complementarity"),
        "verifier_objective": cert_obj,
        "verifier_objective_reference_rel_diff": ref_diff,
        "verifier_objective_highs_rel_diff": highs_diff,
        "highs_status": highs.get("status"),
        "highs_objective": highs_obj,
        "highs_time_s": highs.get("time"),
        "highs_iterations": highs.get("iterations"),
        "highs_objective_reference_rel_diff": highs_ref_diff,
        "failure_cause": failure_cause,
        "failure_detail": (
            reference_note
            or "verified original-model KKT certificate and HiGHS objective comparison passed"
        )
        if passed
        else f"{failure_detail}; verifier={verification.get('detail', '')}",
    }
    (instance_dir / "record.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def expected_missing(found: dict[str, dict[str, Path]], references: dict[str, dict[str, dict]]) -> dict[str, list[str]]:
    return {
        suite: sorted(set(references.get(suite, {})) - set(found.get(suite, {})), key=str.casefold)
        for suite in SUITES
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.{digits}g}"
    return str(value)


def sgm(times: list[float], shift: float = SGM_SHIFT) -> float | None:
    values = [max(float(t), 0.0) for t in times if t is not None and math.isfinite(float(t))]
    if not values:
        return None
    return math.exp(sum(math.log(t + shift) for t in values) / len(values)) - shift


def performance_profile(rows: list[dict]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ratios = {"Sovereign": [], "HiGHS": []}
    for row in rows:
        so_ok = row["sovereign_verified_status"] == "OPTIMAL"
        hi_ok = str(row.get("highs_status", "")).upper() == "OPTIMAL"
        times = {
            "Sovereign": float(row["sovereign_time_s"]) if so_ok and row["sovereign_time_s"] else math.inf,
            "HiGHS": float(row["highs_time_s"]) if hi_ok and row["highs_time_s"] else math.inf,
        }
        best = min(times.values())
        for key in ratios:
            ratios[key].append(times[key] / best if math.isfinite(best) else math.inf)
    if not rows:
        return False
    taus = [10 ** (i / 50) for i in range(151)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for key, color in (("Sovereign", "#0F766E"), ("HiGHS", "#64748B")):
        values = [sum(r <= tau for r in ratios[key]) / len(rows) for tau in taus]
        ax.step(taus, values, where="post", label=key, color=color, linewidth=2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("within this factor of the faster verified solver")
    ax.set_ylabel("fraction of instances")
    ax.set_title(f"Netlib performance profile ({len(rows)} instances)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    return True


def write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows: list[dict],
    missing: dict[str, list[str]],
    metadata: dict[str, str],
    limit: float,
    plot_available: bool,
) -> None:
    solved = [row for row in rows if row["sovereign_verified_status"] == "OPTIMAL"]
    failures = [row for row in rows if row["failure_cause"]]
    so_times = [
        min(float(row["sovereign_time_s"] or limit), limit)
        if row["sovereign_verified_status"] == "OPTIMAL"
        else limit
        for row in rows
    ]
    hi_times = [
        min(float(row["highs_time_s"] or limit), limit)
        if str(row["highs_status"]).upper() == "OPTIMAL"
        else limit
        for row in rows
    ]
    causes: dict[str, int] = {}
    for row in failures:
        causes[row["failure_cause"]] = causes.get(row["failure_cause"], 0) + 1
    lines = [
        "# Reproducible Netlib sweep",
        "",
        "Generated by `benchmarks/runners/run_netlib_sweep.py`.",
        "No solver algorithm, pricing rule, crossover, perturbation, or tolerance was changed for this sweep; only result export and diagnostics were extended.",
        "",
        "## Protocol",
        "",
        "- Suite: local feasible Netlib files. The existing harness `infeasible` directory is MIPLIB 2017, not a Netlib set, and is excluded.",
        f"- Wall-clock limit: `{limit:g}` seconds per Sovereign and HiGHS run.",
        "- One fixed solver thread; CUDA disabled.",
        f"- Independent verifier tolerance: `{VERIFY_TOL:.1e}` for scaled primal, dual and complementarity.",
        f"- Pass criterion: the independent verifier passes on the ORIGINAL model and the verified objective agrees with HiGHS to relative difference `<= {OBJECTIVE_TOL:.1e}`.",
        f"- The published Netlib objective is secondary documentation only; its source is `{NETLIB_REFERENCE_SOURCE}` and it does not decide pass/fail.",
        "- Exact Koch 2004 values were not found in the local corpus or reports, so they were not used.",
        "- Primary solve runs with presolve enabled. A separate no-presolve run supplies the original-space LP certificate; its time is reported separately.",
        "- Raw solver outputs and stderr logs are under `results/netlib-raw/`.",
        "",
        "## Reproducibility",
        "",
    ]
    for key in (
        "commit",
        "working_tree",
        "platform",
        "machine",
        "processor",
        "python",
        "highspy",
        "binary",
        "fixed_threads",
        "build_cache",
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
    ):
        lines.append(f"- {key}: `{metadata.get(key, 'unavailable')}`")
    lines += [
        "",
        "## Results",
        "",
        f"- Local rows recorded: **{len(rows)}**",
        f"- Independently verified Netlib optimum rows: **{len(solved)}**",
        f"- Rows with a failure or timeout: **{len(failures)}**",
        f"- Sovereign shifted geometric mean (10 s shift, failures capped at limit): **{fmt(sgm(so_times), 8)} s**",
        f"- HiGHS shifted geometric mean (10 s shift, failures capped at limit): **{fmt(sgm(hi_times), 8)} s**",
        "",
        "| Failure cause | Count |",
        "|---|---:|",
    ]
    for cause, count in sorted(causes.items()):
        lines.append(f"| {cause} | {count} |")
    if not causes:
        lines.append("| None | 0 |")
    finnis_rows = [row for row in rows if row["instance"].casefold() == "finnis"]
    if finnis_rows:
        finnis = finnis_rows[0]
        lines += [
            "",
            "## finnis dual-simplex diagnostic",
            "",
            "The temporary-bound handoff is diagnostic evidence, not an LP verdict. "
            "The no-presolve certificate lane is the original-coordinate verifier lane.",
        ]
        for lane, field in (
            ("primary presolve-on lane", "sovereign_lp_diagnostics"),
            ("no-presolve certificate lane", "certificate_lp_diagnostics"),
        ):
            try:
                payload = json.loads(finnis.get(field) or "{}")
            except (TypeError, json.JSONDecodeError):
                payload = {}
            dual = payload.get("dual_simplex")
            if isinstance(dual, dict):
                lines.append(
                    f"- `{lane}`: {dual.get('stop_reason') or 'no stop reason recorded'}"
                )
            else:
                lines.append(f"- `{lane}`: no dual-simplex diagnostic payload was returned.")
    pilot_rows = [row for row in rows if row["instance"].casefold() == "pilot87"]
    if pilot_rows:
        try:
            pilot_diagnostics = json.loads(pilot_rows[0].get("sovereign_lp_diagnostics") or "{}")
        except (TypeError, json.JSONDecodeError):
            pilot_diagnostics = {}
        lines += [
            "",
            "## pilot87 diagnostic snapshot",
            "",
            "Diagnostics are serialized in the `sovereign_lp_diagnostics` CSV column and in the raw primary JSON.",
        ]
        for method in ("dual_simplex", "ipm", "revised_simplex"):
            diag = pilot_diagnostics.get(method)
            if not isinstance(diag, dict):
                lines.append(f"- `{method}`: no diagnostic payload was returned.")
                continue
            before = (
                f"{fmt(diag.get('coefficient_min_abs_before'))}.."
                f"{fmt(diag.get('coefficient_max_abs_before'))}"
            )
            after = (
                f"{fmt(diag.get('coefficient_min_abs_after'))}.."
                f"{fmt(diag.get('coefficient_max_abs_after'))}"
            )
            lines.append(
                f"- `{method}`: scaling_applied=`{diag.get('scaling_applied')}`, "
                f"coefficient abs range before=`{before}`, after=`{after}`, "
                f"degenerate_pivots=`{diag.get('degenerate_pivots')}`, "
                f"refactorizations=`{diag.get('refactorizations')}`."
            )
            if method == "ipm":
                stall_iteration = diag.get("stall_iteration", -1)
                if stall_iteration is None or int(stall_iteration) < 0:
                    stall = "not detected"
                else:
                    stall = (
                        f"iteration {stall_iteration}, mu={fmt(diag.get('stall_mu'))}, "
                        f"primal_step={fmt(diag.get('stall_primal_step'))}, "
                        f"dual_step={fmt(diag.get('stall_dual_step'))}, "
                        f"gap={fmt(diag.get('stall_gap'))}"
                    )
                lines.append(
                    f"  IPM gap stall: **{stall}**; final residuals "
                    f"primal=`{fmt(diag.get('final_primal_residual'))}`, "
                    f"dual=`{fmt(diag.get('final_dual_residual'))}`, "
                    f"gap=`{fmt(diag.get('final_gap'))}`."
                )
        lines.append("")
    reference_notes = [
        row for row in rows if row.get("reference_note")
    ]
    lines += [
        "",
        "## Published reference values (secondary)",
        "",
        f"Source for `reference_objective` and its metadata: `{NETLIB_REFERENCE_SOURCE}`.",
        "The following published values differ from the verified certificate; the certificate and HiGHS agree, so these rows are verified passes.",
        "",
        "| Instance | Published readme objective | Certificate objective | Certificate/readme relative difference | Note |",
        "|---|---:|---:|---:|---|",
    ]
    if reference_notes:
        for row in sorted(reference_notes, key=lambda item: item["instance"].casefold()):
            lines.append(
                f"| {row['instance']} | {fmt(row.get('reference_objective'))} | "
                f"{fmt(row.get('verifier_objective'))} | "
                f"{fmt(row.get('verifier_objective_reference_rel_diff'))} | "
                f"{row['reference_note']} |"
            )
    else:
        lines.append("| - | - | - | - | None |")
    lines += ["", "## Missing instances", ""]
    for suite in SUITES:
        label = SUITE_LABELS[suite]
        names = missing.get(suite, [])
        lines.append(f"### {label}")
        lines.append("")
        if names:
            lines.append(f"Missing locally ({len(names)}): `{', '.join(names)}`")
        else:
            lines.append("No instances are missing relative to the local corpus manifest.")
        lines.append("")
    lines += [
        "To fetch the public corpus without inventing local data:",
        "",
        "```powershell",
        "python benchmarks/tools/fetch_coverage_corpus.py --suite netlib",
        "```",
        "",
        "## Every failure",
        "",
        "| Suite | Instance | Sovereign status | Certificate status | Verifier status | Cause | Detail |",
        "|---|---|---|---|---|---|---|",
    ]
    if failures:
        for row in sorted(failures, key=lambda item: (item["suite"], item["instance"].casefold())):
            detail = str(row["failure_detail"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['suite']} | {row['instance']} | {row['sovereign_status']} | "
                f"{row['certificate_status']} | {row['sovereign_verified_status']} | "
                f"{row['failure_cause']} | {detail} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | None |")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- CSV: `{CSV_PATH.relative_to(ROOT).as_posix()}`",
        f"- Performance profile: `{PLOT_PATH.relative_to(ROOT).as_posix()}`"
        if plot_available
        else "- Performance profile: not generated; install `matplotlib` and rerun.",
        "- SHA-256 for every local instance is recorded in the CSV.",
        "- Statuses in `sovereign_verified_status` come from the original-model independent verifier plus the HiGHS comparison; raw solver and certificate-lane statuses are retained separately.",
        "- The pre-rebaseline report and CSV are preserved under `results/netlib-history/`.",
        "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", help="Sovereign executable; otherwise use build64/build")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_LIMIT)
    parser.add_argument("--only", nargs="*", help="run only these case names")
    parser.add_argument("--force", action="store_true", help="ignore existing raw records and rerun")
    parser.add_argument("--no-run", action="store_true", help="only write a report from existing raw records")
    parser.add_argument("--pin-highs", default="4,5")
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")

    found, references, corpus = discover_files()
    missing = expected_missing(found, references)
    names_filter = {name.casefold() for name in args.only or []}
    binary = args.binary
    if not binary:
        candidates = [
            ROOT / "build64" / "solver" / ("sovereign.exe" if os.name == "nt" else "sovereign"),
            ROOT / "build" / "solver" / ("sovereign.exe" if os.name == "nt" else "sovereign"),
        ]
        binary = next((str(path) for path in candidates if path.exists()), None)
    if not args.no_run and not binary:
        parser.error("Sovereign binary not found; build it or pass --binary")
    pin_highs = [int(value) for value in args.pin_highs.split(",") if value.strip()]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for suite in SUITES:
        for name, path in sorted(found[suite].items(), key=lambda item: item[0]):
            if names_filter and name not in names_filter:
                continue
            record_path = RAW_DIR / name / "record.json"
            if not args.force and record_path.exists():
                try:
                    cached = json.loads(record_path.read_text(encoding="utf-8"))
                    rows.append(refresh_record(cached, path, references.get(suite, {}).get(name)))
                    continue
                except (OSError, json.JSONDecodeError):
                    pass
            if args.no_run:
                continue
            entry = references.get(suite, {}).get(name)
            print(f"[{suite}] {name}", flush=True)
            try:
                rows.append(run_one(binary, suite, name, path, entry, args.time_limit, pin_highs))
            except Exception as exc:
                fallback = {field: None for field in CSV_FIELDS}
                fallback.update(
                    {
                        "suite": suite,
                        "instance": name,
                        "source_path": path.relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                        "sovereign_verified_status": "UNKNOWN",
                        "failure_cause": "other",
                        "failure_detail": f"harness error: {exc}",
                    }
                )
                rows.append(fallback)
    rows.sort(key=lambda row: (row.get("suite", ""), str(row.get("instance", "")).casefold()))
    if rows:
        write_csv(rows)
    plot_available = performance_profile(rows)
    metadata = environment_metadata(parse_highs_version(), binary)
    write_report(rows, missing, metadata, args.time_limit, plot_available)
    print(f"wrote {CSV_PATH}")
    print(f"wrote {MD_PATH}")
    if missing["netlib"]:
        print("missing local instances; see netlib.md for the exact fetch command", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
