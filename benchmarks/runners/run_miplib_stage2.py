"""Reproducible Stage 2 MIPLIB 2017 MILP sweep.

The checked-in manifest fixes the 30-instance selection.  Inputs are fetched
unmodified from MIPLIB when ``--fetch`` is supplied and are never synthesized.
Sovereign and HiGHS read the same original MPS bytes.  Solver status is only
accepted as SOLVED after the independent original-model checks pass.

Example:
  python benchmarks/runners/run_miplib_stage2.py --fetch
  python benchmarks/runners/run_miplib_stage2.py --report-only
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
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "runners"))
import run_coverage as coverage  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "miplib_stage2.json"
INPUT_DIR = ROOT / "benchmarks" / "datasets" / "coverage" / "miplib_stage2"
OUT_DIR = ROOT / "benchmarks" / "reports" / "miplib_stage2"
SOLUTION_NAME = "miplib2017-v37.solu"
MIPLIB_SOLUTION_URL = "https://miplib.zib.de/downloads/miplib2017-v37.solu"
FEAS_TOL = coverage.FEAS_TOL
INT_TOL = coverage.INT_TOL
MIP_GAP = coverage.MIP_GAP
LIMIT = 600.0
GAP_TARGETS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)
PROGRESS_RE = re.compile(
    r"\[bb\].*?\bgap=([-+0-9.eE]+).*?\bt=([-+0-9.eE]+)s"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "sovereign-stage2"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def parse_solu(text: str) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0] not in ("=opt=", "=best=", "=inf=", "=unkn="):
            continue
        kind = fields[0].strip("=")
        refs[fields[1]] = {
            "kind": kind,
            "value": float(fields[2]) if len(fields) > 2 else None,
        }
    return refs


def ensure_solution_file(fetch: bool) -> tuple[Path | None, dict[str, dict[str, Any]], str | None]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / SOLUTION_NAME
    try:
        if not path.exists():
            if not fetch:
                return None, {}, f"missing {path.relative_to(ROOT)}; rerun with --fetch"
            path.write_bytes(fetch_bytes(MIPLIB_SOLUTION_URL))
        return path, parse_solu(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, {}, f"could not fetch {MIPLIB_SOLUTION_URL}: {exc}"


def prepare_inputs(manifest: dict[str, Any], fetch: bool) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    solution_path, references, solution_error = ensure_solution_file(fetch)
    missing: list[dict[str, str]] = []
    if solution_error:
        missing.append({"name": SOLUTION_NAME, "reason": solution_error, "url": MIPLIB_SOLUTION_URL})

    records: list[dict[str, Any]] = []
    for name in manifest["instances"]:
        path = INPUT_DIR / f"{name}.mps"
        url = manifest["instance_url"].format(name=name)
        try:
            if not path.exists():
                if not fetch:
                    raise FileNotFoundError("input is not present; rerun with --fetch")
                import gzip

                path.write_bytes(gzip.decompress(fetch_bytes(url)))
            ref = coverage.ref_from_mps(path)
            ref_value = references.get(name)
            if ref_value is None or ref_value.get("kind") != "opt" or ref_value.get("value") is None:
                raise RuntimeError("no =opt= reference in miplib2017-v37.solu")
            records.append(
                {
                    "name": name,
                    "suite": manifest["suite"],
                    "synthetic": False,
                    "file": path.relative_to(ROOT).as_posix(),
                    "source": url,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    **ref.shape,
                    "reference": ref_value,
                }
            )
        except Exception as exc:
            missing.append({"name": name, "reason": str(exc), "url": url})

    lock = {
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "selection_rule": manifest["selection_rule"],
        "solution_file": solution_path.relative_to(ROOT).as_posix() if solution_path else None,
        "solution_sha256": sha256(solution_path) if solution_path and solution_path.exists() else None,
        "instances": records,
        "missing": missing,
    }
    (OUT_DIR / "corpus.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return records, missing


def find_binary() -> str:
    for path in (
        ROOT / "build-agent" / "solver" / "sovereign.exe",
        ROOT / "build64" / "solver" / "sovereign.exe",
        ROOT / "build" / "solver" / "sovereign.exe",
        ROOT / "build" / "solver" / "sovereign",
    ):
        if path.exists():
            return str(path)
    return coverage.find_sovereign()


def clean_solver_env(limit: float) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SOVEREIGN_")}
    # These are benchmark controls, not alternate algorithms.  They make the
    # time budget meaningful and give both solvers one worker.
    env["SOVEREIGN_TIME_LIMIT"] = str(limit)
    env["SOVEREIGN_PARALLEL_WORKERS"] = "1"
    env["SOVEREIGN_MAX_NODES"] = str(2**31 - 1)
    env["SOVEREIGN_BB_LOG"] = "1"
    env["SOVEREIGN_DISABLE_CUDA"] = "1"
    return env


def run_sovereign(binary: str, model: Path, limit: float, cpus: list[int]) -> dict[str, Any]:
    out, err, wall, killed = coverage._run(
        [binary, "solve", str(model)],
        clean_solver_env(limit),
        limit + 60.0,
        cpus,
    )
    result: dict[str, Any] = {
        "status": "TIME_LIMIT" if killed else "ERROR",
        "time": limit if killed else wall,
        "wall": wall,
        "x": None,
        "stderr": err,
    }
    if killed:
        return result
    try:
        payload = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        result["message"] = (err or out).strip()[-1000:]
        return result

    result.update(
        {
            "status": payload.get("status", "ERROR"),
            "time": float(payload.get("runtime_seconds") or wall),
            "objective": payload.get("objective_value"),
            "x": payload.get("primal"),
            "iterations": payload.get("iterations"),
            "nodes": payload.get("nodes"),
            "gap": payload.get("optimality_gap"),
            "best_bound": (payload.get("mip_diagnostics") or {}).get("best_bound"),
            "mip_diagnostics": payload.get("mip_diagnostics") or {},
            "presolve": payload.get("presolve") or {},
            "message": payload.get("message", ""),
            "warnings": payload.get("warnings") or [],
        }
    )
    result["stdout"] = out
    return result


def highs_worker(model: str, limit: float) -> dict[str, Any]:
    import highspy

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("threads", 1)
    highs.setOptionValue("time_limit", float(limit))
    highs.setOptionValue("mip_rel_gap", MIP_GAP)
    highs.setOptionValue("large_matrix_value", coverage.LARGE_MATRIX_VALUE)
    highs.readModel(model)
    highs.run()
    status = highs.modelStatusToString(highs.getModelStatus()).upper().replace(" ", "_")
    status = {
        "TIME_LIMIT_REACHED": "TIME_LIMIT",
        "PRIMAL_INFEASIBLE_OR_UNBOUNDED": "INFEASIBLE_OR_UNBOUNDED",
        "ITERATION_LIMIT_REACHED": "ITERATION_LIMIT",
    }.get(status, status)
    solution = highs.getSolution()
    info = highs.getInfo()
    valid = bool(solution.value_valid)
    objective = info.objective_function_value if valid else None
    best_bound = getattr(info, "mip_dual_bound", None)
    if status == "OPTIMAL" and objective is not None:
        best_bound = objective if best_bound is None else best_bound
    return {
        "status": status,
        "time": None,
        "objective": objective,
        "best_bound": best_bound if best_bound is not None and math.isfinite(best_bound) else None,
        "x": list(solution.col_value) if valid else None,
        "gap": info.mip_gap if math.isfinite(info.mip_gap) else None,
        "iterations": info.simplex_iteration_count + info.ipm_iteration_count,
        "nodes": info.mip_node_count if info.mip_node_count >= 0 else None,
        "version": highs.version(),
    }


def run_highs(model: Path, limit: float, cpus: list[int]) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--highs-worker",
        str(model),
        "--time-limit",
        str(limit),
    ]
    out, err, wall, killed = coverage._run(command, os.environ.copy(), limit + 60.0, cpus)
    if killed:
        return {"status": "TIME_LIMIT", "time": limit, "wall": wall, "x": None, "stderr": err}
    try:
        result = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"status": "ERROR", "time": wall, "wall": wall, "x": None, "stderr": err}
    result["time"] = wall if result.get("time") is None else result["time"]
    result["wall"] = wall
    result["stderr"] = err
    return result


def time_to_gap(stderr: str, final_gap: float | None, runtime: float, solver: str) -> dict[str, float | None]:
    observed: list[tuple[float, float]] = []
    for match in PROGRESS_RE.finditer(stderr or ""):
        gap, elapsed = float(match.group(1)), float(match.group(2))
        if gap >= 0.0 and math.isfinite(gap):
            observed.append((elapsed, gap))
    result: dict[str, float | None] = {}
    for target in GAP_TARGETS:
        key = f"{target:g}"
        hit = next((elapsed for elapsed, gap in observed if gap <= target), None)
        if hit is None and final_gap is not None and math.isfinite(final_gap) and final_gap <= target:
            # HiGHS does not expose a progress stream in this harness; its
            # final time is the only defensible observation in that case.
            hit = runtime if solver == "highs" else None
        result[key] = hit
    return result


def bound_check(result: dict[str, Any], ref: coverage.RefModel, reference: dict[str, Any]) -> None:
    bound = result.get("best_bound")
    optimum = reference.get("value")
    if bound is None or optimum is None:
        result["best_bound_verified"] = None
        result["best_bound_violation"] = None
        return
    tolerance = coverage.reference_tolerance(reference, "MILP")
    scale = max(1.0, abs(float(bound)), abs(float(optimum)))
    if ref.sense == 1.0:
        violation = max(0.0, float(bound) - float(optimum) - tolerance * scale)
    else:
        violation = max(0.0, float(optimum) - float(bound) - tolerance * scale)
    result["best_bound_verified"] = violation == 0.0
    result["best_bound_violation"] = violation


def classify(result: dict[str, Any], solver: str) -> str:
    if result.get("best_bound_verified") is False:
        return "wrong_prune"
    status = result.get("status", "ERROR")
    check = result.get("check") or {}
    if status == "OPTIMAL" and not check.get("ok", False):
        return "wrong_primal"
    if status == "UNBOUNDED" or status == "INFEASIBLE_OR_UNBOUNDED":
        message = str(result.get("message", "")).lower()
        return "unbounded_by_temporary_bound" if "temporary bound" in message else "wrong_status"
    if status == "NUMERICAL_ERROR" or status == "ERROR":
        return "lp_numerical"
    if status == "ITERATION_LIMIT":
        return "lp_iteration_limit"
    if status == "TIME_LIMIT":
        return "time_limit"
    if status == "FEASIBLE":
        gap = result.get("gap")
        return "weak_bound" if gap is not None and gap > MIP_GAP else "time_limit"
    if status == "OPTIMAL" and not result.get("independent_solved", False):
        return "reference_or_bound_mismatch"
    return "unsolved"


def score_lane(
    entry: dict[str, Any],
    ref: coverage.RefModel,
    result: dict[str, Any],
    limit: float,
    solver: str,
) -> dict[str, Any]:
    scored = coverage.score(dict(entry), "MILP", result, ref, limit)
    bound_check(scored, ref, entry["reference"])
    check = scored.get("check") or {}
    scored["incumbent_verified"] = bool(
        check.get("ok", False) and not scored.get("missing_values", 0)
    )
    scored["strict_incumbent_verified"] = bool(
        check.get("strict_ok", False) and not scored.get("missing_values", 0)
    )
    scored["independent_solved"] = bool(
        scored.get("status") == "OPTIMAL"
        and scored["incumbent_verified"]
        and scored.get("outcome") == "SOLVED"
        and scored.get("best_bound_verified") is not False
    )
    scored["verified_outcome"] = "SOLVED" if scored["independent_solved"] else "NOT_SOLVED"
    scored["failure_class"] = None if scored["independent_solved"] else classify(scored, solver)
    scored["time_to_gap"] = time_to_gap(
        scored.get("stderr", ""),
        scored.get("gap"),
        float(scored.get("time") or limit),
        solver,
    )
    return scored


def run_instance(
    binary: str,
    entry: dict[str, Any],
    limit: float,
    pin_ours: list[int],
    pin_highs: list[int],
) -> dict[str, Any]:
    model = ROOT / entry["file"]
    ref = coverage.ref_from_mps(model)
    ours = run_sovereign(binary, model, limit, pin_ours)
    highs = run_highs(model, limit, pin_highs)

    ours_raw = dict(ours)
    highs_raw = dict(highs)
    row = {
        **entry,
        "feasibility_tolerance": FEAS_TOL,
        "integrality_tolerance": INT_TOL,
        "mip_gap_tolerance": MIP_GAP,
        "sovereign": score_lane(entry, ref, ours, limit, "sovereign"),
        "highs": score_lane(entry, ref, highs, limit, "highs"),
    }
    # Keep complete raw process output outside the summary JSONL, but preserve
    # enough provenance in the row to audit the two lanes.
    raw_dir = OUT_DIR / "raw" / entry["name"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "sovereign.stdout.json").write_text(
        str(ours_raw.get("stdout", "")), encoding="utf-8"
    )
    (raw_dir / "sovereign.stderr.log").write_text(
        str(ours_raw.get("stderr", "")), encoding="utf-8"
    )
    (raw_dir / "highs.stderr.log").write_text(
        str(highs_raw.get("stderr", "")), encoding="utf-8"
    )
    return row


def machine_info(binary: str, limit: float, pin_ours: list[int], pin_highs: list[int]) -> dict[str, Any]:
    try:
        import highspy

        highs_version = highspy.Highs().version()
    except Exception as exc:
        highs_version = f"unavailable: {exc}"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        commit = "unavailable"
        dirty = None
    return {
        "commit": commit,
        "working_tree_dirty": dirty,
        "binary": str(Path(binary).resolve()),
        "binary_bytes": Path(binary).stat().st_size if Path(binary).exists() else None,
        "binary_sha256": sha256(Path(binary)) if Path(binary).exists() else None,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "highs_version": highs_version,
        "time_limit_seconds": limit,
        "threads_sovereign": 1,
        "threads_highs": 1,
        "pin_sovereign": pin_ours,
        "pin_highs": pin_highs,
        "flags": {
            "SOVEREIGN_TIME_LIMIT": str(limit),
            "SOVEREIGN_PARALLEL_WORKERS": "1",
            "SOVEREIGN_MAX_NODES": str(2**31 - 1),
            "SOVEREIGN_BB_LOG": "1",
            "SOVEREIGN_DISABLE_CUDA": "1",
            "HiGHS_threads": 1,
            "HiGHS_mip_rel_gap": MIP_GAP,
        },
        "started": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def load_rows() -> list[dict[str, Any]]:
    path = OUT_DIR / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shifted_geomean(times: list[float], shift: float = 10.0) -> float:
    if not times:
        return float("nan")
    return math.exp(sum(math.log(max(0.0, t) + shift) for t in times) / len(times)) - shift


def capped_time(lane: dict[str, Any], limit: float) -> float:
    return float(lane.get("time") or limit) if lane.get("independent_solved") else limit


def write_profile(rows: list[dict[str, Any]], limit: float) -> bool:
    points: list[dict[str, Any]] = []
    shifted: dict[str, list[float]] = {}
    for row in rows:
        times = {
            "sovereign": capped_time(row["sovereign"], limit),
            "highs": capped_time(row["highs"], limit),
        }
        shifted[row["name"]] = [times["sovereign"] + 10.0, times["highs"] + 10.0]
        best = min(value for value in shifted[row["name"]])
        for solver in ("sovereign", "highs"):
            points.append(
                {
                    "instance": row["name"],
                    "solver": solver,
                    "shifted_time": shifted[row["name"]][0 if solver == "sovereign" else 1],
                    "ratio": shifted[row["name"]][0 if solver == "sovereign" else 1] / best,
                }
            )
    with (OUT_DIR / "performance_profile.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", "solver", "shifted_time", "ratio"])
        writer.writeheader()
        writer.writerows(points)

    try:
        import matplotlib.pyplot as plt

        ratios = {
            solver: sorted(p["ratio"] for p in points if p["solver"] == solver)
            for solver in ("sovereign", "highs")
        }
        taus = sorted(set(r for values in ratios.values() for r in values))
        fig, ax = plt.subplots(figsize=(7, 4))
        for solver, values in ratios.items():
            ax.step(
                taus,
                [sum(value <= tau for value in values) / len(values) for tau in taus],
                where="post",
                label=solver,
            )
        ax.set_xlabel("shifted time ratio (time + 10 s)")
        ax.set_ylabel("fraction of instances")
        ax.set_title("MIPLIB Stage 2 performance profile")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "performance_profile.png", dpi=150)
        plt.close(fig)
        return True
    except Exception as exc:
        (OUT_DIR / "performance_profile.error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return False


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "rows",
        "cols",
        "integers",
        "nnz",
        "bytes",
        "sha256",
        "reference",
        "sovereign_status",
        "sovereign_outcome",
        "sovereign_objective",
        "sovereign_best_bound",
        "sovereign_gap",
        "sovereign_nodes",
        "sovereign_iterations",
        "sovereign_time",
        "sovereign_failure_class",
        "sovereign_cuts_by_family",
        "sovereign_presolve",
        "sovereign_time_to_gap",
        "highs_status",
        "highs_outcome",
        "highs_objective",
        "highs_best_bound",
        "highs_gap",
        "highs_nodes",
        "highs_iterations",
        "highs_time",
        "highs_failure_class",
        "highs_time_to_gap",
    ]
    with (OUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            s, h = row["sovereign"], row["highs"]
            writer.writerow(
                {
                    "name": row["name"],
                    "rows": row["rows"],
                    "cols": row["cols"],
                    "integers": row["integers"],
                    "nnz": row["nnz"],
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                    "reference": row["reference"]["value"],
                    "sovereign_status": s.get("status"),
                    "sovereign_outcome": s.get("verified_outcome"),
                    "sovereign_objective": s.get("objective"),
                    "sovereign_best_bound": s.get("best_bound"),
                    "sovereign_gap": s.get("gap"),
                    "sovereign_nodes": s.get("nodes"),
                    "sovereign_iterations": s.get("iterations"),
                    "sovereign_time": s.get("time"),
                    "sovereign_failure_class": s.get("failure_class"),
                    "sovereign_cuts_by_family": json.dumps(
                        (s.get("mip_diagnostics") or {}).get("cuts_by_family", {}),
                        sort_keys=True,
                    ),
                    "sovereign_presolve": json.dumps(s.get("presolve") or {}, sort_keys=True),
                    "sovereign_time_to_gap": json.dumps(s.get("time_to_gap") or {}, sort_keys=True),
                    "highs_status": h.get("status"),
                    "highs_outcome": h.get("verified_outcome"),
                    "highs_objective": h.get("objective"),
                    "highs_best_bound": h.get("best_bound"),
                    "highs_gap": h.get("gap"),
                    "highs_nodes": h.get("nodes"),
                    "highs_iterations": h.get("iterations"),
                    "highs_time": h.get("time"),
                    "highs_failure_class": h.get("failure_class"),
                    "highs_time_to_gap": json.dumps(h.get("time_to_gap") or {}, sort_keys=True),
                }
            )


def write_report(manifest: dict[str, Any], meta: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    write_csv(rows)
    plot = write_profile(rows, float(meta["time_limit_seconds"]))
    solved = {
        solver: sum(row[solver].get("independent_solved", False) for row in rows)
        for solver in ("sovereign", "highs")
    }
    total = len(rows)
    all_times = {
        solver: [capped_time(row[solver], float(meta["time_limit_seconds"])) for row in rows]
        for solver in ("sovereign", "highs")
    }
    lines = [
        "# Stage 2 MIPLIB 2017 MILP sweep",
        "",
        "This report contains only official MIPLIB instances from the checked-in manifest. "
        "Synthetic fixtures are not included.",
        "",
        f"- Selection rule: {manifest['selection_rule']}",
        f"- Instances with complete inputs and results: {total}/{len(manifest['instances'])}",
        f"- Time limit: {meta['time_limit_seconds']} s per solver",
        f"- Threads: Sovereign {meta['threads_sovereign']}, HiGHS {meta['threads_highs']}",
        f"- Commit: `{meta['commit']}`",
        f"- Working tree dirty at run time: `{meta.get('working_tree_dirty')}`",
        f"- HiGHS: `{meta['highs_version']}`",
        f"- Independent feasibility tolerance: `{FEAS_TOL}`",
        f"- Independent integrality tolerance: `{INT_TOL}`",
        f"- MIP relative-gap tolerance: `{MIP_GAP}`",
        "",
        "## Summary",
        "",
        f"- Sovereign verified solved: **{solved['sovereign']}/{total}**",
        f"- HiGHS verified solved: **{solved['highs']}/{total}**",
        f"- Sovereign shifted geometric mean, 10 s shift: **{shifted_geomean(all_times['sovereign']):.6g} s**",
        f"- HiGHS shifted geometric mean, 10 s shift: **{shifted_geomean(all_times['highs']):.6g} s**",
        f"- Performance profile data: `performance_profile.csv`",
        f"- Performance profile plot: `performance_profile.png`"
        if plot
        else "- Performance profile plot: unavailable; see `performance_profile.error.txt`",
        "",
        "## Missing inputs",
        "",
    ]
    missing = meta.get("missing") or []
    if missing:
        lines.extend(
            f"- `{item['name']}`: {item['reason']} ({item['url']})"
            for item in missing
        )
    else:
        lines.append("- None; all manifest inputs were materialized.")
    lines.extend(
        [
            "",
        "## Instance provenance",
        "",
        ]
    )
    for row in rows:
        lines.append(
            f"- `{row['name']}`: rows={row['rows']}, cols={row['cols']}, "
            f"integers={row['integers']}, nnz={row['nnz']}, bytes={row['bytes']}, "
            f"sha256=`{row['sha256']}`, reference={row['reference']['value']}"
        )

    lines.extend(["", "## Unsolved or unverified results", ""])
    failures = 0
    for row in rows:
        for solver in ("sovereign", "highs"):
            lane = row[solver]
            if lane.get("independent_solved"):
                continue
            failures += 1
            lines.append(
                f"- `{row['name']}` / {solver}: status={lane.get('status')}, "
                f"incumbent_verified={lane.get('incumbent_verified')}, "
                f"best_bound_verified={lane.get('best_bound_verified')}, "
                f"gap={lane.get('gap')}, nodes={lane.get('nodes')}, "
                f"class={lane.get('failure_class')}, time_to_gap={lane.get('time_to_gap')}"
            )
    if failures == 0:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Branch-and-bound status behavior and correctness risk",
            "",
            "- A node LP returning `NUMERICAL_ERROR` or `ITERATION_LIMIT` is retried through "
            "the configured LP fallbacks. If it still fails, the subtree is dropped, a warning "
            "is recorded, and the final result is downgraded to `FEASIBLE` when an incumbent "
            "exists or `NUMERICAL_ERROR` otherwise; it is not treated as infeasible.",
            "- A node LP returning `UNBOUNDED` is propagated as overall MILP `UNBOUNDED`. "
            "The finnis temporary-bound path currently returns LP `NUMERICAL_ERROR`, but any "
            "future temporary-bound path that returns `UNBOUNDED` would be a correctness risk "
            "because the node is not independently proven unbounded.",
            "",
            "## Missing input instructions",
            "",
            "If this report has missing instances, fetch them with:",
            "",
            "```powershell",
            "python benchmarks/runners/run_miplib_stage2.py --fetch",
            "```",
            "",
            "The runner uses the official URL recorded in the manifest and never invents "
            "replacement data.",
        ]
    )
    (OUT_DIR / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="fetch missing official MIPLIB inputs")
    parser.add_argument("--prepare-only", action="store_true", help="fetch and fingerprint inputs without solving")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--time-limit", type=float, default=LIMIT)
    parser.add_argument("--binary", help="explicit Sovereign executable path")
    parser.add_argument("--pin-ours", type=lambda s: [int(v) for v in s.split(",")], default=[2, 3])
    parser.add_argument("--pin-highs", type=lambda s: [int(v) for v in s.split(",")], default=[4, 5])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--highs-worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.highs_worker:
        print(json.dumps(highs_worker(args.highs_worker, args.time_limit)))
        return 0

    manifest = load_manifest()
    records, missing = prepare_inputs(manifest, args.fetch)
    if args.prepare_only:
        if missing:
            print(json.dumps({"missing": missing}, indent=2))
            return 2
        print(f"prepared {len(records)} official MIPLIB inputs")
        return 0
    if missing:
        (OUT_DIR / "missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
        if args.report_only:
            print(json.dumps({"missing": missing}, indent=2))
        else:
            print(f"{len(missing)} required inputs are missing; use --fetch.")
        if not records or not args.report_only:
            if missing and not args.fetch:
                return 2

    if args.report_only:
        rows = load_rows()
        meta_path = OUT_DIR / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else machine_info(
            args.binary or find_binary(), args.time_limit, args.pin_ours, args.pin_highs
        )
        write_report(manifest, meta, rows)
        return 0

    binary = args.binary or find_binary()
    selected = set(args.only or manifest["instances"])
    records = [record for record in records if record["name"] in selected]
    if missing:
        print("Missing instances:")
        for item in missing:
            print(f"  {item['name']}: {item['reason']} ({item['url']})")
        if not records:
            return 2

    meta = machine_info(binary, args.time_limit, args.pin_ours, args.pin_highs)
    meta["manifest"] = MANIFEST.relative_to(ROOT).as_posix()
    meta["missing"] = missing
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    output = OUT_DIR / "results.jsonl"
    completed = set()
    if output.exists() and not args.force:
        completed = {
            json.loads(line)["name"]
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    elif args.force and output.exists():
        output.unlink()

    for index, record in enumerate((r for r in records if r["name"] not in completed), 1):
        print(f"[{index}] {record['name']}", flush=True)
        try:
            row = run_instance(binary, record, args.time_limit, args.pin_ours, args.pin_highs)
        except Exception as exc:
            row = {
                **record,
                "harness_error": str(exc),
                "sovereign": {"status": "ERROR", "verified_outcome": "NOT_SOLVED", "failure_class": "harness_error"},
                "highs": {"status": "ERROR", "verified_outcome": "NOT_SOLVED", "failure_class": "harness_error"},
            }
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"  Sovereign={row['sovereign'].get('verified_outcome')} "
            f"HiGHS={row['highs'].get('verified_outcome')}",
            flush=True,
        )

    rows = load_rows()
    write_report(manifest, meta, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
