"""Live benchmark runs: a run is a batch of ordinary worker jobs plus optional HiGHS reference jobs.

HiGHS never solves production jobs. Reference jobs are only handed to workers that
advertise the "highs" reference solver, and their results are shown next to
Sovereign's, never in place of them.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

from api.datasets import CASES, ROOT, dataset
from api.routing import summarize

RELAXED_MIPLIB = ("flugpl", "gt2", "b-ball", "pk1", "gen-ip016")

PROFILES = {
    "lp_simplex": {"label": "Dual simplex", "kind": "LP", "config": {"algorithm": "simplex"}},
    "lp_ipm": {"label": "Interior point", "kind": "LP", "config": {"algorithm": "ipm"}},
    "lp_auto": {"label": "Automatic", "kind": "LP", "config": {"algorithm": "auto"}},
    "milp_bc": {"label": "Branch & cut", "kind": "MILP",
                "config": {"algorithm": "auto", "milpMethod": "branch_and_cut", "branchRule": "strong"}},
    "milp_bb": {"label": "Branch & bound", "kind": "MILP",
                "config": {"algorithm": "auto", "milpMethod": "branch_and_bound", "branchRule": "most_fractional"}},
    "qp_ipm": {"label": "QP interior point", "kind": "QP", "config": {"qpAlgorithm": "ipm"}},
    "qp_fw": {"label": "Frank-Wolfe", "kind": "QP", "config": {"qpAlgorithm": "frank_wolfe"}},
    "lp_ipm_cpu": {"label": "Interior point · CPU baseline", "kind": "LP", "config": {"algorithm": "ipm"}},
    "qp_ipm_cpu": {"label": "QP interior point · CPU baseline", "kind": "QP", "config": {"qpAlgorithm": "ipm"}},
}
# Branch and bound solves node LPs with the dual simplex, so MILP never reaches
# the GPU interior-point factorization. The baselines pin interior point to the
# CPU so one run can time the same solve on both devices.
CPU_ONLY_PROFILES = {"lp_simplex", "qp_fw", "milp_bc", "milp_bb", "lp_ipm_cpu", "qp_ipm_cpu"}

PRESETS = [
    {"id": "quick", "label": "Quick demo",
     "description": "One model from every family, up to a 1,500-row multi-plant product mix. One to two minutes on a laptop.",
     "datasets": ["afiro", "flugpl-lp", "gt2-lp", "kuhn_degeneracy", "illconditioned", "weak_lp_relaxation",
                  "industrial_blending_lp", "industrial_logistics_milp", "industrial_product_mix_lp", "sample_qp",
                  "transport_20x20"],
     "profiles": ["lp_simplex", "lp_ipm", "milp_bc", "qp_ipm"], "device": "auto", "timeLimitSeconds": 60},
    {"id": "lp", "label": "Netlib + MIPLIB relaxations", "description": "Official public files, solved as LPs.",
     "datasets": ["afiro", *[f"{n}-lp" for n in RELAXED_MIPLIB]], "profiles": ["lp_simplex", "lp_ipm"]},
    {"id": "robustness", "label": "Robustness", "description": "Degenerate, ill-conditioned and weak-relaxation models.",
     "datasets": ["kuhn_degeneracy", "illconditioned", "weak_lp_relaxation"],
     "profiles": ["lp_simplex", "lp_ipm", "milp_bc", "milp_bb"]},
    {"id": "scale", "label": "Scale sweep", "description": "Transport LPs from 400 to 22,500 variables.",
     "datasets": ["transport_20x20", "transport_50x50", "transport_100x100", "transport_150x150"],
     "profiles": ["lp_simplex", "lp_ipm", "lp_auto"]},
    {"id": "industrial", "label": "Industrial examples", "description": "Refinery, blending, power dispatch and logistics.",
     "datasets": ["industrial_refinery_lp", "industrial_blending_lp", "industrial_power_dispatch_lp",
                  "industrial_logistics_milp"],
     "profiles": ["lp_simplex", "lp_ipm", "milp_bc"]},
    {"id": "qp", "label": "Quadratic", "description": "Convex QP examples with the Hessian passed to both solvers.",
     "datasets": ["sample_qp", "qp_ge"], "profiles": ["qp_ipm"]},
    {"id": "gpu", "label": "GPU speed-up",
     "description": "Interior point on CUDA against the CPU's sparse factorization. The GPU wins on dense systems. Needs a CUDA machine.",
     "datasets": ["plan_1000x1500", "plan_1500x2200", "portfolio_qp_600"],
     "profiles": ["lp_ipm", "lp_ipm_cpu", "qp_ipm", "qp_ipm_cpu"], "device": "cuda", "timeLimitSeconds": 60},
    {"id": "sparse", "label": "Sparse scale",
     "description": "Production-planning LPs with 2,100 and 10,200 rows (sparse interior point and dual simplex) "
                    "and portfolio QPs with 5,000 and 20,000 assets (sparse QP interior point).",
     "datasets": ["staircase_20x100", "staircase_50x200", "portfolio_qp_5000", "portfolio_qp_20000"],
     "profiles": ["lp_ipm", "lp_simplex", "qp_ipm"],
     "timeLimitSeconds": 60},
    {"id": "miplib", "label": "Official MIPLIB MILPs", "description": "Full integer problems. Some reach the time limit.",
     "datasets": list(RELAXED_MIPLIB), "profiles": ["milp_bc"]},
]


def relax(text: str) -> str:
    model = json.loads(text)
    for var in model["variables"]:
        if var.get("type") == "binary":
            var["lower_bound"] = max(float(var.get("lower_bound", 0.0)), 0.0)
            var["upper_bound"] = min(float(var.get("upper_bound", 1.0)), 1.0)
        var["type"] = "continuous"
    model["problem_type"] = "LP"
    return json.dumps(model)


def _relaxed_case(name: str) -> dict:
    path = ROOT / "benchmarks" / "datasets" / "miplib" / "official" / f"{name}.json"
    text = relax(path.read_text(encoding="utf-8"))
    return {"id": f"{name}-lp", "suite": "MIPLIB LP relaxation",
            "source": f"https://miplib.zib.de/instance_details_{name}.html (integrality removed)",
            "modelFormat": "json", "shape": summarize(text, "json"), "modelJson": text}


def lab_dataset(dataset_id: str) -> dict:
    if dataset_id.endswith("-lp") and dataset_id[:-3] in RELAXED_MIPLIB:
        return _relaxed_case(dataset_id[:-3])
    return dataset(dataset_id)


@lru_cache(maxsize=1)
def _catalogue() -> tuple:
    ids = [name for name, *_ in CASES]
    for name in RELAXED_MIPLIB:
        ids.insert(ids.index(name), f"{name}-lp")
    items = []
    for dataset_id in ids:
        try:
            item = lab_dataset(dataset_id)
        except (KeyError, OSError):
            continue
        item.pop("modelJson")
        item.pop("sha256", None)
        items.append(item)
    return tuple(items)


def catalogue() -> dict:
    known = {item["id"] for item in _catalogue()}
    presets = [{**p, "datasets": [d for d in p["datasets"] if d in known]} for p in PRESETS]
    profiles = [{"id": key, "label": p["label"], "kind": p["kind"], "cpuOnly": key in CPU_ONLY_PROFILES}
                for key, p in PROFILES.items()]
    return {"datasets": list(_catalogue()), "profiles": profiles, "presets": presets}


def kind_of(shape: dict) -> str:
    return str(shape.get("problem_type", "LP")).upper()


def compact_shape(shape: dict) -> dict:
    return {k: shape.get(k) for k in ("columns", "rows", "nonzeros", "integer_variables", "problem_type")}


def summarize_result(result: dict | None, message: str | None) -> dict:
    """The fields a benchmark row needs, so polling never reloads full solutions."""
    result = result or {}
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    status = result.get("status")
    if not status and message:
        status = "TIME_LIMIT" if "Time limit" in message else "ERROR"
    return {
        "status": status,
        "objective": result.get("objective_value"),
        "runtimeSeconds": result.get("runtime_seconds"),
        "verified": verification.get("is_valid") is True,
        "gpuUsed": bool(result.get("gpu_used")),
        "gpuOperations": result.get("gpu_operations") or 0,
        "device": result.get("requested_device"),
        "iterations": result.get("iterations"),
        "nodes": result.get("nodes"),
        "solver": result.get("solver", "sovereign"),
        "version": result.get("version"),
        "message": (message or result.get("message") or "")[:500],
    }


def _same_objective(row: dict, ref: dict | None) -> bool:
    a, b = row.get("objective"), (ref or {}).get("objective")
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)


def _agree(row: dict, ref: dict | None) -> bool | None:
    if not ref or ref.get("status") is None or row.get("status") is None:
        return None
    status, ref_status = row["status"], ref["status"]
    conclusive = ("OPTIMAL", "INFEASIBLE", "UNBOUNDED")
    if status not in conclusive and ref_status not in conclusive:
        return None  # neither solver finished, so there is no answer to compare
    if status == "OPTIMAL" and ref_status == "OPTIMAL":
        return bool(row.get("verified")) and _same_objective(row, ref)
    if status in ("INFEASIBLE", "UNBOUNDED") and status == ref_status:
        return True
    return False


def _job_outcome(job: dict) -> dict:
    summary = job.get("summary")
    if isinstance(summary, str):
        summary = json.loads(summary) if summary else None
    if summary:
        return summary
    state = job["state"]
    if state == "CANCELLED":
        return {"status": "CANCELLED", "message": job.get("message") or ""}
    if state == "FAILED":
        return summarize_result(None, job.get("message") or "Failed")
    return {"status": None}


def build_rows(jobs: list[dict]) -> list[dict]:
    references, rows = {}, []
    for job in jobs:
        bench = json.loads(job["bench"]) if isinstance(job.get("bench"), str) else (job.get("bench") or {})
        entry = {"job": job, "bench": bench, "outcome": _job_outcome(job)}
        if bench.get("role") == "reference":
            references[bench["dataset"]] = entry
        else:
            rows.append(entry)
    out = []
    for entry in rows:
        job, bench, outcome = entry["job"], entry["bench"], entry["outcome"]
        ref_entry = references.get(bench["dataset"])
        reference = None
        if ref_entry:
            ref_out = ref_entry["outcome"]
            reference = {"state": ref_entry["job"]["state"], "status": ref_out.get("status"),
                         "objective": ref_out.get("objective"), "runtimeSeconds": ref_out.get("runtimeSeconds"),
                         "version": ref_out.get("version"), "message": ref_out.get("message", "")}
        row = {"id": job["id"], "dataset": bench["dataset"], "suite": bench.get("suite"),
               "kind": bench.get("kind"), "shape": bench.get("shape"), "profile": bench.get("profile"),
               "profileLabel": bench.get("profileLabel"), "state": job["state"],
               "executionDevice": bench.get("device"), **outcome, "reference": reference}
        row["agrees"] = _agree(row, reference) if job["state"] in ("COMPLETED", "FAILED") else None
        row["sameObjective"] = _same_objective(row, reference)
        out.append(row)
    return out


def summarize_rows(rows: list[dict], jobs: list[dict]) -> dict:
    finished = [r for r in rows if r["state"] in ("COMPLETED", "FAILED", "CANCELLED")]
    # Engines built before the performance-counter timer report sub-millisecond solves as 0 s.
    ratios = [r["runtimeSeconds"] / r["reference"]["runtimeSeconds"] for r in rows
              if r.get("agrees") and r.get("runtimeSeconds") and r["reference"].get("runtimeSeconds")]
    geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else None
    return {
        "jobs": len(jobs),
        "jobsFinished": sum(j["state"] in ("COMPLETED", "FAILED", "CANCELLED") for j in jobs),
        "rows": len(rows),
        "finished": len(finished),
        "running": sum(r["state"] == "SOLVING" for r in rows),
        "queued": sum(r["state"] == "QUEUED" for r in rows),
        "verified": sum(bool(r.get("verified")) for r in finished),
        "compared": sum(r.get("agrees") is not None for r in rows),
        "agreements": sum(r.get("agrees") is True for r in rows),
        "disagreements": sum(r.get("agrees") is False for r in rows),
        "timeLimits": sum(r.get("status") == "TIME_LIMIT" for r in finished),
        "errors": sum(r.get("status") in ("ERROR", "NUMERICAL_ERROR") for r in finished),
        "geomeanTimeRatio": geo,
        "ratioCount": len(ratios),
    }


def recorded_run() -> dict:
    path = ROOT / "benchmarks" / "reports" / "sih-online.json"
    if not path.is_file():
        return {"rows": [], "summary": None, "recorded": None}
    report = json.loads(path.read_text(encoding="utf-8"))
    label_to_profile = {"Revised simplex": "lp_simplex", "Dual simplex": "lp_simplex", "Interior point": "lp_ipm",
                        "Branch & cut / strong": "milp_bc", "Branch & bound / fractional": "milp_bb",
                        "QP interior point": "qp_ipm", "Frank-Wolfe": "qp_fw"}
    rows = []
    for i, r in enumerate(report.get("rows", [])):
        profile = label_to_profile.get(r["profile"])
        if not profile:
            continue
        ref = r.get("reference") or {}
        row = {"id": f"recorded-{i}", "dataset": r["dataset"], "suite": r["suite"],
               "kind": kind_of(r.get("shape", {})), "shape": compact_shape(r.get("shape", {})),
               "profile": profile, "profileLabel": PROFILES[profile]["label"], "state": "COMPLETED",
               "executionDevice": "cpu", "status": r["status"], "objective": r.get("objective"),
               "runtimeSeconds": r.get("runtime_seconds"), "verified": bool(r.get("verified")),
               "gpuUsed": bool(r.get("gpu_used")), "gpuOperations": r.get("gpu_operations") or 0,
               "message": (r.get("message") or "")[:500],
               "reference": {"state": "COMPLETED", "status": ref.get("status"), "objective": ref.get("objective"),
                             "runtimeSeconds": ref.get("runtime_seconds"), "version": ref.get("version"),
                             "message": ""}}
        row["agrees"] = _agree(row, row["reference"])
        row["sameObjective"] = _same_objective(row, row["reference"])
        rows.append(row)
    fake_jobs = [{"state": "COMPLETED"} for _ in rows]
    return {"rows": rows, "summary": summarize_rows(rows, fake_jobs),
            "recorded": {"generatedAt": report.get("generated_at"), "platform": report.get("platform"),
                         "cpuThreads": report.get("cpu_threads"), "gpu": report.get("gpu"),
                         "timeLimitSeconds": report.get("time_limit_seconds"), "note": report.get("note")}}
