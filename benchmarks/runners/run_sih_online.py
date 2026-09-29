"""SIH evidence through the real coordinator and worker. HiGHS is reference-only."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from api.cloud import MAX_MODEL_CHARACTERS
from api.datasets import CASES, dataset
from worker.runner import highs_reference


def write_report(report):
    target = ROOT / "benchmarks" / "reports" / "sih-online.json"
    rows = report["rows"]
    report["summary"] = {"comparisons": len(rows), "verified_optimum_matches": sum(bool(r["match"]) for r in rows),
                         "verified_results": sum(bool(r["verified"]) for r in rows),
                         "numerical_errors": sum(r["status"] == "NUMERICAL_ERROR" for r in rows),
                         "time_limits": sum(r["status"] == "TIME_LIMIT" for r in rows)}
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# SIH 26119: online worker evidence", "", f"Generated: {report['generated_at']}", "",
             report["note"], "", f"{len(rows)} comparisons; {report['summary']['verified_optimum_matches']} verified optimum matches; {report['summary']['numerical_errors']} numerical errors; {report['summary']['time_limits']} timeouts.", "",
             f"Machine: {report['platform']} | {report['cpu_threads']} logical CPUs | {report['gpu']}", "",
             f"Per-job wall-clock limit: {report['time_limit_seconds']} seconds. HiGHS: one CPU thread.", "",
             "Sovereign timings below are solver-reported; timeouts use the enforced wall-clock limit. Full HTTP turnaround is recorded separately in JSON. Single runs, not statistical performance claims.", "",
             "| Dataset | Source | Algorithm | Sovereign status | Objective | Seconds | Verified | HiGHS status | HiGHS objective | HiGHS seconds | Verified optimum match |",
             "|---|---|---|---|---:|---:|---|---|---:|---:|---|"]
    for r in rows:
        ref = r["reference"]
        lines.append(f"| {r['dataset']} | {r['suite']} | {r['profile']} | {r['status']} | {r.get('objective')} | {r['runtime_seconds']:.4f} | {r['verified']} | {ref.get('status')} | {ref.get('objective')} | {ref.get('runtime_seconds',0):.4f} | {r['match']} |")
    lines += ["", "## Coverage and limits", "", "- Netlib: AFIRO. Official MIPLIB: flugpl, gt2, b-ball, pk1, gen-ip016 (original MPS passed independently to each solver).",
              "- Synthetic robustness: degeneracy, ill-conditioning, weak relaxation. Synthetic scale: transport LPs to 40,000 variables, planning LPs to 1,500 rows x 2,200 variables, staircase LPs to 10,200 rows x 20,000 variables and portfolio QPs to 5,000 assets.",
              "- Repository industrial examples: refinery, blending, power dispatch, logistics. These are not claimed as published or proprietary industrial data.",
              "- QP examples compared with HiGHS using the quadratic Hessian, not an LP relaxation.",
              "- Mittelmann and QPLIB instances are not bundled or tested. Million-variable scale is shown only on synthetic structured LP/QP (scale-ladder.md).",
              "- GPU: measured separately in gpu-dense-ipm.md. CUDA interior point is 1.6x faster only on the 1,500-row dense planning LP; the sparse CPU path wins on the other models tested.",
              "- Only OPTIMAL + independent verification + reference OPTIMAL + objective tolerance agreement counts as a match. Failed verification and timeouts remain visible.",
              "- HiGHS is used only in this benchmark harness, never to solve production jobs."]
    lines += [f"- Not run: {s['dataset']} ({s['reason']})." for s in report.get("skipped", [])]
    target.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine")
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference:
        print(json.dumps(highs_reference(Path(args.reference), args.timeout), allow_nan=False))
        return
    import httpx
    if not args.engine:
        parser.error("--engine is required")
    engine = str(Path(args.engine).resolve())
    caps = json.loads(subprocess.check_output([engine, "capabilities"], text=True))
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "No NVIDIA GPU detected"
    report = {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "platform": platform.platform(), "cpu_threads": os.cpu_count(), "gpu": gpu, "engine": caps,
              "time_limit_seconds": args.timeout, "note": "Measured through a local HTTP coordinator and outbound worker using the same protocol as Render. All runs in this report use CPU. No live Render deployment or GPU speedup is claimed.", "rows": [], "skipped": []}
    with tempfile.TemporaryDirectory() as folder:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        # A throwaway SQLite store and the coordinator's local test sign-in; never a hosted database.
        env = {k: v for k, v in os.environ.items() if k not in ("MONGODB_URI", "SOVEREIGN_MONGODB_URI", "VERCEL")}
        env.update({"SOVEREIGN_DB": str(Path(folder) / "workspace.db"), "SOVEREIGN_TEST_AUTH": "1"})
        url = f"http://127.0.0.1:{port}"
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.cloud:app", "--port", str(port)], cwd=ROOT, env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        worker = None
        try:
            with httpx.Client(base_url=url, headers={"Authorization": "Bearer test-firebase-token"}, timeout=15) as client:
                for _ in range(100):
                    try:
                        if client.get("/api/health").status_code == 200: break
                    except httpx.HTTPError: pass
                    time.sleep(.1)
                else: raise RuntimeError("Coordinator startup failed")
                paired = client.post("/api/workers", json={"name": "SIH benchmark worker"})
                paired.raise_for_status()
                worker = subprocess.Popen([sys.executable, "-m", "worker.runner", "--server", url, "--engine", engine],
                    cwd=ROOT, env={**os.environ, "SOVEREIGN_WORKER_TOKEN": paired.json()["token"]},
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for dataset_id, suite, relative, _ in CASES:
                    entry = dataset(dataset_id)
                    path = ROOT / relative
                    if len(entry["modelJson"]) > MAX_MODEL_CHARACTERS:
                        reason = f"{len(entry['modelJson']) / 1e6:.1f} MB exceeds the hosted job limit ({MAX_MODEL_CHARACTERS / 1e6:.1f} MB)"
                        report["skipped"].append({"dataset": dataset_id, "suite": suite, "reason": reason})
                        print(f"Skipped: {dataset_id} ({reason})", flush=True)
                        continue
                    print(f"Reference: {dataset_id} ({suite})", flush=True)
                    try:
                        proc = subprocess.run([sys.executable, __file__, "--reference", str(path), "--timeout", str(args.timeout)],
                                              capture_output=True, text=True, timeout=args.timeout + 20)
                        reference = json.loads(proc.stdout) if proc.returncode == 0 else {"status": "ERROR", "message": proc.stderr[-1000:]}
                    except subprocess.TimeoutExpired:
                        reference = {"status": "TIME_LIMIT", "runtime_seconds": args.timeout + 20}
                    kind = entry["shape"]["problem_type"].upper()
                    profiles = [("Dual simplex", {"algorithm": "simplex"}), ("Interior point", {"algorithm": "ipm"})]
                    if kind == "MILP":
                        profiles = [("Branch & cut / strong", {"algorithm": "simplex", "branchRule": "strong"}),
                                    ("Branch & bound / fractional", {"algorithm": "simplex", "milpMethod": "branch_and_bound", "branchRule": "most_fractional"}),
                                    ("Branch & cut / pseudocost", {"algorithm": "ipm", "branchRule": "pseudocost"})]
                    elif kind == "QP":
                        profiles = [("QP interior point", {"qpAlgorithm": "ipm"}), ("Frank-Wolfe", {"qpAlgorithm": "frank_wolfe"})]
                    for label, config in profiles:
                        start = time.perf_counter()
                        submitted = client.post("/api/jobs", json={"name": f"{dataset_id} · {label}", "modelJson": entry["modelJson"],
                            "modelFormat": entry["modelFormat"], "device": "cpu", "maxNodes": 100000,
                            "timeLimitSeconds": args.timeout, **config})
                        submitted.raise_for_status()
                        job_id = submitted.json()["jobId"]
                        deadline = time.monotonic() + args.timeout + 25
                        while True:
                            job = client.get("/api/jobs/" + job_id).json()
                            if job["state"] in ("COMPLETED", "FAILED", "CANCELLED"): break
                            if worker.poll() is not None: raise RuntimeError("Worker exited unexpectedly")
                            if time.monotonic() > deadline: raise RuntimeError("Worker/coordinator did not complete the job within the watchdog")
                            time.sleep(.15)
                        result = job.get("result") or {}
                        status = result.get("status", "TIME_LIMIT" if "Time limit" in (job.get("message") or "") else job["state"])
                        verified = bool(result.get("verification", {}).get("is_valid"))
                        obj, ref_obj = result.get("objective_value"), reference.get("objective")
                        match = status == "OPTIMAL" and verified and reference.get("status") == "OPTIMAL" and obj is not None and ref_obj is not None and math.isclose(obj, ref_obj, rel_tol=1e-6, abs_tol=1e-7)
                        row = {"dataset": dataset_id, "suite": suite, "source": entry["source"], "sha256": entry["sha256"],
                               "shape": entry["shape"], "profile": label, "configuration": config, "job_id": job_id, "status": status,
                               "objective": obj, "runtime_seconds": result.get("runtime_seconds", args.timeout),
                               "turnaround_seconds": time.perf_counter() - start, "verified": verified, "gpu_used": result.get("gpu_used", False),
                               "gpu_operations": result.get("gpu_operations", 0), "match": match, "reference": reference,
                               "message": job.get("message") or result.get("message"), "verification": result.get("verification"),
                               "routing": job.get("routing")}
                        report["rows"].append(row)
                        write_report(report)
                        print(f"  {label}: {status}, objective={obj}, verified={verified}, match={match}", flush=True)
                print(f"Wrote {len(report['rows'])} comparisons to benchmarks/reports/sih-online.json and .md", flush=True)
        finally:
            if worker:
                worker.terminate(); worker.wait(timeout=10)
            server.terminate(); server.wait(timeout=10)


if __name__ == "__main__":
    main()
