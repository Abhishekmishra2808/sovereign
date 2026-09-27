"""Send harder models through the running PC worker and preserve full API replies."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("transport_100x100", "benchmarks/datasets/scale/transport_100x100.json", "json", {"algorithm": "ipm"}),
    ("flugpl", "benchmarks/datasets/miplib/official/flugpl.mps", "mps", {"algorithm": "ipm", "branchRule": "pseudocost"}),
    ("gt2", "benchmarks/datasets/miplib/official/gt2.mps", "mps", {"algorithm": "ipm", "branchRule": "pseudocost"}),
    ("industrial_logistics_milp", "examples/models/industrial_logistics_milp.json", "json", {"algorithm": "ipm", "branchRule": "pseudocost"}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / "data" / "preview.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "reports" / "hard-pc-roundtrip.json")
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    old = json.loads((ROOT / "benchmarks" / "reports" / "sih-online.json").read_text(encoding="utf-8"))
    references = {row["dataset"]: row["reference"] for row in old["rows"]}
    report = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "coordinator": state["url"], "time_limit_seconds": args.limit,
              "note": "Live HTTP coordinator -> PC workers -> C++ solver -> persisted result. CUDA executes only when a CUDA-capable worker is online.",
              "runs": []}
    headers = {"Authorization": "Bearer " + state["admin_token"]}
    with httpx.Client(base_url=state["url"], headers=headers, timeout=15) as client:
        workspace = client.get("/api/workspace")
        workspace.raise_for_status()
        online = [w for w in workspace.json()["workers"] if w["online"]]
        if not online:
            raise RuntimeError("No connected PC worker is online")
        report["workers"] = online
        for name, relative, model_format, config in CASES:
            path = ROOT / relative
            model = path.read_text(encoding="utf-8")
            request = {"name": "Hard PC test: " + name, "modelJson": model,
                       "modelFormat": model_format, "device": "cpu", "timeLimitSeconds": args.limit,
                       "maxNodes": 2000, **config}
            start = time.monotonic()
            response = client.post("/api/jobs", json=request)
            response.raise_for_status()
            job_id = response.json()["jobId"]
            print(f"Submitted {name}: {job_id}", flush=True)
            deadline = start + args.limit + 35
            while True:
                response = client.get("/api/jobs/" + job_id)
                response.raise_for_status()
                job = response.json()
                if job["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Job {job_id} exceeded coordinator watchdog")
                time.sleep(.25)
            result = job.get("result") or {}
            report["runs"].append({"dataset": name, "source_file": relative,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "request_options": {k: v for k, v in request.items() if k != "modelJson"},
                                   "reference": references.get(name),
                                   "turnaround_seconds": time.monotonic() - start,
                                   "response": job})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(f"  {job['state']} / {result.get('status')} / objective={result.get('objective_value')} / verified={result.get('verification', {}).get('is_valid')}", flush=True)

        name, relative, model_format, config = CASES[0]
        path = ROOT / relative
        request = {"name": "CUDA availability check: " + name,
                   "modelJson": path.read_text(encoding="utf-8"), "modelFormat": model_format,
                   "device": "cuda", "timeLimitSeconds": args.limit, **config}
        response = client.post("/api/jobs", json=request)
        response.raise_for_status()
        job_id = response.json()["jobId"]
        cuda_online = any(w["capabilities"].get("cuda_available") for w in online)
        if cuda_online:
            deadline = time.monotonic() + args.limit + 35
            while True:
                response = client.get("/api/jobs/" + job_id)
                response.raise_for_status()
                queue_response = response.json()
                if queue_response["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("CUDA job exceeded coordinator watchdog")
                time.sleep(.25)
            final_response = queue_response
        else:
            time.sleep(5)
            queued = client.get("/api/jobs/" + job_id)
            queued.raise_for_status()
            queue_response = queued.json()
            assert queue_response["state"] == "QUEUED", queue_response
            cancel = client.post("/api/jobs/" + job_id + "/cancel")
            cancel.raise_for_status()
            final = client.get("/api/jobs/" + job_id)
            final.raise_for_status()
            final_response = final.json()
        report["cuda_attempt"] = {"request_options": {k: v for k, v in request.items() if k != "modelJson"},
                                  "first_response": queue_response, "final_response": final_response,
                                  "reason": "CUDA worker completed job." if cuda_online else "No CUDA worker was online; CUDA job stayed queued."}
        env = os.environ.copy()
        env["SOVEREIGN_DEVICE"] = "cuda"
        engine = ROOT / "build-cloud" / "solver" / "Release" / "sovereign.exe"
        direct = subprocess.run([str(engine), "solve", str(path), "--verify"],
                                env=env, capture_output=True, text=True, timeout=10)
        report["cuda_direct_attempt"] = {"return_code": direct.returncode,
                                         "stdout": direct.stdout, "stderr": direct.stderr,
                                         "gpu_invoked": False}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        lines = ["# Hard PC worker round trip", "", f"Generated: {report['generated_at']}", "",
                 "The coordinator and PC worker exchanged each job over HTTP. Full stored API responses are in `hard-pc-roundtrip.json`.", "",
                 "| Model | Job ID | State | Solver status | Objective | Verified | Turnaround |",
                 "|---|---|---|---|---:|---|---:|"]
        for run in report["runs"]:
            response = run["response"]
            result = response.get("result") or {}
            lines.append(f"| {run['dataset']} | `{response['id']}` | {response['state']} | {result.get('status', 'none')} | {result.get('objective_value', 'none')} | {result.get('verification', {}).get('is_valid', False)} | {run['turnaround_seconds']:.2f}s |")
        lines += ["", f"CUDA request `{job_id}` ended in {final_response['state']}; GPU kernels reported: {(final_response.get('result') or {}).get('gpu_operations', 0)}.",
                  f"Direct CUDA CLI attempt returned code {direct.returncode}: `{direct.stderr.strip()}`", "",
                  "The direct CLI check uses the CPU build. See the CUDA job result for actual kernel use when a CUDA worker is connected.", ""]
        args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
        print(f"CUDA request: {final_response['state']}. Full responses: {args.output}", flush=True)


if __name__ == "__main__":
    main()
