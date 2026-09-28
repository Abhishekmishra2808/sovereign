"""Live benchmark run through a local coordinator and the Python worker (needs a built engine and highspy)."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
AUTH = {"Authorization": "Bearer test-firebase-token"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {**os.environ, "SOVEREIGN_DB": str(Path(folder) / "lab.db"), "SOVEREIGN_TEST_AUTH": "1"}
        env.pop("MONGODB_URI", None)
        env.pop("SOVEREIGN_MONGODB_URI", None)
        url = f"http://127.0.0.1:{port}"
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.cloud:app", "--port", str(port)],
                                  cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        worker = None
        try:
            with httpx.Client(base_url=url, headers=AUTH, timeout=30) as client:
                for _ in range(100):
                    try:
                        if client.get("/api/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Coordinator startup failed")
                token = client.post("/api/workers", json={"name": "Lab worker"}).json()["token"]
                worker = subprocess.Popen([sys.executable, "-m", "worker.runner", "--server", url, "--engine", args.engine],
                                          cwd=ROOT, env={**env, "SOVEREIGN_WORKER_TOKEN": token},
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                catalogue = client.get("/api/benchmarks/catalogue").json()
                preset = next(p for p in catalogue["presets"] if p["id"] == args.preset)
                run = client.post("/api/benchmarks/runs", json={
                    "datasets": preset["datasets"], "profiles": preset["profiles"], "device": "cpu",
                    "reference": True, "timeLimitSeconds": args.timeout}).json()
                print(f"Run {run['runId']}: {run['jobs']} jobs", flush=True)
                started = time.monotonic()
                while True:
                    state = client.get(f"/api/benchmarks/runs/{run['runId']}").json()
                    if state["state"] == "finished":
                        break
                    if worker.poll() is not None:
                        raise RuntimeError("Worker exited")
                    if time.monotonic() - started > run["jobs"] * (args.timeout + 25):
                        raise RuntimeError("Run did not finish")
                    time.sleep(1)
                for r in state["rows"]:
                    ref = r["reference"] or {}
                    print(f"{r['dataset']:<28} {r['profileLabel']:<18} {str(r['status']):<12} "
                          f"{r['runtimeSeconds'] if r['runtimeSeconds'] is not None else '-':<12} "
                          f"verified={r.get('verified')} highs={ref.get('status')} {ref.get('runtimeSeconds')} "
                          f"agrees={r['agrees']}")
                print(json.dumps(state["summary"], indent=2))
                print(json.dumps(state["machines"], indent=2))
                print(f"Wall clock: {time.monotonic() - started:.1f}s")
        finally:
            if worker:
                worker.terminate()
                worker.wait(timeout=10)
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
