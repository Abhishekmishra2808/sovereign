"""Real HTTP -> worker process -> C++ engine -> stored verified result smoke test."""
import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    engine = str(Path(args.engine).resolve())
    with tempfile.TemporaryDirectory() as folder:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {**os.environ, "SOVEREIGN_DB": str(Path(folder) / "jobs.db"), "SOVEREIGN_ADMIN_TOKEN": secrets.token_urlsafe(32)}
        url = f"http://127.0.0.1:{port}"
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.cloud:app", "--port", str(port)],
                                  cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            with httpx.Client(base_url=url, timeout=5, headers={"Authorization": "Bearer " + env["SOVEREIGN_ADMIN_TOKEN"]}) as client:
                for _ in range(100):
                    try:
                        if client.get("/api/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.1)
                else:
                    raise RuntimeError("Coordinator did not start")
                pair = client.post("/api/workers", json={"name": "Smoke worker"})
                pair.raise_for_status()
                worker_env = {**os.environ, "SOVEREIGN_WORKER_TOKEN": pair.json()["token"]}
                for filename in ("sample_lp.json", "sample_milp.json", "sample_qp.json", "sample_knapsack.mps"):
                    model = (ROOT / "examples" / "models" / filename).read_text()
                    submitted = client.post("/api/jobs", json={"name": filename, "modelJson": model,
                        "modelFormat": "mps" if filename.endswith(".mps") else "json", "device": "cpu"})
                    submitted.raise_for_status()
                    worker = subprocess.run([sys.executable, "-m", "worker.runner", "--server", url, "--engine", engine, "--once"],
                                            cwd=ROOT, env=worker_env, capture_output=True, text=True, timeout=60)
                    assert worker.returncode == 0, worker.stderr
                    job = client.get("/api/jobs/" + submitted.json()["jobId"]).json()
                    assert job["state"] == "COMPLETED", job
                    result = job["result"]
                    assert result["status"] == "OPTIMAL", result
                    assert result["verification"]["is_valid"], result
                    assert result["gpu_used"] is False, result
                    print(f"PASS {filename}: verified optimum {result['objective_value']}, CPU, HTTP round-trip")
                # The exact UI example has a known optimum of 9.
                model = {"problem_type": "MILP", "sense": "maximize", "variables": [
                    {"name": n, "type": "binary", "lower_bound": 0, "upper_bound": 1} for n in "abc"],
                    "objective": {"linear": {"a": 5, "b": 4, "c": 3}},
                    "constraints": [{"name": "cap", "linear": {"a": 1, "b": 1, "c": 1}, "sense": "<=", "rhs": 2}]}
                job_id = client.post("/api/jobs", json={"modelJson": json.dumps(model), "device": "cpu"}).json()["jobId"]
                subprocess.run([sys.executable, "-m", "worker.runner", "--server", url, "--engine", engine, "--once"],
                               cwd=ROOT, env=worker_env, capture_output=True, timeout=60, check=True)
                result = client.get("/api/jobs/" + job_id).json()["result"]
                assert result["objective_value"] == 9 and result["verification"]["is_valid"], result
                print("PASS dashboard example: optimum 9, independently verified")
                response = client.get("/")
                assert response.status_code == 200 and response.json()["service"] == "Sovereign coordinator"
                assert client.get("/worker.py").status_code == 200
                print("PASS backend status and worker download served")
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
