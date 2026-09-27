"""Process-level coordinator to installed-worker-module to CUDA result smoke test."""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "build-gpu-real" / "solver" / "Release" / "sovereign.exe"
BASE = "http://127.0.0.1:8011"


def call(path, token=None, body=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def main():
    if not ENGINE.is_file():
        raise SystemExit("Build the CUDA executable before running this smoke test.")
    admin = secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory() as folder:
        env = os.environ.copy()
        env.update({"SOVEREIGN_ADMIN_TOKEN": admin,
                    "SOVEREIGN_DB": str(Path(folder) / "jobs.db")})
        env.pop("DATABASE_URL", None)
        env.pop("SOVEREIGN_DATABASE_URL", None)
        env.pop("VERCEL", None)
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.cloud:app",
            "--host", "127.0.0.1", "--port", "8011", "--log-level", "error"],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for _ in range(50):
                if server.poll() is not None:
                    raise RuntimeError("Coordinator exited during startup.")
                try:
                    call("/api/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("Coordinator did not become ready.")
            worker = call("/api/workers", admin, {"name": "Packaged connector smoke"})
            model = (ROOT / "examples" / "models" / "sample_lp.json").read_text(encoding="utf-8")
            job_id = call("/api/jobs", admin, {"name": "CUDA connector smoke",
                "modelJson": model, "device": "cuda", "algorithm": "ipm"})["jobId"]
            env["SOVEREIGN_WORKER_TOKEN"] = worker["token"]
            run = subprocess.run([sys.executable, "-m", "worker.runner", "--server", BASE,
                "--engine", str(ENGINE), "--once"], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=30)
            if run.returncode:
                raise RuntimeError(run.stdout + run.stderr)
            result = call("/api/jobs/" + job_id, admin)
            assert result["state"] == "COMPLETED", result.get("message")
            assert result["result"]["verification"]["is_valid"]
            assert result["result"]["gpu_operations"] > 0
            print(f"PASS connector process: {result['state']} "
                  f"gpu_operations={result['result']['gpu_operations']} "
                  f"objective={result['result']['objective_value']}")
        finally:
            server.terminate()
            try:
                server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.communicate()


if __name__ == "__main__":
    main()
