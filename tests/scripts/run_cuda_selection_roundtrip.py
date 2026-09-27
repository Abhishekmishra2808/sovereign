"""Real coordinator/worker/CUDA contract test; writes full returned job payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api.cloud import app
from worker.runner import capabilities, execute

DEFAULT_ENGINE = ROOT / "build-gpu-real/solver/Release/sovereign.exe"
OUTPUT = ROOT / "benchmarks/reports/cuda-selection-roundtrip.json"
SECRET = "test-workspace-secret-123456789"


def models():
    ordinary = (ROOT / "examples/models/sample_lp.json").read_text(encoding="utf-8")
    fixed = json.dumps({"problem_type": "LP", "sense": "minimize",
        "variables": [{"name": "x", "type": "continuous", "lower_bound": 1, "upper_bound": 1}],
        "objective": {"linear": {"x": 1}},
        "constraints": [{"name": "cap", "linear": {"x": 1}, "sense": "<=", "rhs": 2}]})
    unconstrained = json.dumps({"problem_type": "LP", "sense": "minimize",
        "variables": [{"name": "x", "type": "continuous", "lower_bound": 0,
                       "upper_bound": 1e30}],
        "objective": {"linear": {"x": 1}}, "constraints": []})
    return [("automatic_lp", ordinary, "auto"),
            ("ordinary_lp", ordinary, "cuda"),
            ("presolve_fixed_lp", fixed, "cuda"),
            ("unconstrained_lp", unconstrained, "cuda")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    engine = str(args.engine.resolve())
    caps = capabilities(engine)
    if not caps["cuda_available"]:
        parser.error("The selected executable does not report CUDA available.")
    entries = []
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "SOVEREIGN_DB": str(Path(tmp) / "db.sqlite"),
            "SOVEREIGN_ADMIN_TOKEN": SECRET}):
        with TestClient(app) as client:
            admin = {"Authorization": "Bearer " + SECRET}
            pair = client.post("/api/workers", headers=admin,
                               json={"name": "CUDA acceptance worker"})
            pair.raise_for_status()
            worker = {"Authorization": "Bearer " + pair.json()["token"]}
            for name, model, device in models():
                created = client.post("/api/jobs", headers=admin,
                    json={"name": name, "modelJson": model, "device": device,
                          "presolve": True})
                created.raise_for_status()
                claimed = client.post("/api/worker/claim", headers=worker,
                                      json=caps).json()["job"]
                assert claimed and claimed["id"] == created.json()["jobId"]
                completion = execute(None, engine, claimed)
                returned = client.post("/api/worker/complete", headers=worker,
                                       json=completion)
                returned.raise_for_status()
                final = client.get("/api/jobs/" + claimed["id"], headers=admin).json()
                entries.append({"case": name, "model_sha256": hashlib.sha256(model.encode()).hexdigest(),
                                "requested_device": device, "assigned_device": claimed["request"]["executionDevice"],
                                "job": final})
                result = final["result"]
                print(f"{name}: {final['state']} {result['status']} "
                      f"gpu_operations={result['gpu_operations']} "
                      f"verified={result['verification']['is_valid']}")
    expected = {
        "automatic_lp": ("COMPLETED", 0),
        "ordinary_lp": ("COMPLETED", 1),
        "presolve_fixed_lp": ("COMPLETED", 1),
        "unconstrained_lp": ("FAILED", 0),
    }
    for entry in entries:
        job = entry["job"]
        state, min_ops = expected[entry["case"]]
        assert job["state"] == state, entry["case"]
        assert entry["assigned_device"] == ("cpu" if entry["case"] == "automatic_lp" else "cuda"), entry["case"]
        assert job["result"]["verification"]["is_valid"], entry["case"]
        assert job["result"]["gpu_operations"] >= min_ops, entry["case"]
        if entry["case"] == "unconstrained_lp":
            assert "0 GPU operations" in job["message"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"engine": engine, "capabilities": caps,
                                       "entries": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"Captured full job responses in {args.output}")


if __name__ == "__main__":
    main()
