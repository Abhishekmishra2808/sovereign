"""Real coordinator/worker/CUDA contract test; writes full returned job payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api.cloud import app
from benchmarks.tools.generate_sparse_scale import staircase
from worker.runner import capabilities, execute

DEFAULT_ENGINE = ROOT / "build/solver/sovereign.exe"
OUTPUT = ROOT / "benchmarks/reports/cuda-selection-roundtrip.json"
AUTH = {"Authorization": "Bearer test-firebase-token"}


def planning_lp(rows, cols):
    """Production-planning LP whose interior-point system has `rows` rows."""
    rng = random.Random(rows * 7919 + cols)
    linear = [dict() for _ in range(rows)]
    for j in range(cols):
        for i in rng.sample(range(rows), 6):
            linear[i][f"x{j}"] = round(rng.uniform(0.5, 3.0), 3)
    return json.dumps({"problem_type": "LP", "sense": "maximize",
        "variables": [{"name": f"x{j}", "type": "continuous", "lower_bound": 0, "upper_bound": 1e30}
                      for j in range(cols)],
        "objective": {"linear": {f"x{j}": round(rng.uniform(1.0, 10.0), 3) for j in range(cols)}},
        "constraints": [{"name": f"r{i}", "linear": row, "sense": "<=", "rhs": round(rng.uniform(50, 150), 2)}
                        for i, row in enumerate(linear) if row]})


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
    milp = (ROOT / "examples/models/sample_milp.json").read_text(encoding="utf-8")
    # (case, model, requested device, expected state, assigned device, minimum GPU factorizations)
    # Both large automatic jobs go to the GPU machine; the engine then factors the
    # dense planning LP on the GPU and the block-banded staircase sparsely on the CPU.
    return [("automatic_small_lp", ordinary, "auto", "COMPLETED", "cpu", 0),
            ("automatic_large_lp", planning_lp(1500, 2200), "auto", "COMPLETED", "cuda", 1),
            ("automatic_sparse_lp", json.dumps(staircase(20, 100)), "auto", "COMPLETED", "cuda", 0),
            ("automatic_milp", milp, "auto", "COMPLETED", "cpu", 0),
            ("ordinary_lp", ordinary, "cuda", "COMPLETED", "cuda", 1),
            ("presolve_fixed_lp", fixed, "cuda", "COMPLETED", "cuda", 0),
            ("unconstrained_lp", unconstrained, "cuda", "FAILED", "cuda", 0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    engine = str(args.engine.resolve())
    caps = capabilities(engine)
    if not caps["cuda_available"]:
        parser.error("The selected executable does not report CUDA available: " + caps["cuda_reason"])
    entries = []
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "SOVEREIGN_DB": str(Path(tmp) / "db.sqlite"), "SOVEREIGN_TEST_AUTH": "1"}):
        with TestClient(app) as client:
            pair = client.post("/api/workers", headers=AUTH, json={"name": "CUDA acceptance worker"})
            pair.raise_for_status()
            worker = {"Authorization": "Bearer " + pair.json()["token"]}
            for name, model, device, state, assigned, min_factorizations in models():
                created = client.post("/api/jobs", headers=AUTH,
                    json={"name": name, "modelJson": model, "device": device, "presolve": True,
                          "timeLimitSeconds": 120})
                created.raise_for_status()
                claimed = client.post("/api/worker/claim", headers=worker, json=caps).json()["job"]
                assert claimed and claimed["id"] == created.json()["jobId"]
                completion = execute(None, engine, claimed)
                returned = client.post("/api/worker/complete", headers=worker, json=completion)
                returned.raise_for_status()
                final = client.get("/api/jobs/" + claimed["id"], headers=AUTH).json()
                result = final["result"]
                entries.append({"case": name, "model_sha256": hashlib.sha256(model.encode()).hexdigest(),
                                "requested_device": device, "assigned_device": claimed["request"]["executionDevice"],
                                "routing_reason": claimed["request"]["routing"]["reason"], "job": final})
                print(f"{name}: {final['state']} {result['status']} on {claimed['request']['executionDevice']} "
                      f"gpu_operations={result['gpu_operations']} "
                      f"gpu_factorizations={result.get('gpu_factorizations')} "
                      f"verified={result['verification']['is_valid']}")
                assert final["state"] == state, name
                assert claimed["request"]["executionDevice"] == assigned, name
                assert result["verification"]["is_valid"], name
                assert result.get("gpu_factorizations", 0) >= min_factorizations, name
                if device == "cuda" and state == "COMPLETED":
                    assert result["gpu_operations"] > 0, name
                if name == "automatic_sparse_lp":
                    assert "sparse LDL^T" in result["message"], result["message"]
                    assert result.get("gpu_factorizations", 0) == 0, name
                if name == "unconstrained_lp":
                    assert "0 GPU operations" in final["message"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"engine": engine, "capabilities": caps,
                                       "entries": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"Captured full job responses in {args.output}")


if __name__ == "__main__":
    main()
