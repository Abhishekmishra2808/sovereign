"""Coordinator contract/security tests. No solver or external service required."""
import json
import os
import tempfile
import unittest
import sys
from types import ModuleType
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi import HTTPException, Request
from api.cloud import app, database, admin as authorize

CAPS = {"hostname": "test-host", "platform": "Linux", "cpu_threads": 4, "cuda_available": False}
MODEL = json.dumps({"variables": [{"name": "x"}], "objective": {"linear": {"x": 1}}})


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"SOVEREIGN_DB": str(Path(self.temp.name) / "jobs.db"),
                                          "SOVEREIGN_ADMIN_TOKEN": "test-workspace-secret-123456789"})
        self.env.start()
        self.client = TestClient(app)
        self.admin = {"Authorization": "Bearer test-workspace-secret-123456789"}

    def tearDown(self):
        self.client.close()
        self.env.stop()
        self.temp.cleanup()

    def pair(self, name="Test machine"):
        response = self.client.post("/api/workers", headers=self.admin, json={"name": name})
        self.assertEqual(response.status_code, 201, response.text)
        data = response.json()
        return data["id"], {"Authorization": "Bearer " + data["token"]}

    def submit(self, **kwargs):
        response = self.client.post("/api/jobs", headers=self.admin, json={"modelJson": MODEL, **kwargs})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["jobId"]

    def claim(self, headers, **kwargs):
        return self.client.post("/api/worker/claim", headers=headers, json={**CAPS, **kwargs}).json()["job"]

    def test_private_workspace_and_separate_worker_role(self):
        self.assertEqual(self.client.get("/api/workspace").status_code, 401)
        _, worker = self.pair()
        self.assertEqual(self.client.get("/api/workspace", headers=worker).status_code, 401)
        self.assertEqual(self.client.post("/api/jobs", headers=worker, json={"modelJson": MODEL}).status_code, 401)
        self.assertEqual(self.client.post("/api/worker/claim", headers=self.admin, json=CAPS).status_code, 401)
        workspace = self.client.get("/api/workspace", headers=self.admin).text
        self.assertNotIn("token", workspace)
        self.assertNotIn("modelJson", workspace)

    def test_session_and_cross_origin(self):
        response = self.client.post("/api/session", json={"token": "test-workspace-secret-123456789"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertEqual(self.client.get("/api/workspace").status_code, 200)
        self.assertEqual(self.client.post("/api/workers", json={"name": "bad"}, headers={"Origin": "https://elsewhere.invalid"}).status_code, 403)
        self.client.delete("/api/session")
        self.assertEqual(self.client.get("/api/workspace").status_code, 401)

    def test_configured_frontend_origin_can_proxy_session(self):
        with patch.dict(os.environ, {"SOVEREIGN_PUBLIC_ORIGIN": "https://ui.vercel.app"}):
            response = self.client.get("/api/workspace", headers={
                **self.admin, "Origin": "https://ui.vercel.app"})
            other = self.client.get("/api/workspace", headers={
                **self.admin, "Origin": "https://other.vercel.app"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(other.status_code, 403)

    def test_gpu_and_target_routing(self):
        _, cpu = self.pair("CPU")
        gpu_id, gpu = self.pair("GPU")
        job_id = self.submit(device="cuda", workerId=gpu_id)
        self.assertIsNone(self.claim(cpu))
        self.assertIsNone(self.claim(gpu))
        self.assertEqual(self.claim(gpu, cuda_available=True)["id"], job_id)

    def test_cuda_rejects_cpu_only_methods(self):
        lp = self.client.post("/api/jobs", headers=self.admin,
            json={"modelJson": MODEL, "device": "cuda", "algorithm": "simplex"})
        self.assertEqual(lp.status_code, 422)
        self.assertIn("CPU", lp.json()["detail"])
        qp_model = json.dumps({"problem_type": "QP", "variables": [{"name": "x"}],
            "objective": {"quadratic": {"x": {"x": 2}}}})
        qp = self.client.post("/api/jobs", headers=self.admin,
            json={"modelJson": qp_model, "device": "cuda", "qpAlgorithm": "frank_wolfe"})
        self.assertEqual(qp.status_code, 422)
        self.assertIn("CPU", qp.json()["detail"])

    def test_auto_routing_defaults_to_cpu_after_measured_gpu_trials(self):
        self.pair("CPU")
        _, gpu = self.pair("GPU")
        large_model = json.dumps({"variables": [{"name": "x"}],
            "constraints": [{"name": "row", "linear": {"x": 1}, "sense": "<=", "rhs": 2}],
            "objective": {"linear": {"x": 1}}})
        with patch.dict(os.environ, {"SOVEREIGN_GPU_MIN_NONZEROS": "1",
                                      "SOVEREIGN_GPU_AUTO_ENABLED": "0"}):
            self.assertIsNone(self.claim(gpu, cuda_available=True))
            job_id = self.submit(modelJson=large_model)
            # A GPU-capable machine may still claim a CPU-routed job; its
            # executionDevice must remain CPU.
            claimed = self.claim(gpu, cuda_available=True)
            self.assertEqual(claimed["id"], job_id)
            self.assertEqual(claimed["request"]["executionDevice"], "cpu")
            self.assertEqual(claimed["request"]["routing"]["policy"], "cpu-baseline-v2")

    def test_experimental_auto_routing_prefers_online_gpu_and_falls_back(self):
        _, cpu = self.pair("CPU")
        _, gpu = self.pair("GPU")
        large_model = json.dumps({"variables": [{"name": "x"}],
            "constraints": [{"name": "row", "linear": {"x": 1}, "sense": "<=", "rhs": 2}],
            "objective": {"linear": {"x": 1}}})
        with patch.dict(os.environ, {"SOVEREIGN_GPU_MIN_NONZEROS": "1",
                                      "SOVEREIGN_GPU_AUTO_ENABLED": "1"}):
            self.assertIsNone(self.claim(gpu, cuda_available=True))
            job_id = self.submit(modelJson=large_model)
            self.assertIsNone(self.claim(cpu))
            claimed = self.claim(gpu, cuda_available=True)
            self.assertEqual(claimed["id"], job_id)
            self.assertEqual(claimed["request"]["executionDevice"], "cuda")
            second = self.submit(modelJson=large_model, workerId=None)
            with database() as db:
                db.execute("UPDATE workers SET seen=0 WHERE name='GPU'")
            fallback = self.claim(cpu)
            self.assertEqual(fallback["id"], second)
            self.assertEqual(fallback["request"]["executionDevice"], "cpu")

    def test_completion_idempotent_and_persistent(self):
        _, worker = self.pair()
        job_id = self.submit()
        claimed = self.claim(worker)
        lease = {"jobId": job_id, "lease": claimed["lease"]}
        self.assertEqual(self.client.post("/api/worker/heartbeat", headers=worker, json=lease).status_code, 200)
        payload = {**lease, "result": {"status": "OPTIMAL", "objective_value": 9,
                                       "primal": {"x": 9}, "verification": {"is_valid": True}}}
        for _ in range(2):
            self.assertEqual(self.client.post("/api/worker/complete", headers=worker, json=payload).status_code, 200)
        with TestClient(app) as fresh:
            result = fresh.get(f"/api/jobs/{job_id}", headers=self.admin).json()
        self.assertEqual(result["state"], "COMPLETED")
        self.assertEqual(result["result"]["objective_value"], 9)
        self.assertNotIn("lease", result)

    def test_unverified_optimum_is_failed_and_visible(self):
        _, worker = self.pair()
        job_id = self.submit()
        claimed = self.claim(worker)
        lease = {"jobId": job_id, "lease": claimed["lease"]}
        response = self.client.post("/api/worker/complete", headers=worker,
                                    json={**lease, "result": {"status": "OPTIMAL", "objective_value": 9,
                                        "verification": {"is_valid": False, "issues": ["constraint violated"]}}})
        self.assertEqual(response.status_code, 200)
        job = self.client.get(f"/api/jobs/{job_id}", headers=self.admin).json()
        self.assertEqual(job["state"], "FAILED")
        self.assertIn("verification", job["message"])
        self.assertFalse(job["result"]["verification"]["is_valid"])

    def test_cuda_contract_error_keeps_verified_answer_visible(self):
        _, worker = self.pair()
        job_id = self.submit(device="cuda")
        claimed = self.claim(worker, cuda_available=True)
        reason = "CUDA was selected, but the worker reported 0 GPU operations."
        response = self.client.post("/api/worker/complete", headers=worker, json={
            "jobId": job_id, "lease": claimed["lease"], "error": reason,
            "result": {"status": "OPTIMAL", "objective_value": 1,
                       "gpu_operations": 0, "gpu_used": False,
                       "verification": {"is_valid": True}}})
        self.assertEqual(response.status_code, 200)
        job = self.client.get(f"/api/jobs/{job_id}", headers=self.admin).json()
        self.assertEqual(job["state"], "FAILED")
        self.assertEqual(job["message"], reason)
        self.assertEqual(job["result"]["objective_value"], 1)

    def test_expired_lease_cannot_overwrite_new_result(self):
        _, worker = self.pair()
        job_id = self.submit()
        old = self.claim(worker)
        with database() as db:
            db.execute("UPDATE jobs SET expires=0 WHERE id=?", (job_id,))
        fresh = self.claim(worker)
        self.assertNotEqual(old["lease"], fresh["lease"])
        response = self.client.post("/api/worker/complete", headers=worker,
                                    json={"jobId": job_id, "lease": old["lease"], "result": {"status": "OPTIMAL"}})
        self.assertEqual(response.status_code, 409)

    def test_cancel_prevents_heartbeat_and_result(self):
        _, worker = self.pair()
        job_id = self.submit()
        claimed = self.claim(worker)
        self.client.post(f"/api/jobs/{job_id}/cancel", headers=self.admin)
        lease = {"jobId": job_id, "lease": claimed["lease"]}
        self.assertEqual(self.client.post("/api/worker/heartbeat", headers=worker, json=lease).status_code, 409)
        self.assertEqual(self.client.post("/api/worker/complete", headers=worker, json={**lease, "result": {"status": "OPTIMAL"}}).status_code, 409)
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}", headers=self.admin).json()["state"], "CANCELLED")

    def test_revocation_and_invalid_model(self):
        worker_id, worker = self.pair()
        self.client.delete(f"/api/workers/{worker_id}", headers=self.admin)
        self.assertEqual(self.client.post("/api/worker/claim", headers=worker, json=CAPS).status_code, 401)
        for text in ("{oops", "[]", '{"variables": []}', '{"variables": [NaN]}'):
            self.assertEqual(self.client.post("/api/jobs", headers=self.admin, json={"modelJson": text}).status_code, 422)

    def test_concurrent_claims_assign_once(self):
        workers = [self.pair(str(i))[1] for i in range(4)]
        self.submit()
        with ThreadPoolExecutor(max_workers=4) as pool:
            claims = list(pool.map(self.claim, workers))
        self.assertEqual(sum(c is not None for c in claims), 1)

    def test_three_disconnects_stop_retrying(self):
        _, worker = self.pair()
        job_id = self.submit()
        for _ in range(3):
            self.assertIsNotNone(self.claim(worker))
            with database() as db:
                db.execute("UPDATE jobs SET expires=0 WHERE id=?", (job_id,))
        self.assertIsNone(self.claim(worker))
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}", headers=self.admin).json()["state"], "FAILED")

    def test_vercel_requires_persistent_database(self):
        with patch("api.cloud.VERCEL_FUNCTION", True), patch.dict(os.environ,
                {"DATABASE_URL": "", "SOVEREIGN_DATABASE_URL": ""}):
            response = self.client.get("/api/workspace", headers=self.admin)
            health = self.client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        self.assertIn("DATABASE_URL", response.json()["detail"])
        self.assertEqual(health.status_code, 503)

    def test_postgres_adapter_uses_transaction_lock_and_parameters(self):
        class FakeConnection:
            def __init__(self):
                self.commands = []
                self.committed = self.closed = False

            def execute(self, sql, params=()):
                self.commands.append((sql, params))

            def commit(self):
                self.committed = True

            def rollback(self):
                raise AssertionError("Successful transaction rolled back")

            def close(self):
                self.closed = True

        connection = FakeConnection()
        psycopg = ModuleType("psycopg")
        rows = ModuleType("psycopg.rows")
        rows.dict_row = object()
        psycopg.connect = lambda *args, **kwargs: connection
        with patch.dict(sys.modules, {"psycopg": psycopg, "psycopg.rows": rows}), \
                patch.dict(os.environ, {"SOVEREIGN_DATABASE_URL": "postgresql://test"}):
            with database() as db:
                db.execute("SELECT id FROM workers WHERE token_hash=?", ("abc",))
        self.assertIn("pg_advisory_xact_lock", connection.commands[0][0])
        self.assertEqual(connection.commands[-1],
                         ("SELECT id FROM workers WHERE token_hash=%s", ("abc",)))
        self.assertTrue(connection.committed)
        self.assertTrue(connection.closed)

    def test_vercel_forwarded_origin_and_secure_cookie(self):
        headers = {**self.admin, "Host": "example.vercel.app",
                   "Origin": "https://example.vercel.app", "X-Forwarded-Proto": "https"}
        scope = {"type": "http", "scheme": "http", "server": ("testserver", 80),
                 "path": "/api/workspace", "root_path": "",
                 "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()]}
        with patch("api.cloud.VERCEL_FUNCTION", True):
            authorize(Request(scope))
            login = self.client.post("/api/session", headers=headers,
                json={"token": "test-workspace-secret-123456789"})
            scope["headers"] = [(key, b"https://elsewhere.invalid" if key == b"origin" else value)
                                for key, value in scope["headers"]]
            with self.assertRaises(HTTPException):
                authorize(Request(scope))
        self.assertIn("Secure", login.headers["set-cookie"])


if __name__ == "__main__":
    unittest.main()
