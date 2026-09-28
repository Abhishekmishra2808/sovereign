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
from api.cloud import app, database, current_user

CAPS = {"hostname": "test-host", "platform": "Linux", "cpu_threads": 4, "cuda_available": False}
MODEL = json.dumps({"variables": [{"name": "x"}], "objective": {"linear": {"x": 1}}})
TEST_AUTH = {"Authorization": "Bearer test-firebase-token"}


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "SOVEREIGN_DB": str(Path(self.temp.name) / "jobs.db"),
            "SOVEREIGN_TEST_AUTH": "1",
        })
        self.env.start()
        self.client = TestClient(app)
        self.admin = TEST_AUTH

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

    def test_firebase_auth_and_cross_origin(self):
        self.assertEqual(self.client.get("/api/workspace").status_code, 401)
        self.assertEqual(self.client.get("/api/workspace", headers=self.admin).status_code, 200)
        self.assertEqual(self.client.post("/api/workers", json={"name": "bad"}, headers={
            **self.admin, "Origin": "https://elsewhere.invalid"}).status_code, 403)

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
            claimed = self.claim(gpu, cuda_available=True)
            self.assertEqual(claimed["id"], job_id)
            self.assertEqual(claimed["request"]["executionDevice"], "cpu")
            self.assertEqual(claimed["request"]["routing"]["policy"], "cpu-baseline-v2")

    def test_experimental_auto_routing_prefers_online_gpu_and_falls_back(self):
        cpu_id, cpu = self.pair("CPU")
        gpu_id, gpu = self.pair("GPU")
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
            with database() as store:
                store.touch_worker(gpu_id, 0)
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
        with database() as store:
            store.force_expire_job(job_id)
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

    def test_delete_job_requires_owner_and_not_running(self):
        _, worker = self.pair()
        job_id = self.submit()
        self.assertEqual(self.client.delete(f"/api/jobs/{job_id}").status_code, 401)
        self.claim(worker)
        running = self.client.delete(f"/api/jobs/{job_id}", headers=self.admin)
        self.assertEqual(running.status_code, 409)
        self.assertIn("Cancel", running.json()["detail"])
        self.client.post(f"/api/jobs/{job_id}/cancel", headers=self.admin)
        self.assertEqual(self.client.delete(f"/api/jobs/{job_id}", headers=self.admin).status_code, 200)
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}", headers=self.admin).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/jobs/{job_id}", headers=self.admin).status_code, 404)
        queued = self.submit()
        self.assertEqual(self.client.delete(f"/api/jobs/{queued}", headers=self.admin).status_code, 200)
        jobs = self.client.get("/api/workspace", headers=self.admin).json()["jobs"]
        self.assertEqual(jobs, [])

    def test_benchmark_run_reference_routing_and_rows(self):
        catalogue = self.client.get("/api/benchmarks/catalogue", headers=self.admin).json()
        ids = {d["id"] for d in catalogue["datasets"]}
        self.assertIn("afiro", ids)
        self.assertIn("flugpl-lp", ids)
        relaxed = next(d for d in catalogue["datasets"] if d["id"] == "b-ball-lp")
        self.assertEqual(relaxed["shape"]["problem_type"], "LP")
        self.assertEqual(relaxed["shape"]["integer_variables"], 0)

        _, connector = self.pair("npm connector")
        _, python = self.pair("python worker")
        created = self.client.post("/api/benchmarks/runs", headers=self.admin, json={
            "datasets": ["afiro", "sample_qp"], "profiles": ["lp_simplex", "lp_ipm", "qp_ipm"],
            "device": "cuda", "reference": True, "timeLimitSeconds": 20})
        self.assertEqual(created.status_code, 201, created.text)
        run_id = created.json()["runId"]
        self.assertEqual(created.json()["jobs"], 5)
        self.assertEqual(self.client.get("/api/workspace", headers=self.admin).json()["jobs"], [])

        # The HiGHS job for afiro is first in the queue, but a worker without HiGHS never receives it.
        first = self.claim(connector)
        self.assertNotEqual(first["request"].get("solver"), "highs")
        self.assertEqual(first["request"]["algorithm"], "simplex")
        self.assertEqual(first["request"]["device"], "cpu")
        reference = self.claim(python, reference_solvers=["highs"], highs_version="1.15.1")
        self.assertEqual(reference["request"]["solver"], "highs")

        bad = {"jobId": reference["id"], "lease": reference["lease"], "result": {"status": "OPTIMAL"}}
        self.assertEqual(self.client.post("/api/worker/complete", headers=python, json=bad).status_code, 422)
        self.assertEqual(self.client.post("/api/worker/complete", headers=python, json={
            **bad, "result": {"solver": "highs", "status": "OPTIMAL", "objective_value": -464.7531428571,
                              "runtime_seconds": 0.02, "version": "1.15.1"}}).status_code, 200)
        self.assertEqual(self.client.post("/api/worker/complete", headers=connector, json={
            "jobId": first["id"], "lease": first["lease"],
            "result": {"status": "OPTIMAL", "objective_value": -464.7531428571, "runtime_seconds": 0.001,
                       "verification": {"is_valid": True}, "primal": {"x": 1}}}).status_code, 200)

        run = self.client.get(f"/api/benchmarks/runs/{run_id}", headers=self.admin).json()
        self.assertEqual(run["state"], "running")
        rows = {r["profile"]: r for r in run["rows"]}
        self.assertEqual(set(rows), {"lp_simplex", "lp_ipm", "qp_ipm"})
        self.assertTrue(rows["lp_simplex"]["agrees"])
        self.assertEqual(rows["lp_simplex"]["reference"]["status"], "OPTIMAL")
        self.assertEqual(rows["lp_ipm"]["executionDevice"], "cuda")
        self.assertIsNone(rows["lp_ipm"]["agrees"])
        self.assertEqual(run["summary"]["agreements"], 1)
        self.assertAlmostEqual(run["summary"]["geomeanTimeRatio"], 0.05)
        self.assertNotIn("modelJson", json.dumps(run))
        self.assertEqual({m["name"] for m in run["machines"]}, {"npm connector", "python worker"})

        self.assertEqual(self.client.get("/api/benchmarks/runs", headers=self.admin).json()["runs"][0]["runId"], run_id)
        self.assertEqual(self.client.post(f"/api/benchmarks/runs/{run_id}/cancel", headers=self.admin).json()["cancelled"], 3)
        self.assertEqual(self.client.get(f"/api/benchmarks/runs/{run_id}", headers=self.admin).json()["state"], "finished")
        self.assertEqual(self.client.delete(f"/api/benchmarks/runs/{run_id}", headers=self.admin).status_code, 200)
        self.assertEqual(self.client.get(f"/api/benchmarks/runs/{run_id}", headers=self.admin).status_code, 404)

    def test_benchmark_run_validation(self):
        bad_profile = self.client.post("/api/benchmarks/runs", headers=self.admin,
                                       json={"datasets": ["afiro"], "profiles": ["nope"]})
        self.assertEqual(bad_profile.status_code, 422)
        mismatch = self.client.post("/api/benchmarks/runs", headers=self.admin,
                                    json={"datasets": ["afiro"], "profiles": ["qp_ipm"]})
        self.assertEqual(mismatch.status_code, 422)
        unknown = self.client.post("/api/benchmarks/runs", headers=self.admin,
                                   json={"datasets": ["../../etc/passwd"], "profiles": ["lp_ipm"]})
        self.assertEqual(unknown.status_code, 422)
        recorded = self.client.get("/api/benchmarks/recorded", headers=self.admin).json()
        self.assertGreater(len(recorded["rows"]), 0)

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
            with database() as store:
                store.force_expire_job(job_id)
        self.assertIsNone(self.claim(worker))
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}", headers=self.admin).json()["state"], "FAILED")

    def test_vercel_requires_persistent_database(self):
        with patch("api.cloud.VERCEL_FUNCTION", True), patch.dict(os.environ,
                {"VERCEL": "1", "MONGODB_URI": "", "SOVEREIGN_MONGODB_URI": ""}, clear=False):
            response = self.client.get("/api/workspace", headers=self.admin)
            health = self.client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        self.assertIn("MONGODB_URI", response.json()["detail"])
        self.assertEqual(health.status_code, 503)

    def test_connect_pairing_flow(self):
        request = self.client.post("/api/worker/connect/request", json={
            "deviceId": "device-test-001",
            "name": "Laptop",
            "capabilities": CAPS,
            "requestedDurationHours": 8,
        })
        self.assertEqual(request.status_code, 200)
        code = request.json()["code"]
        pending = self.client.get("/api/worker/connect/pending", headers=self.admin)
        self.assertEqual(pending.status_code, 200)
        self.assertEqual(pending.json()["pending"][0]["code"], code)
        approved = self.client.post("/api/worker/connect/approve", headers=self.admin,
                                    json={"code": code, "durationHours": 8})
        self.assertEqual(approved.status_code, 200)
        status = self.client.get("/api/worker/connect/status", params={"deviceId": "device-test-001"})
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "approved")
        self.assertTrue(status.json()["token"])

    def test_vercel_forwarded_origin(self):
        headers = {**self.admin, "Host": "example.vercel.app",
                   "Origin": "https://example.vercel.app", "X-Forwarded-Proto": "https"}
        scope = {"type": "http", "scheme": "http", "server": ("testserver", 80),
                 "path": "/api/workspace", "root_path": "",
                 "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()]}
        with patch("api.cloud.VERCEL_FUNCTION", True):
            current_user(Request(scope))
            scope["headers"] = [(key, b"https://elsewhere.invalid" if key == b"origin" else value)
                                for key, value in scope["headers"]]
            with self.assertRaises(HTTPException):
                current_user(Request(scope))


if __name__ == "__main__":
    unittest.main()
