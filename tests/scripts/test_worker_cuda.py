"""An explicit CUDA request must not be reported as a GPU solve with zero kernels."""
import json
import unittest
from unittest.mock import patch

from worker.runner import execute, fit_completion


class FakeProcess:
    returncode = 0

    def __init__(self, output):
        self.output = output

    def communicate(self, timeout=None):
        return self.output, ""

    def poll(self):
        return 0


class CudaExecutionTests(unittest.TestCase):
    def run_case(self, device, operations, reported_device):
        result = {"status": "OPTIMAL", "objective_value": 1,
                  "gpu_operations": operations, "gpu_used": operations > 0,
                  "requested_device": reported_device}
        output = json.dumps(result) + "\nverification:\n" + json.dumps({"is_valid": True})
        job = {"id": "job", "lease": "lease", "request": {
            "modelFormat": "json", "modelJson": '{"variables":[{"name":"x"}]}',
            "device": device, "executionDevice": device, "algorithm": "ipm",
            "timeLimitSeconds": 5}}
        with patch("worker.runner.subprocess.Popen", return_value=FakeProcess(output)):
            return execute(None, "ignored-engine", job)

    def test_zero_cuda_operations_fail_explicit_request(self):
        completion = self.run_case("cuda", 0, "cuda")
        self.assertIn("error", completion)
        self.assertIn("0 GPU operations", completion["error"])
        self.assertEqual(completion["result"]["status"], "OPTIMAL")

    def test_reported_cpu_device_fails_explicit_request(self):
        completion = self.run_case("cuda", 12, "cpu")
        self.assertIn("error", completion)
        self.assertIn("executable=cpu", completion["error"])

    def test_real_cuda_operations_pass(self):
        completion = self.run_case("cuda", 12, "cuda")
        self.assertNotIn("error", completion)
        self.assertEqual(completion["result"]["gpu_operations"], 12)

    def test_presolve_only_answer_retries_and_runs_cuda(self):
        seen = []

        def launch(*args, **kwargs):
            presolve = kwargs["env"]["SOVEREIGN_PRESOLVE"]
            seen.append(presolve)
            operations = 0 if presolve == "1" else 14
            result = {"status": "OPTIMAL", "objective_value": 1,
                      "gpu_operations": operations, "gpu_used": operations > 0,
                      "requested_device": "cuda"}
            output = json.dumps(result) + "\nverification:\n" + json.dumps({"is_valid": True})
            return FakeProcess(output)

        job = {"id": "job", "lease": "lease", "request": {
            "modelFormat": "json", "modelJson": '{"variables":[{"name":"x"}]}',
            "device": "cuda", "executionDevice": "cuda", "algorithm": "ipm",
            "presolve": True, "timeLimitSeconds": 5}}
        with patch("worker.runner.subprocess.Popen", side_effect=launch):
            completion = execute(None, "ignored-engine", job)
        self.assertEqual(seen, ["1", "0"])
        self.assertNotIn("error", completion)
        self.assertEqual(completion["result"]["gpu_operations"], 14)
        self.assertFalse(completion["result"]["configuration"]["effectivePresolve"])

    def test_oversize_completion_becomes_clear_failure(self):
        completion = {"jobId": "job", "lease": "lease", "result": {"primal": "x" * 100}}
        with patch.dict("os.environ", {"SOVEREIGN_WORKER_MAX_PAYLOAD_BYTES": "80"}):
            limited = fit_completion(completion)
        self.assertEqual(limited["jobId"], "job")
        self.assertNotIn("result", limited)
        self.assertIn("exceeded", limited["error"])


if __name__ == "__main__":
    unittest.main()
