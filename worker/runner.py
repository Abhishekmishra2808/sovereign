"""Run with Python 3.10+. Only outbound HTTPS and a local Sovereign CLI are needed."""
from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import math
import os
import platform
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class Client:
    def __init__(self, url, token):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")):
            raise ValueError("Use HTTPS for remote coordinators (HTTP is only allowed on loopback).")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use the coordinator's base URL without credentials, query, or fragment.")
        self.url, self.token = url.rstrip("/"), token
        # Do not forward a worker credential to a redirect destination.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        self.opener = urllib.request.build_opener(NoRedirect())

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        with self.opener.open(req, timeout=30) as response:
            return json.load(response)

    def get(self, path):
        with self.opener.open(urllib.request.Request(self.url + path), timeout=30) as response:
            return json.load(response)


def highs_version():
    if importlib.util.find_spec("highspy") is None:
        return ""
    try:
        return subprocess.check_output([sys.executable, "-c", "import highspy; print(highspy.Highs().version())"],
                                       text=True, timeout=30).strip()[:60]
    except (OSError, subprocess.SubprocessError):
        return ""


def capabilities(engine):
    info = json.loads(subprocess.check_output([engine, "capabilities"], text=True, timeout=15))
    gpu_name = ""
    try:
        gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                           text=True, timeout=10).strip()[:250]
    except (OSError, subprocess.SubprocessError):
        pass
    highs = highs_version()
    return {"hostname": socket.gethostname(), "platform": platform.system(), "cpu_threads": os.cpu_count() or 1,
            "cuda_available": bool(info.get("cuda_available")), "gpu_name": gpu_name,
            "engine_version": info.get("version", "unknown"),
            "reference_solvers": ["highs"] if highs else [], "highs_version": highs}


def highs_reference(path, limit):
    """Benchmark reference only; never used for production jobs."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(limit))
    h.setOptionValue("threads", 1)
    path = Path(path)
    if path.suffix == ".mps":
        assert h.readModel(str(path)) == highspy.HighsStatus.kOk
    else:
        m = json.loads(path.read_text(encoding="utf-8"))
        variables = m["variables"]
        index = {v["name"]: i for i, v in enumerate(variables)}
        for i, v in enumerate(variables):
            lb, ub = v.get("lower_bound", 0), v.get("upper_bound", 1 if v.get("type") == "binary" else 1e30)
            h.addVar(lb, h.getInfinity() if ub >= 1e29 else ub)
            if v.get("type") in ("integer", "binary"):
                h.changeColIntegrality(i, highspy.HighsVarType.kInteger)
        obj = m.get("objective", {})
        for name, value in obj.get("linear", {}).items():
            h.changeColCost(index[name], value)
        h.changeObjectiveOffset(obj.get("constant", 0))
        h.changeObjectiveSense(highspy.ObjSense.kMaximize if m.get("sense") == "maximize" else highspy.ObjSense.kMinimize)
        for row in m.get("constraints", []):
            coeffs = row.get("linear", {})
            rhs, sense = row["rhs"], row["sense"]
            h.addRow(rhs if sense in (">=", "=") else -h.getInfinity(),
                     rhs if sense in ("<=", "=") else h.getInfinity(), len(coeffs),
                     list(index[k] for k in coeffs), list(coeffs.values()))
        q = obj.get("quadratic", {})
        if q:
            import numpy as np
            # Objective is 0.5*x'Q*x. Symmetrize off-diagonal entries before
            # passing HiGHS' lower triangular Hessian, preserving that objective.
            entries = {}
            for a, row in q.items():
                for b, value in row.items():
                    i, j = index[a], index[b]
                    key = (max(i, j), min(i, j))
                    entries[key] = entries.get(key, 0) + value * (1 if i == j else .5)
            starts, indices, values = [0], [], []
            for j in range(len(variables)):
                for (i, col), value in sorted(entries.items()):
                    if col == j:
                        indices.append(i); values.append(value)
                starts.append(len(indices))
            status = h.passHessian(len(variables), len(values), highspy.HessianFormat.kTriangular,
                                  np.array(starts, dtype=np.int32), np.array(indices, dtype=np.int32), np.array(values))
            assert status == highspy.HighsStatus.kOk
    start = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - start
    status = h.modelStatusToString(h.getModelStatus()).upper().replace(" ", "_")
    info = h.getInfo()
    return {"status": status, "objective": h.getObjectiveValue() if h.getSolution().value_valid else None,
            "runtime_seconds": elapsed, "mip_gap": info.mip_gap if math.isfinite(info.mip_gap) else None,
            "version": h.version()}


HIGHS_STATUS = {"TIME_LIMIT_REACHED": "TIME_LIMIT", "ITERATION_LIMIT_REACHED": "ITERATION_LIMIT"}


def parse_result(stdout):
    head, marker, tail = stdout.partition("\nverification:")
    result = json.loads(head)
    if marker:
        result["verification"] = json.loads(tail)
    return result


def _solve_once(client, engine, path, env, lease, deadline, last_heartbeat, command=None):
    proc = subprocess.Popen(command or [engine, "solve", str(path), "--verify"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=1)
                return (stdout, stderr, proc.returncode), last_heartbeat, None
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() > deadline:
                return None, last_heartbeat, "time_limit"
            if time.monotonic() - last_heartbeat >= 10:
                try:
                    client.post("/api/worker/heartbeat", lease)
                    last_heartbeat = time.monotonic()
                except urllib.error.HTTPError as exc:
                    if exc.code in (401, 409):
                        return None, last_heartbeat, "lease_lost"
                    if time.monotonic() - last_heartbeat > 45:
                        return None, last_heartbeat, "lease_lost"
                except (OSError, ValueError):
                    if time.monotonic() - last_heartbeat > 45:
                        return None, last_heartbeat, "lease_lost"
                time.sleep(1)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def execute_reference(client, job, path):
    req = job["request"]
    lease = {"jobId": job["id"], "lease": job["lease"]}
    limit = req["timeLimitSeconds"]
    command = [sys.executable, str(Path(__file__).resolve()), "--highs-reference", str(path), "--time-limit", str(limit)]
    output, _, problem = _solve_once(client, None, path, os.environ.copy(), lease,
                                     time.monotonic() + limit + 20, time.monotonic(), command)
    if problem == "lease_lost":
        return None
    if problem == "time_limit":
        return {**lease, "result": {"solver": "highs", "status": "TIME_LIMIT", "objective_value": None,
                                    "runtime_seconds": float(limit), "message": "HiGHS reached the time limit."}}
    stdout, stderr, returncode = output
    if returncode:
        return {**lease, "error": "HiGHS reference failed: " + (stderr or stdout or "unknown error")[-1000:]}
    ref = json.loads(stdout)
    status = HIGHS_STATUS.get(ref["status"], ref["status"])
    return {**lease, "result": {"solver": "highs", "version": ref.get("version"), "status": status,
                                "objective_value": ref.get("objective"), "runtime_seconds": ref.get("runtime_seconds"),
                                "mip_gap": ref.get("mip_gap"), "message": f"HiGHS {ref.get('version')} (1 thread): {status}"}}


def execute(client, engine, job):
    req = job["request"]
    lease = {"jobId": job["id"], "lease": job["lease"]}
    with tempfile.TemporaryDirectory(prefix="sovereign-job-") as folder:
        path = Path(folder) / ("model." + req["modelFormat"])
        path.write_text(req["modelJson"], encoding="utf-8")
        if req.get("solver") == "highs":
            return execute_reference(client, job, path)
        env = os.environ.copy()
        for key in [k for k in env if k.startswith("SOVEREIGN_")]:
            env.pop(key, None)
        env["SOVEREIGN_DEVICE"] = req.get("executionDevice", req["device"])
        env["SOVEREIGN_LP_ALGORITHM"] = req["algorithm"]
        env["SOVEREIGN_QP_ALGORITHM"] = req.get("qpAlgorithm", "auto")
        env["SOVEREIGN_BRANCH_RULE"] = req.get("branchRule", "strong")
        env["SOVEREIGN_ENABLE_CUTS"] = "1" if req.get("milpMethod", "branch_and_cut") == "branch_and_cut" else "0"
        env["SOVEREIGN_PRESOLVE"] = "1" if req.get("presolve", True) else "0"
        env["SOVEREIGN_PARALLEL_WORKERS"] = "2" if req.get("parallelBranching", False) else "1"
        env["SOVEREIGN_MAX_NODES"] = str(req.get("maxNodes", 100000))
        deadline = time.monotonic() + req["timeLimitSeconds"]
        last_heartbeat = time.monotonic()
        first_result = None
        retried_without_presolve = False
        for attempt in range(2):
            if time.monotonic() >= deadline:
                if first_result:
                    return {**lease, "result": first_result, "error":
                        f"Time limit reached ({req['timeLimitSeconds']} seconds) during the CUDA retry."}
                return {**lease, "error": f"Time limit reached ({req['timeLimitSeconds']} seconds)."}
            # The engine stops its own search a little early so it can still
            # report the incumbent instead of being killed at the deadline.
            remaining = deadline - time.monotonic()
            env["SOVEREIGN_TIME_LIMIT"] = f"{max(0.5, remaining - max(1.0, 0.1 * remaining)):.2f}"
            output, last_heartbeat, problem = _solve_once(
                client, engine, path, env, lease, deadline, last_heartbeat)
            if problem == "lease_lost":
                return None
            if problem == "time_limit":
                if first_result:
                    return {**lease, "result": first_result, "error":
                        f"Time limit reached ({req['timeLimitSeconds']} seconds) during the CUDA retry."}
                return {**lease, "error": f"Time limit reached ({req['timeLimitSeconds']} seconds)."}
            stdout, stderr, returncode = output
            if returncode:
                if first_result is not None:
                    return {**lease, "result": first_result, "error":
                        "CUDA retry without presolve failed: " + (stderr or stdout or "Solver failed")[-1000:]}
                return {**lease, "error": (stderr or stdout or "Solver failed")[-4000:]}
            try:
                result = parse_result(stdout)
            except (ValueError, TypeError) as exc:
                return {**lease, "error": f"Invalid solver output: {exc}"}
            result["configuration"] = {k: v for k, v in req.items() if k not in ("modelJson", "name")}
            result["configuration"]["effectivePresolve"] = env["SOVEREIGN_PRESOLVE"] == "1"
            assigned = req.get("executionDevice", req["device"])
            reported = result.get("requested_device")
            operations = result.get("gpu_operations", 0)
            used_cuda = assigned == "cuda" and reported == "cuda" and isinstance(operations, int) and operations > 0 and result.get("gpu_used")
            if req["device"] == "cuda" and not used_cuda:
                if attempt == 0 and env["SOVEREIGN_PRESOLVE"] == "1" and result.get("status") in ("OPTIMAL", "FEASIBLE"):
                    first_result = result
                    env["SOVEREIGN_PRESOLVE"] = "0"
                    retried_without_presolve = True
                    continue
                return {**lease, "result": first_result or result, "error":
                    f"CUDA was selected, but the worker reported {operations} GPU operations "
                    f"(assigned={assigned}, executable={reported}). The job was not accepted as GPU execution. "
                    "Check the CUDA-enabled worker and choose an IPM model with constraints."}
            if retried_without_presolve:
                result.setdefault("warnings", []).append(
                    "CUDA request reran without presolve because the first solve used zero GPU kernels.")
            return {**lease, "result": result}


def fit_completion(completion):
    """Keep a worker upload below the hosted function's body limit."""
    max_bytes = int(os.environ.get("SOVEREIGN_WORKER_MAX_PAYLOAD_BYTES", "4000000"))
    if len(json.dumps(completion, allow_nan=False).encode("utf-8")) <= max_bytes:
        return completion
    return {"jobId": completion["jobId"], "lease": completion["lease"],
            "error": "Solver result exceeded the hosted 4 MB response limit; use a smaller model or a coordinator with large-result storage."}


def run(client, engine, once=False):
    caps = capabilities(engine)
    print(f"Connected worker: {caps['hostname']} | CUDA {'ready' if caps['cuda_available'] else 'unavailable'}", flush=True)
    while True:
        try:
            job = client.post("/api/worker/claim", caps)["job"]
            if job:
                print(f"Running job {job['id']}", flush=True)
                completion = execute(client, engine, job)
                if completion:
                    completion = fit_completion(completion)
                    # Keep the result in memory until acknowledged; never claim another job first.
                    while True:
                        try:
                            client.post("/api/worker/complete", completion)
                            print(f"Returned job {job['id']}", flush=True)
                            break
                        except urllib.error.HTTPError as exc:
                            if exc.code == 409:
                                print("Lease expired or job cancelled; discarded stale result.", flush=True)
                                break
                            if exc.code < 500:
                                raise
                        except OSError:
                            pass
                        time.sleep(5)
                if once:
                    return
                continue
            elif once:
                return
            time.sleep(3)
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise RuntimeError(f"Coordinator rejected worker request (HTTP {exc.code}). Check the worker key.") from exc
            print(f"Coordinator unavailable ({exc.code}); reconnecting in 5s.", flush=True)
            time.sleep(5)
        except OSError:
            print("Connection interrupted; reconnecting in 5s.", flush=True)
            time.sleep(5)


def pair(server, engine, hours):
    """Get a worker key by approving a 6-digit code on the website, like the npm connector."""
    client = Client(server, "")
    device_id = "py-" + secrets.token_hex(12)
    caps = capabilities(engine)
    code = client.post("/api/worker/connect/request", {
        "deviceId": device_id, "name": caps["hostname"], "capabilities": caps,
        "requestedDurationHours": hours})["code"]
    print(f"Pairing code: {code}\nApprove it on the website under Machines > Connect machine.", flush=True)
    while True:
        time.sleep(3)
        status = client.get("/api/worker/connect/status?" + urllib.parse.urlencode({"deviceId": device_id}))
        if status["status"] == "approved" and status.get("token"):
            print(f"Approved: {status.get('name')} until {status.get('expiresAt')}", flush=True)
            return status["token"]
        if status["status"] in ("expired", "rejected", "missing"):
            raise RuntimeError(f"Pairing {status['status']}. Run the worker again for a new code.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.environ.get("SOVEREIGN_SERVER_URL"))
    parser.add_argument("--engine", default=os.environ.get("SOVEREIGN_BIN"))
    parser.add_argument("--once", action="store_true", help="Handle at most one job, then exit (smoke test).")
    parser.add_argument("--pair", action="store_true", help="Get a worker key by approving a code on the website.")
    parser.add_argument("--duration", type=int, default=8, help="Requested pairing duration in hours.")
    parser.add_argument("--highs-reference", help=argparse.SUPPRESS)
    parser.add_argument("--time-limit", type=float, default=30, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.highs_reference:
        print(json.dumps(highs_reference(args.highs_reference, args.time_limit), allow_nan=False))
        return
    if not args.server or not args.engine:
        parser.error("--server and --engine are required (or set SOVEREIGN_SERVER_URL and SOVEREIGN_BIN).")
    engine = str(Path(args.engine).resolve())
    try:
        token = os.environ.get("SOVEREIGN_WORKER_TOKEN")
        if not token:
            token = pair(args.server, engine, args.duration) if args.pair else getpass.getpass("Worker key: ")
        run(Client(args.server, token), engine, args.once)
    except KeyboardInterrupt:
        print("Worker stopped.")


if __name__ == "__main__":
    main()
