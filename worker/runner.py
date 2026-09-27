"""Run with Python 3.10+. Only outbound HTTPS and a local Sovereign CLI are needed."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import socket
import subprocess
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
        with self.opener.open(req, timeout=15) as response:
            return json.load(response)


def capabilities(engine):
    info = json.loads(subprocess.check_output([engine, "capabilities"], text=True, timeout=15))
    gpu_name = ""
    try:
        gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                           text=True, timeout=10).strip()[:250]
    except (OSError, subprocess.SubprocessError):
        pass
    return {"hostname": socket.gethostname(), "platform": platform.system(), "cpu_threads": os.cpu_count() or 1,
            "cuda_available": bool(info.get("cuda_available")), "gpu_name": gpu_name,
            "engine_version": info.get("version", "unknown")}


def parse_result(stdout):
    head, marker, tail = stdout.partition("\nverification:")
    result = json.loads(head)
    if marker:
        result["verification"] = json.loads(tail)
    return result


def _solve_once(client, engine, path, env, lease, deadline, last_heartbeat):
    proc = subprocess.Popen([engine, "solve", str(path), "--verify"], stdout=subprocess.PIPE,
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


def execute(client, engine, job):
    req = job["request"]
    lease = {"jobId": job["id"], "lease": job["lease"]}
    with tempfile.TemporaryDirectory(prefix="sovereign-job-") as folder:
        path = Path(folder) / ("model." + req["modelFormat"])
        path.write_text(req["modelJson"], encoding="utf-8")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.environ.get("SOVEREIGN_SERVER_URL"), required=not os.environ.get("SOVEREIGN_SERVER_URL"))
    parser.add_argument("--engine", default=os.environ.get("SOVEREIGN_BIN"), required=not os.environ.get("SOVEREIGN_BIN"))
    parser.add_argument("--once", action="store_true", help="Handle at most one job, then exit (smoke test).")
    args = parser.parse_args()
    token = os.environ.get("SOVEREIGN_WORKER_TOKEN") or getpass.getpass("Worker key: ")
    try:
        run(Client(args.server, token), str(Path(args.engine).resolve()), args.once)
    except KeyboardInterrupt:
        print("Worker stopped.")


if __name__ == "__main__":
    main()
