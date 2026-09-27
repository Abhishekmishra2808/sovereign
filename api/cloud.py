"""Mediator coordinator: website <-> user machines. No solving happens here."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.datasets import catalogue, dataset
from api.routing import route_request
from api.storage import open_store

database = open_store  # Backward compatibility for tests and older imports.

ROOT = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 60
PAIRING_TTL_SECONDS = 900
VERCEL_FUNCTION = os.environ.get("VERCEL") == "1"
MAX_MODEL_CHARACTERS = 3_500_000 if VERCEL_FUNCTION else 5_000_000
MAX_RESULT_BYTES = 4_000_000 if VERCEL_FUNCTION else 20_000_000
ALLOWED_DURATIONS = {1, 2, 4, 8, 12, 24, 48, 72}
app = FastAPI(title="Sovereign workspace", version="2.1")


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def admin_secret() -> str:
    token = os.environ.get("SOVEREIGN_ADMIN_TOKEN", "")
    if len(token) < 24:
        raise HTTPException(503, "Set SOVEREIGN_ADMIN_TOKEN to a random secret of at least 24 characters.")
    return token


def session_value() -> str:
    return hmac.new(admin_secret().encode(), b"sovereign-browser-session-v1", hashlib.sha256).hexdigest()


def storage_mode() -> str:
    if os.environ.get("MONGODB_URI") or os.environ.get("SOVEREIGN_MONGODB_URI"):
        return "mongodb"
    return "sqlite"


def admin(request: Request):
    origin = request.headers.get("origin")
    expected_origin = str(request.base_url).rstrip("/")
    if VERCEL_FUNCTION:
        scheme = request.headers.get("x-forwarded-proto", "https").split(",", 1)[0].strip()
        expected_origin = f"{scheme}://{request.headers.get('host', '')}"
    allowed_origins = {expected_origin}
    public_origin = os.environ.get("SOVEREIGN_PUBLIC_ORIGIN", "").rstrip("/")
    if public_origin:
        allowed_origins.add(public_origin)
    if origin and origin.rstrip("/") not in allowed_origins:
        raise HTTPException(403, "Cross-origin requests are not allowed.")
    bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
    cookie = request.cookies.get("sovereign_session", "")
    if not (hmac.compare_digest(bearer, admin_secret()) or hmac.compare_digest(cookie, session_value())):
        raise HTTPException(401, "Sign in to your workspace.")


def worker_auth(request: Request) -> str:
    token = request.headers.get("authorization", "").removeprefix("Bearer ")
    with open_store() as store:
        worker_id = store.worker_id_for_token(digest(token))
    if not worker_id:
        raise HTTPException(401, "Worker key is invalid or revoked.")
    return worker_id


class Login(BaseModel):
    token: str = Field(max_length=256)


@app.post("/api/session")
def login(body: Login, request: Request, response: Response):
    if not hmac.compare_digest(body.token, admin_secret()):
        raise HTTPException(401, "That workspace key is incorrect.")
    response.set_cookie("sovereign_session", session_value(), httponly=True,
                        secure=VERCEL_FUNCTION or request.url.scheme == "https",
                        samesite="strict", max_age=43200)
    return {"authenticated": True}


@app.delete("/api/session", dependencies=[Depends(admin)])
def logout(response: Response):
    response.delete_cookie("sovereign_session")
    return {"authenticated": False}


@app.get("/api/health")
def health():
    admin_secret()
    with open_store():
        pass
    return {"status": "ok", "mode": "mediator", "storage": storage_mode(),
            "compute": "remote-workers-only"}


@app.get("/api/datasets", dependencies=[Depends(admin)])
def datasets():
    return {"datasets": catalogue()}


@app.get("/api/datasets/{dataset_id}", dependencies=[Depends(admin)])
def get_dataset(dataset_id: str):
    try:
        return dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(404, "Dataset not found.") from exc


@app.get("/api/benchmark-report", dependencies=[Depends(admin)])
def benchmark_report():
    path = ROOT / "benchmarks" / "reports" / "sih-online.json"
    if not path.is_file():
        return {"rows": [], "note": "No SIH run recorded yet."}
    return json.loads(path.read_text(encoding="utf-8"))


def job_view(row, full=False):
    item = dict(row)
    req = json.loads(item.pop("request"))
    item.pop("lease", None)
    item.pop("expires", None)
    item["device"] = req["device"]
    item["target"] = req.get("workerId")
    item["routing"] = req.get("routing")
    item["algorithm"] = req.get("algorithm", "auto")
    result = json.loads(item.pop("result") or "null")
    if full:
        item["result"] = result
    elif result:
        item["solver_status"] = result.get("status")
        item["runtime_seconds"] = result.get("runtime_seconds")
        item["gpu_used"] = result.get("gpu_used", False)
    return item


@app.get("/api/workspace", dependencies=[Depends(admin)])
def workspace():
    now = time.time()
    with open_store() as store:
        store.reap(now)
        workers = []
        for row in store.list_workers():
            item = dict(row)
            item["capabilities"] = json.loads(item["capabilities"])
            item["online"] = item["seen"] > now - LEASE_SECONDS
            if item.get("expires_at"):
                item["connection_expires_at"] = item["expires_at"]
            workers.append(item)
        jobs = [job_view(row) for row in store.list_jobs()]
    return {"workers": workers, "jobs": jobs}


class Pair(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@app.post("/api/workers", dependencies=[Depends(admin)], status_code=201)
def pair(body: Pair):
    token, worker_id = secrets.token_urlsafe(32), secrets.token_hex(12)
    with open_store() as store:
        store.insert_worker(worker_id, body.name, digest(token))
    return {"id": worker_id, "token": token, "name": body.name}


@app.delete("/api/workers/{worker_id}", dependencies=[Depends(admin)])
def revoke(worker_id: str):
    now = time.time()
    with open_store() as store:
        store.revoke_worker(worker_id, now)
        store.reap(now)
    return {"revoked": True}


class ConnectRequest(BaseModel):
    deviceId: str = Field(min_length=8, max_length=80)
    name: str = Field(min_length=1, max_length=80)
    capabilities: dict
    requestedDurationHours: int | None = Field(default=None, ge=1, le=72)


@app.post("/api/worker/connect/request")
def connect_request(body: ConnectRequest):
    code = f"{secrets.randbelow(1_000_000):06d}"
    expires = time.time() + PAIRING_TTL_SECONDS
    with open_store() as store:
        store.create_pairing(body.deviceId, code, body.name, body.capabilities,
                             body.requestedDurationHours, expires)
    return {"code": code, "expiresInSeconds": PAIRING_TTL_SECONDS}


@app.get("/api/worker/connect/status")
def connect_status(deviceId: str = Query(min_length=8, max_length=80)):
    now = time.time()
    with open_store() as store:
        store.reap(now)
        row = store.get_pairing_by_device(deviceId)
        if not row:
            return {"status": "missing"}
        if row["status"] == "pending":
            if row["expires"] < now:
                return {"status": "expired"}
            return {"status": "pending", "code": row["code"], "expiresInSeconds": max(0, int(row["expires"] - now))}
        if row["status"] == "approved":
            return {
                "status": "approved",
                "workerId": row["worker_id"],
                "name": row["name"],
                "token": row.get("issued_token"),
                "durationHours": row.get("duration_hours"),
                "expiresAt": _iso(row.get("connection_expires")),
            }
        return {"status": row["status"]}


def _iso(value):
    if not value:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


class ConnectApprove(BaseModel):
    code: str = Field(min_length=6, max_length=6)
    durationHours: int = Field(default=8, ge=1, le=72)


@app.get("/api/worker/connect/pending", dependencies=[Depends(admin)])
def connect_pending():
    now = time.time()
    with open_store() as store:
        store.reap(now)
        pending = [{
            "code": row["code"],
            "name": row["name"],
            "deviceId": row["device_id"],
            "capabilities": json.loads(row["capabilities"]),
            "requestedDurationHours": row.get("requested_hours"),
            "expiresInSeconds": max(0, int(row["expires"] - now)),
        } for row in store.list_pending_pairings(now)]
    return {"pending": pending}


@app.post("/api/worker/connect/approve", dependencies=[Depends(admin)])
def connect_approve(body: ConnectApprove):
    if body.durationHours not in ALLOWED_DURATIONS:
        raise HTTPException(422, f"Choose one of: {sorted(ALLOWED_DURATIONS)} hours.")
    now = time.time()
    token = secrets.token_urlsafe(32)
    worker_id = secrets.token_hex(12)
    expires_at = now + body.durationHours * 3600
    with open_store() as store:
        store.reap(now)
        row = store.approve_pairing(body.code, worker_id, digest(token), body.durationHours, expires_at, now)
        if not row:
            raise HTTPException(404, "Pairing code is invalid or expired.")
        store.save_issued_token(body.code, token)
    return {"workerId": worker_id, "name": row["name"], "durationHours": body.durationHours,
            "expiresAt": _iso(expires_at)}


class JobRequest(BaseModel):
    name: str = Field(default="Untitled problem", min_length=1, max_length=120)
    modelJson: str = Field(min_length=2, max_length=MAX_MODEL_CHARACTERS)
    modelFormat: Literal["json", "mps"] = "json"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    workerId: str | None = None
    algorithm: Literal["auto", "simplex", "ipm"] = "auto"
    qpAlgorithm: Literal["auto", "ipm", "frank_wolfe"] = "auto"
    milpMethod: Literal["branch_and_cut", "branch_and_bound"] = "branch_and_cut"
    branchRule: Literal["strong", "pseudocost", "most_fractional"] = "strong"
    presolve: bool = True
    parallelBranching: bool = False
    maxNodes: int = Field(default=100000, ge=1, le=1000000)
    timeLimitSeconds: int = Field(default=300, ge=1, le=86400)


@app.post("/api/route", dependencies=[Depends(admin)])
def preview_route(body: JobRequest):
    try:
        return route_request(body.model_dump())
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        raise HTTPException(422, "The model structure could not be read. Check its format.") from exc


@app.post("/api/jobs", dependencies=[Depends(admin)], status_code=201)
def submit(body: JobRequest):
    if body.modelFormat == "json":
        try:
            model = json.loads(body.modelJson, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if not isinstance(model, dict) or not isinstance(model.get("variables"), list) or not model["variables"]:
                raise ValueError("A JSON model needs a non-empty variables array.")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, f"Invalid model: {exc}") from exc
    routing = preview_route(body)
    if body.device == "cuda":
        shape = routing["shape"]
        kind = shape["problem_type"].upper()
        if kind in ("LP", "MILP") and body.algorithm == "simplex":
            raise HTTPException(422, "CUDA jobs need LP interior point or automatic LP selection; revised simplex currently runs on CPU.")
        if kind == "QP" and body.qpAlgorithm == "frank_wolfe":
            raise HTTPException(422, "CUDA jobs need QP interior point or automatic QP selection; Frank-Wolfe currently runs on CPU.")
    job_id, now = secrets.token_hex(12), time.time()
    req = body.model_dump()
    req["routing"] = routing
    with open_store() as store:
        if body.workerId and not store.worker_exists(body.workerId):
            raise HTTPException(422, "Choose an existing machine.")
        store.insert_job(job_id, body.name, now, json.dumps(req))
    return {"jobId": job_id, "state": "QUEUED"}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(admin)])
def job(job_id: str):
    now = time.time()
    with open_store() as store:
        store.reap(now)
        row = store.get_job(job_id)
        if not row:
            raise HTTPException(404, "Job not found.")
        return job_view(row, full=True)


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(admin)])
def cancel(job_id: str):
    with open_store() as store:
        changed = store.cancel_job(job_id, time.time())
    return {"cancelled": changed}


class Capabilities(BaseModel):
    hostname: str = Field(max_length=120)
    platform: str = Field(max_length=120)
    cpu_threads: int = Field(ge=1, le=65536)
    cuda_available: bool = False
    gpu_name: str = Field(default="", max_length=250)
    engine_version: str = Field(default="", max_length=250)


@app.post("/api/worker/claim")
def claim(body: Capabilities, worker_id: str = Depends(worker_auth)):
    now = time.time()
    with open_store() as store:
        store.reap(now)
        store.update_worker_seen(worker_id, body.model_dump_json(), now)
        if store.worker_solving_job(worker_id):
            return {"job": None}
        for row in store.queued_jobs():
            req = json.loads(row["request"])
            if req.get("workerId") not in (None, worker_id):
                continue
            if req["device"] == "cuda" and not body.cuda_available:
                continue
            routing = req.get("routing") or route_request(req)
            execution = req["device"]
            if execution == "auto":
                execution = "cpu"
                if routing["preferred_device"] == "cuda":
                    if body.cuda_available:
                        execution = "cuda"
                    else:
                        gpu_online = any(json.loads(w["capabilities"]).get("cuda_available") and
                            req.get("workerId") in (None, w["id"]) for w in store.online_workers(now - LEASE_SECONDS))
                        if gpu_online:
                            continue
                        routing["reason"] = "Large sparse model; CPU fallback because no matching CUDA worker is online."
            routing["execution_device"] = execution
            req["routing"] = routing
            req["executionDevice"] = execution
            lease = secrets.token_urlsafe(24)
            store.claim_job(row["id"], worker_id, lease, now + LEASE_SECONDS, now, json.dumps(req))
            return {"job": {"id": row["id"], "lease": lease, "request": req}}
    return {"job": None}


class Lease(BaseModel):
    jobId: str
    lease: str


@app.post("/api/worker/heartbeat")
def heartbeat(body: Lease, worker_id: str = Depends(worker_auth)):
    now = time.time()
    with open_store() as store:
        store.reap(now)
        try:
            row = store.owned_job(body.jobId, worker_id, body.lease)
        except ValueError:
            raise HTTPException(409, "Job lease is no longer yours.") from None
        if row["state"] != "SOLVING":
            raise HTTPException(409, "Job is no longer running.")
        store.extend_job_lease(body.jobId, now + LEASE_SECONDS)
        store.touch_worker(worker_id, now)
    return {"continue": True}


class Completion(Lease):
    result: dict | None = None
    error: str | None = Field(default=None, max_length=4000)


@app.post("/api/worker/complete")
def complete(body: Completion, worker_id: str = Depends(worker_auth)):
    if body.result is None and not body.error:
        raise HTTPException(422, "A result or error is required.")
    try:
        encoded = json.dumps(body.result, allow_nan=False) if body.result is not None else None
    except ValueError as exc:
        raise HTTPException(422, "Result contains a non-finite number.") from exc
    if encoded and len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        raise HTTPException(413, f"Result exceeds {MAX_RESULT_BYTES // 1_000_000} MB.")
    status = body.result.get("status") if body.result else None
    verification = body.result.get("verification") if body.result else None
    verified = isinstance(verification, dict) and verification.get("is_valid") is True
    conclusive = status in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED")
    message = body.error
    if body.result and not message and (not conclusive or not verified):
        message = "Solver result did not pass verification." if not verified else f"Solver stopped with status {status}."
    now = time.time()
    with open_store() as store:
        store.reap(now)
        try:
            row = store.owned_job(body.jobId, worker_id, body.lease)
        except ValueError:
            raise HTTPException(409, "Job lease is no longer yours.") from None
        if row["state"] in ("COMPLETED", "FAILED"):
            return {"accepted": True}
        if row["state"] != "SOLVING":
            raise HTTPException(409, "Job is no longer running.")
        store.complete_job(body.jobId, "FAILED" if message else "COMPLETED", encoded, message, now)
        store.touch_worker(worker_id, now)
    return {"accepted": True}


@app.get("/worker.py")
def worker_download():
    return FileResponse(ROOT / "worker" / "runner.py", filename="sovereign-worker.py", media_type="text/x-python")


@app.get("/")
def index():
    return {"service": "Sovereign mediator", "health": "/api/health", "compute": "remote-workers-only"}
