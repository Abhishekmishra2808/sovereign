"""Single-workspace coordinator. The website stores jobs; outbound workers solve."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from api.routing import route_request
from api.datasets import catalogue, dataset

ROOT = Path(__file__).resolve().parents[1]
LEASE_SECONDS = 60
VERCEL_FUNCTION = os.environ.get("VERCEL") == "1"
MAX_MODEL_CHARACTERS = 3_500_000 if VERCEL_FUNCTION else 5_000_000
MAX_RESULT_BYTES = 4_000_000 if VERCEL_FUNCTION else 20_000_000
app = FastAPI(title="Sovereign workspace", version="2.0")


class PostgresDatabase:
    """Small adapter for the coordinator's fixed, parameterized SQLite queries."""

    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql, params=()):
        # All SQL is defined in this module; user content is always a parameter.
        return self.connection.execute(sql.replace("?", "%s"), params)


@contextmanager
def database():
    url = os.environ.get("SOVEREIGN_DATABASE_URL")
    if not url and VERCEL_FUNCTION:
        url = os.environ.get("DATABASE_URL")
    if url:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("Install psycopg[binary] for Postgres storage.") from exc
        db = psycopg.connect(url, row_factory=dict_row, connect_timeout=15)
        try:
            # One transaction lock serializes claims across separate Vercel
            # function instances, preserving the existing one-claim contract.
            db.execute("SELECT pg_advisory_xact_lock(26119)")
            db.execute("""CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, token_hash TEXT NOT NULL,
                capabilities TEXT NOT NULL DEFAULT '{}', seen DOUBLE PRECISION NOT NULL DEFAULT 0,
                revoked INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL,
                created DOUBLE PRECISION NOT NULL, updated DOUBLE PRECISION NOT NULL,
                request TEXT NOT NULL, worker_id TEXT, lease TEXT, expires DOUBLE PRECISION,
                attempts INTEGER NOT NULL DEFAULT 0, result TEXT, message TEXT)""")
            yield PostgresDatabase(db)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return
    if VERCEL_FUNCTION:
        raise HTTPException(503, "Configure DATABASE_URL with a persistent Postgres database on Vercel.")
    path = Path(os.environ.get("SOVEREIGN_DB", str(ROOT / "data" / "workspace.db")))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, token_hash TEXT NOT NULL,
                capabilities TEXT NOT NULL DEFAULT '{}', seen REAL NOT NULL DEFAULT 0,
                revoked INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL, request TEXT NOT NULL,
                worker_id TEXT, lease TEXT, expires REAL, attempts INTEGER NOT NULL DEFAULT 0,
                result TEXT, message TEXT);
        """)
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def admin_secret() -> str:
    token = os.environ.get("SOVEREIGN_ADMIN_TOKEN", "")
    if len(token) < 24:
        raise HTTPException(503, "Set SOVEREIGN_ADMIN_TOKEN to a random secret of at least 24 characters.")
    return token


def session_value() -> str:
    return hmac.new(admin_secret().encode(), b"sovereign-browser-session-v1", hashlib.sha256).hexdigest()


def admin(request: Request):
    # Browser mutations must originate from this site; workers use bearer tokens.
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
    with database() as db:
        row = db.execute("SELECT id FROM workers WHERE token_hash=? AND revoked=0", (digest(token),)).fetchone()
    if not row:
        raise HTTPException(401, "Worker key is invalid or revoked.")
    return row["id"]


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
    if VERCEL_FUNCTION:
        admin_secret()
        with database():
            pass
    return {"status": "ok", "mode": "coordinator"}


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


def reap(db):
    now = time.time()
    db.execute("""UPDATE jobs SET state=CASE WHEN attempts>=3 THEN 'FAILED' ELSE 'QUEUED' END,
        lease=NULL, expires=NULL, updated=?, message='Worker disconnected; lease expired.'
        WHERE state='SOLVING' AND expires<?""", (now, now))


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
    with database() as db:
        reap(db)
        workers = []
        for row in db.execute("SELECT id,name,capabilities,seen FROM workers WHERE revoked=0 ORDER BY name"):
            item = dict(row)
            item["capabilities"] = json.loads(item["capabilities"])
            item["online"] = item["seen"] > time.time() - LEASE_SECONDS
            workers.append(item)
        jobs = [job_view(row) for row in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 100")]
    return {"workers": workers, "jobs": jobs}


class Pair(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@app.post("/api/workers", dependencies=[Depends(admin)], status_code=201)
def pair(body: Pair):
    token, worker_id = secrets.token_urlsafe(32), secrets.token_hex(12)
    with database() as db:
        db.execute("INSERT INTO workers(id,name,token_hash) VALUES(?,?,?)", (worker_id, body.name, digest(token)))
    return {"id": worker_id, "token": token, "name": body.name}


@app.delete("/api/workers/{worker_id}", dependencies=[Depends(admin)])
def revoke(worker_id: str):
    with database() as db:
        db.execute("UPDATE workers SET revoked=1 WHERE id=?", (worker_id,))
        db.execute("UPDATE jobs SET expires=0 WHERE worker_id=? AND state='SOLVING'", (worker_id,))
        reap(db)
    return {"revoked": True}


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
    with database() as db:
        if body.workerId and not db.execute("SELECT id FROM workers WHERE id=? AND revoked=0", (body.workerId,)).fetchone():
            raise HTTPException(422, "Choose an existing machine.")
        db.execute("INSERT INTO jobs(id,name,state,created,updated,request) VALUES(?,?,'QUEUED',?,?,?)",
                   (job_id, body.name, now, now, json.dumps(req)))
    return {"jobId": job_id, "state": "QUEUED"}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(admin)])
def job(job_id: str):
    with database() as db:
        reap(db)
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job not found.")
        return job_view(row, full=True)


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(admin)])
def cancel(job_id: str):
    with database() as db:
        changed = db.execute("UPDATE jobs SET state='CANCELLED',updated=?,lease=NULL WHERE id=? AND state IN ('QUEUED','SOLVING')",
                             (time.time(), job_id)).rowcount
    return {"cancelled": bool(changed)}


class Capabilities(BaseModel):
    hostname: str = Field(max_length=120)
    platform: str = Field(max_length=120)
    cpu_threads: int = Field(ge=1, le=65536)
    cuda_available: bool = False
    gpu_name: str = Field(default="", max_length=250)
    engine_version: str = Field(default="", max_length=250)


@app.post("/api/worker/claim")
def claim(body: Capabilities, worker_id: str = Depends(worker_auth)):
    with database() as db:
        reap(db)
        db.execute("UPDATE workers SET capabilities=?,seen=? WHERE id=?", (body.model_dump_json(), time.time(), worker_id))
        if db.execute("SELECT id FROM jobs WHERE worker_id=? AND state='SOLVING'", (worker_id,)).fetchone():
            return {"job": None}
        for row in db.execute("SELECT * FROM jobs WHERE state='QUEUED' ORDER BY created"):
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
                            req.get("workerId") in (None, w["id"]) for w in db.execute(
                                "SELECT id,capabilities FROM workers WHERE revoked=0 AND seen>?", (time.time() - LEASE_SECONDS,)))
                        if gpu_online:
                            continue
                        routing["reason"] = "Large sparse model; CPU fallback because no matching CUDA worker is online."
            routing["execution_device"] = execution
            req["routing"] = routing
            req["executionDevice"] = execution
            lease = secrets.token_urlsafe(24)
            db.execute("""UPDATE jobs SET state='SOLVING',worker_id=?,lease=?,expires=?,updated=?,
                attempts=attempts+1,message=NULL,request=? WHERE id=?""",
                       (worker_id, lease, time.time() + LEASE_SECONDS, time.time(), json.dumps(req), row["id"]))
            return {"job": {"id": row["id"], "lease": lease, "request": req}}
    return {"job": None}


class Lease(BaseModel):
    jobId: str
    lease: str


def owned(db, body, worker_id):
    row = db.execute("SELECT * FROM jobs WHERE id=?", (body.jobId,)).fetchone()
    if not row or row["worker_id"] != worker_id or row["lease"] != body.lease:
        raise HTTPException(409, "Job lease is no longer yours.")
    return row


@app.post("/api/worker/heartbeat")
def heartbeat(body: Lease, worker_id: str = Depends(worker_auth)):
    with database() as db:
        reap(db)
        row = owned(db, body, worker_id)
        if row["state"] != "SOLVING":
            raise HTTPException(409, "Job is no longer running.")
        db.execute("UPDATE jobs SET expires=? WHERE id=?", (time.time() + LEASE_SECONDS, body.jobId))
        db.execute("UPDATE workers SET seen=? WHERE id=?", (time.time(), worker_id))
    return {"continue": True}


class Completion(Lease):
    result: dict | None = None
    error: str | None = Field(default=None, max_length=4000)


@app.post("/api/worker/complete")
def complete(body: Completion, worker_id: str = Depends(worker_auth)):
    if body.result is None and not body.error:
        raise HTTPException(422, "A result or error is required.")
    try:
        encoded = json.dumps(body.result, allow_nan=False)
    except ValueError as exc:
        raise HTTPException(422, "Result contains a non-finite number.") from exc
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        raise HTTPException(413, f"Result exceeds {MAX_RESULT_BYTES // 1_000_000} MB.")
    status = body.result.get("status") if body.result else None
    verification = body.result.get("verification") if body.result else None
    verified = isinstance(verification, dict) and verification.get("is_valid") is True
    conclusive = status in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED")
    message = body.error
    if body.result and not message and (not conclusive or not verified):
        message = "Solver result did not pass verification." if not verified else f"Solver stopped with status {status}."
    with database() as db:
        reap(db)
        row = owned(db, body, worker_id)
        if row["state"] in ("COMPLETED", "FAILED"):
            return {"accepted": True}  # Idempotent retry after a lost response.
        if row["state"] != "SOLVING":
            raise HTTPException(409, "Job is no longer running.")
        db.execute("UPDATE jobs SET state=?,result=?,message=?,updated=? WHERE id=?",
                   ("FAILED" if message else "COMPLETED", encoded, message, time.time(), body.jobId))
        db.execute("UPDATE workers SET seen=? WHERE id=?", (time.time(), worker_id))
    return {"accepted": True}


@app.get("/worker.py")
def worker_download():
    return FileResponse(ROOT / "worker" / "runner.py", filename="sovereign-worker.py", media_type="text/x-python")


@app.get("/")
def index():
    return {"service": "Sovereign coordinator", "health": "/api/health"}
