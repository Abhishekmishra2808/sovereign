"""Coordinator storage: MongoDB on Render, SQLite for local preview."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]


class Row(dict):
    def __getitem__(self, key):
        return super().get(key)


class Store:
    def reap(self, now: float) -> None:
        raise NotImplementedError

    def list_workers(self, user_id: str) -> list[Row]:
        raise NotImplementedError

    def list_jobs(self, user_id: str, limit: int = 100) -> list[Row]:
        raise NotImplementedError

    def insert_worker(self, worker_id: str, name: str, token_hash: str, user_id: str) -> None:
        raise NotImplementedError

    def revoke_worker(self, worker_id: str, now: float) -> None:
        raise NotImplementedError

    def worker_exists(self, worker_id: str, user_id: str) -> bool:
        raise NotImplementedError

    def worker_id_for_token(self, token_hash: str) -> str | None:
        raise NotImplementedError

    def worker_user_id(self, worker_id: str) -> str | None:
        raise NotImplementedError

    def update_worker_seen(self, worker_id: str, capabilities: str, seen: float) -> None:
        raise NotImplementedError

    def touch_worker(self, worker_id: str, seen: float) -> None:
        raise NotImplementedError

    def worker_solving_job(self, worker_id: str) -> Row | None:
        raise NotImplementedError

    def insert_job(self, job_id: str, name: str, now: float, request: str, user_id: str) -> None:
        raise NotImplementedError

    def get_job(self, job_id: str, user_id: str) -> Row | None:
        raise NotImplementedError

    def cancel_job(self, job_id: str, now: float, user_id: str) -> bool:
        raise NotImplementedError

    def queued_jobs(self, user_id: str) -> list[Row]:
        raise NotImplementedError

    def online_workers(self, since: float, user_id: str | None = None) -> list[Row]:
        raise NotImplementedError

    def claim_job(self, job_id: str, worker_id: str, lease: str, expires: float,
                  now: float, request: str) -> None:
        raise NotImplementedError

    def owned_job(self, job_id: str, worker_id: str, lease: str) -> Row:
        raise NotImplementedError

    def extend_job_lease(self, job_id: str, expires: float) -> None:
        raise NotImplementedError

    def force_expire_job(self, job_id: str) -> None:
        raise NotImplementedError

    def complete_job(self, job_id: str, state: str, result: str | None, message: str | None,
                     now: float) -> None:
        raise NotImplementedError

    def create_pairing(self, device_id: str, code: str, name: str, capabilities: dict,
                       requested_hours: int | None, expires: float) -> None:
        raise NotImplementedError

    def get_pairing_by_device(self, device_id: str) -> Row | None:
        raise NotImplementedError

    def get_pairing_by_code(self, code: str) -> Row | None:
        raise NotImplementedError

    def approve_pairing(self, code: str, worker_id: str, token_hash: str, duration_hours: int,
                        expires_at: float, now: float, user_id: str) -> Row | None:
        raise NotImplementedError

    def reject_pairing(self, code: str) -> None:
        raise NotImplementedError

    def expire_pairings(self, now: float) -> None:
        raise NotImplementedError

    def list_pending_pairings(self, now: float) -> list[Row]:
        raise NotImplementedError

    def save_issued_token(self, code: str, token: str) -> None:
        raise NotImplementedError


class SQLiteStore(Store):
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, token_hash TEXT NOT NULL,
                capabilities TEXT NOT NULL DEFAULT '{}', seen REAL NOT NULL DEFAULT 0,
                revoked INTEGER NOT NULL DEFAULT 0, expires_at REAL);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL, request TEXT NOT NULL,
                worker_id TEXT, lease TEXT, expires REAL, attempts INTEGER NOT NULL DEFAULT 0,
                result TEXT, message TEXT);
            CREATE TABLE IF NOT EXISTS pairings (
                device_id TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
                capabilities TEXT NOT NULL, status TEXT NOT NULL, requested_hours INTEGER,
                duration_hours INTEGER, worker_id TEXT, token_hash TEXT, issued_token TEXT,
                created REAL NOT NULL, expires REAL NOT NULL, approved_at REAL,
                connection_expires REAL);
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(workers)")}
        if "expires_at" not in columns:
            self.db.execute("ALTER TABLE workers ADD COLUMN expires_at REAL")
        if "user_id" not in columns:
            self.db.execute("ALTER TABLE workers ADD COLUMN user_id TEXT")
        job_columns = {row[1] for row in self.db.execute("PRAGMA table_info(jobs)")}
        if "user_id" not in job_columns:
            self.db.execute("ALTER TABLE jobs ADD COLUMN user_id TEXT")
        pair_columns = {row[1] for row in self.db.execute("PRAGMA table_info(pairings)")}
        if pair_columns and "issued_token" not in pair_columns:
            self.db.execute("ALTER TABLE pairings ADD COLUMN issued_token TEXT")
        if pair_columns and "user_id" not in pair_columns:
            self.db.execute("ALTER TABLE pairings ADD COLUMN user_id TEXT")

    def reap(self, now: float) -> None:
        self.db.execute("UPDATE workers SET revoked=1 WHERE expires_at IS NOT NULL AND expires_at<? AND revoked=0",
                        (now,))
        self.db.execute("""UPDATE jobs SET state=CASE WHEN attempts>=3 THEN 'FAILED' ELSE 'QUEUED' END,
            lease=NULL, expires=NULL, updated=?, message='Worker disconnected; lease expired.'
            WHERE state='SOLVING' AND expires<?""", (now, now))
        self.db.execute("UPDATE pairings SET status='expired' WHERE status='pending' AND expires<?", (now,))

    def list_workers(self, user_id: str) -> list[Row]:
        return [Row(row) for row in self.db.execute(
            "SELECT id,name,capabilities,seen,expires_at FROM workers WHERE revoked=0 AND user_id=? ORDER BY name",
            (user_id,))]

    def list_jobs(self, user_id: str, limit: int = 100) -> list[Row]:
        return [Row(row) for row in self.db.execute(
            "SELECT * FROM jobs WHERE user_id=? ORDER BY created DESC LIMIT ?", (user_id, limit))]

    def insert_worker(self, worker_id: str, name: str, token_hash: str, user_id: str) -> None:
        self.db.execute("INSERT INTO workers(id,name,token_hash,user_id) VALUES(?,?,?,?)",
                        (worker_id, name, token_hash, user_id))

    def revoke_worker(self, worker_id: str, now: float) -> None:
        self.db.execute("UPDATE workers SET revoked=1 WHERE id=?", (worker_id,))
        self.db.execute("UPDATE jobs SET expires=0 WHERE worker_id=? AND state='SOLVING'", (worker_id,))

    def worker_exists(self, worker_id: str, user_id: str) -> bool:
        return bool(self.db.execute(
            "SELECT id FROM workers WHERE id=? AND user_id=? AND revoked=0", (worker_id, user_id)).fetchone())

    def worker_id_for_token(self, token_hash: str) -> str | None:
        row = self.db.execute("SELECT id FROM workers WHERE token_hash=? AND revoked=0", (token_hash,)).fetchone()
        return row["id"] if row else None

    def worker_user_id(self, worker_id: str) -> str | None:
        row = self.db.execute("SELECT user_id FROM workers WHERE id=? AND revoked=0", (worker_id,)).fetchone()
        return row["user_id"] if row else None

    def update_worker_seen(self, worker_id: str, capabilities: str, seen: float) -> None:
        self.db.execute("UPDATE workers SET capabilities=?,seen=? WHERE id=?", (capabilities, seen, worker_id))

    def touch_worker(self, worker_id: str, seen: float) -> None:
        self.db.execute("UPDATE workers SET seen=? WHERE id=?", (seen, worker_id))

    def worker_solving_job(self, worker_id: str) -> Row | None:
        row = self.db.execute("SELECT id FROM jobs WHERE worker_id=? AND state='SOLVING'", (worker_id,)).fetchone()
        return Row(row) if row else None

    def insert_job(self, job_id: str, name: str, now: float, request: str, user_id: str) -> None:
        self.db.execute("INSERT INTO jobs(id,name,state,created,updated,request,user_id) VALUES(?,?,'QUEUED',?,?,?,?)",
                        (job_id, name, now, now, request, user_id))

    def get_job(self, job_id: str, user_id: str) -> Row | None:
        row = self.db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user_id)).fetchone()
        return Row(row) if row else None

    def cancel_job(self, job_id: str, now: float, user_id: str) -> bool:
        return bool(self.db.execute(
            """UPDATE jobs SET state='CANCELLED',updated=?,lease=NULL
               WHERE id=? AND user_id=? AND state IN ('QUEUED','SOLVING')""",
            (now, job_id, user_id)).rowcount)

    def queued_jobs(self, user_id: str) -> list[Row]:
        return [Row(row) for row in self.db.execute(
            "SELECT * FROM jobs WHERE state='QUEUED' AND user_id=? ORDER BY created", (user_id,))]

    def online_workers(self, since: float, user_id: str | None = None) -> list[Row]:
        if user_id:
            return [Row(row) for row in self.db.execute(
                "SELECT id,capabilities FROM workers WHERE revoked=0 AND seen>? AND user_id=?",
                (since, user_id))]
        return [Row(row) for row in self.db.execute(
            "SELECT id,capabilities FROM workers WHERE revoked=0 AND seen>?", (since,))]

    def claim_job(self, job_id: str, worker_id: str, lease: str, expires: float,
                  now: float, request: str) -> None:
        self.db.execute("""UPDATE jobs SET state='SOLVING',worker_id=?,lease=?,expires=?,updated=?,
            attempts=attempts+1,message=NULL,request=? WHERE id=?""",
                        (worker_id, lease, expires, now, request, job_id))

    def owned_job(self, job_id: str, worker_id: str, lease: str) -> Row:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row or row["worker_id"] != worker_id or row["lease"] != lease:
            raise ValueError("lease")
        return Row(row)

    def extend_job_lease(self, job_id: str, expires: float) -> None:
        self.db.execute("UPDATE jobs SET expires=? WHERE id=?", (expires, job_id))

    def force_expire_job(self, job_id: str) -> None:
        self.db.execute("UPDATE jobs SET expires=0 WHERE id=?", (job_id,))

    def complete_job(self, job_id: str, state: str, result: str | None, message: str | None,
                     now: float) -> None:
        self.db.execute("UPDATE jobs SET state=?,result=?,message=?,updated=? WHERE id=?",
                        (state, result, message, now, job_id))

    def create_pairing(self, device_id: str, code: str, name: str, capabilities: dict,
                       requested_hours: int | None, expires: float) -> None:
        now = time.time()
        self.db.execute("""INSERT INTO pairings(device_id,code,name,capabilities,status,requested_hours,
            created,expires) VALUES(?,?,?,?,'pending',?,?,?)""",
                        (device_id, code, name, json.dumps(capabilities), requested_hours, now, expires))

    def get_pairing_by_device(self, device_id: str) -> Row | None:
        row = self.db.execute("SELECT * FROM pairings WHERE device_id=?", (device_id,)).fetchone()
        return Row(row) if row else None

    def get_pairing_by_code(self, code: str) -> Row | None:
        row = self.db.execute("SELECT * FROM pairings WHERE code=?", (code,)).fetchone()
        return Row(row) if row else None

    def approve_pairing(self, code: str, worker_id: str, token_hash: str, duration_hours: int,
                        expires_at: float, now: float, user_id: str) -> Row | None:
        row = self.get_pairing_by_code(code)
        if not row or row["status"] != "pending" or row["expires"] < now:
            return None
        self.db.execute("""UPDATE pairings SET status='approved', worker_id=?, token_hash=?, user_id=?,
            duration_hours=?, approved_at=?, connection_expires=? WHERE code=?""",
                        (worker_id, token_hash, user_id, duration_hours, now, expires_at, code))
        self.db.execute("""INSERT INTO workers(id,name,token_hash,seen,expires_at,user_id) VALUES(?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET name=excluded.name, token_hash=excluded.token_hash,
            revoked=0, seen=excluded.seen, expires_at=excluded.expires_at, user_id=excluded.user_id""",
                        (worker_id, row["name"], token_hash, now, expires_at, user_id))
        return self.get_pairing_by_code(code)

    def reject_pairing(self, code: str) -> None:
        self.db.execute("UPDATE pairings SET status='rejected' WHERE code=?", (code,))

    def expire_pairings(self, now: float) -> None:
        self.db.execute("UPDATE pairings SET status='expired' WHERE status='pending' AND expires<?", (now,))

    def list_pending_pairings(self, now: float) -> list[Row]:
        return [Row(row) for row in self.db.execute(
            "SELECT * FROM pairings WHERE status='pending' AND expires>? ORDER BY created DESC", (now,))]

    def save_issued_token(self, code: str, token: str) -> None:
        self.db.execute("UPDATE pairings SET issued_token=? WHERE code=?", (token, code))


class MongoStore(Store):
    def __init__(self, client, database_name: str):
        self.client = client
        self.db = client[database_name]
        self.workers = self.db.workers
        self.jobs = self.db.jobs
        self.pairings = self.db.pairings
        self.workers.create_index("token_hash")
        self.workers.create_index("seen")
        self.jobs.create_index([("state", 1), ("created", 1)])
        self.pairings.create_index("code", unique=True)
        self.pairings.create_index("device_id", unique=True)

    def reap(self, now: float) -> None:
        self.workers.update_many({"expires_at": {"$lt": now}, "revoked": 0}, {"$set": {"revoked": 1}})
        self.jobs.update_many({"state": "SOLVING", "expires": {"$lt": now}, "attempts": {"$gte": 3}},
                              {"$set": {"state": "FAILED", "lease": None, "expires": None, "updated": now,
                                        "message": "Worker disconnected; lease expired."}})
        self.jobs.update_many({"state": "SOLVING", "expires": {"$lt": now}, "attempts": {"$lt": 3}},
                              {"$set": {"state": "QUEUED", "lease": None, "expires": None, "updated": now,
                                        "message": "Worker disconnected; lease expired."}})
        self.pairings.update_many({"status": "pending", "expires": {"$lt": now}},
                                  {"$set": {"status": "expired"}})

    def list_workers(self, user_id: str) -> list[Row]:
        return [Row(doc) for doc in self.workers.find({"revoked": 0, "user_id": user_id}, sort=[("name", 1)])]

    def list_jobs(self, user_id: str, limit: int = 100) -> list[Row]:
        return [Row(doc) for doc in self.jobs.find({"user_id": user_id}, sort=[("created", -1)], limit=limit)]

    def insert_worker(self, worker_id: str, name: str, token_hash: str, user_id: str) -> None:
        self.workers.insert_one({"id": worker_id, "name": name, "token_hash": token_hash, "user_id": user_id,
                                 "capabilities": "{}", "seen": 0.0, "revoked": 0, "expires_at": None})

    def revoke_worker(self, worker_id: str, now: float) -> None:
        self.workers.update_one({"id": worker_id}, {"$set": {"revoked": 1}})
        self.jobs.update_many({"worker_id": worker_id, "state": "SOLVING"}, {"$set": {"expires": 0}})

    def worker_exists(self, worker_id: str, user_id: str) -> bool:
        return self.workers.find_one({"id": worker_id, "user_id": user_id, "revoked": 0}) is not None

    def worker_id_for_token(self, token_hash: str) -> str | None:
        row = self.workers.find_one({"token_hash": token_hash, "revoked": 0})
        return row["id"] if row else None

    def worker_user_id(self, worker_id: str) -> str | None:
        row = self.workers.find_one({"id": worker_id, "revoked": 0})
        return row.get("user_id") if row else None

    def update_worker_seen(self, worker_id: str, capabilities: str, seen: float) -> None:
        self.workers.update_one({"id": worker_id}, {"$set": {"capabilities": capabilities, "seen": seen}})

    def touch_worker(self, worker_id: str, seen: float) -> None:
        self.workers.update_one({"id": worker_id}, {"$set": {"seen": seen}})

    def worker_solving_job(self, worker_id: str) -> Row | None:
        row = self.jobs.find_one({"worker_id": worker_id, "state": "SOLVING"})
        return Row(row) if row else None

    def insert_job(self, job_id: str, name: str, now: float, request: str, user_id: str) -> None:
        self.jobs.insert_one({"id": job_id, "name": name, "state": "QUEUED", "created": now, "updated": now,
                              "request": request, "user_id": user_id, "worker_id": None, "lease": None,
                              "expires": None, "attempts": 0, "result": None, "message": None})

    def get_job(self, job_id: str, user_id: str) -> Row | None:
        row = self.jobs.find_one({"id": job_id, "user_id": user_id})
        return Row(row) if row else None

    def cancel_job(self, job_id: str, now: float, user_id: str) -> bool:
        result = self.jobs.update_one({"id": job_id, "user_id": user_id, "state": {"$in": ["QUEUED", "SOLVING"]}},
                                      {"$set": {"state": "CANCELLED", "updated": now, "lease": None}})
        return result.modified_count > 0

    def queued_jobs(self, user_id: str) -> list[Row]:
        return [Row(doc) for doc in self.jobs.find({"state": "QUEUED", "user_id": user_id}, sort=[("created", 1)])]

    def online_workers(self, since: float, user_id: str | None = None) -> list[Row]:
        query: dict[str, Any] = {"revoked": 0, "seen": {"$gt": since}}
        if user_id:
            query["user_id"] = user_id
        return [Row(doc) for doc in self.workers.find(query)]

    def claim_job(self, job_id: str, worker_id: str, lease: str, expires: float,
                  now: float, request: str) -> None:
        self.jobs.update_one({"id": job_id}, {"$set": {"state": "SOLVING", "worker_id": worker_id,
                                                       "lease": lease, "expires": expires, "updated": now,
                                                       "request": request, "message": None},
                                            "$inc": {"attempts": 1}})

    def owned_job(self, job_id: str, worker_id: str, lease: str) -> Row:
        row = self.jobs.find_one({"id": job_id})
        if not row or row.get("worker_id") != worker_id or row.get("lease") != lease:
            raise ValueError("lease")
        return Row(row)

    def extend_job_lease(self, job_id: str, expires: float) -> None:
        self.jobs.update_one({"id": job_id}, {"$set": {"expires": expires}})

    def force_expire_job(self, job_id: str) -> None:
        self.jobs.update_one({"id": job_id}, {"$set": {"expires": 0}})

    def complete_job(self, job_id: str, state: str, result: str | None, message: str | None,
                     now: float) -> None:
        self.jobs.update_one({"id": job_id}, {"$set": {"state": state, "result": result,
                                                       "message": message, "updated": now}})

    def create_pairing(self, device_id: str, code: str, name: str, capabilities: dict,
                       requested_hours: int | None, expires: float) -> None:
        now = time.time()
        self.pairings.replace_one({"device_id": device_id}, {
            "device_id": device_id, "code": code, "name": name,
            "capabilities": json.dumps(capabilities), "status": "pending",
            "requested_hours": requested_hours, "duration_hours": None,
            "worker_id": None, "token_hash": None, "created": now, "expires": expires,
            "approved_at": None, "connection_expires": None,
        }, upsert=True)

    def get_pairing_by_device(self, device_id: str) -> Row | None:
        row = self.pairings.find_one({"device_id": device_id})
        return Row(row) if row else None

    def get_pairing_by_code(self, code: str) -> Row | None:
        row = self.pairings.find_one({"code": code})
        return Row(row) if row else None

    def approve_pairing(self, code: str, worker_id: str, token_hash: str, duration_hours: int,
                        expires_at: float, now: float, user_id: str) -> Row | None:
        row = self.get_pairing_by_code(code)
        if not row or row["status"] != "pending" or row["expires"] < now:
            return None
        self.pairings.update_one({"code": code}, {"$set": {
            "status": "approved", "worker_id": worker_id, "token_hash": token_hash, "user_id": user_id,
            "duration_hours": duration_hours, "approved_at": now, "connection_expires": expires_at,
        }})
        self.workers.replace_one({"id": worker_id}, {
            "id": worker_id, "name": row["name"], "token_hash": token_hash, "user_id": user_id,
            "capabilities": row["capabilities"], "seen": now, "revoked": 0, "expires_at": expires_at,
        }, upsert=True)
        return self.get_pairing_by_code(code)

    def reject_pairing(self, code: str) -> None:
        self.pairings.update_one({"code": code}, {"$set": {"status": "rejected"}})

    def expire_pairings(self, now: float) -> None:
        self.pairings.update_many({"status": "pending", "expires": {"$lt": now}},
                                  {"$set": {"status": "expired"}})

    def list_pending_pairings(self, now: float) -> list[Row]:
        return [Row(doc) for doc in self.pairings.find({"status": "pending", "expires": {"$gt": now}},
                                                        sort=[("created", -1)])]

    def save_issued_token(self, code: str, token: str) -> None:
        self.pairings.update_one({"code": code}, {"$set": {"issued_token": token}})


@contextmanager
def open_store() -> Iterator[Store]:
    vercel = os.environ.get("VERCEL") == "1"
    mongo_uri = os.environ.get("MONGODB_URI") or os.environ.get("SOVEREIGN_MONGODB_URI")
    if vercel and not mongo_uri:
        from fastapi import HTTPException
        raise HTTPException(503, "Configure MONGODB_URI with a persistent MongoDB database on Vercel.")
    if mongo_uri:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=15000)
        client.admin.command("ping")
        database_name = os.environ.get("SOVEREIGN_MONGODB_DB", "sovereign")
        try:
            yield MongoStore(client, database_name)
        finally:
            client.close()
        return
    path = Path(os.environ.get("SOVEREIGN_DB", str(ROOT / "data" / "workspace.db")))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("BEGIN IMMEDIATE")
        yield SQLiteStore(db)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
