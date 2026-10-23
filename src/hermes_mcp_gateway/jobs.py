"""Recoverable, bounded job receipts over Hermes' existing background terminal.

This module does not execute shell commands or manage PIDs: Hermes owns both.
Recovery credentials authorize one job, independent of a short MCP session.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from weakref import WeakValueDictionary

from mcp import types
from agent.redact import redact_sensitive_text

ACTIVE = ("starting", "running")
FINAL = ("succeeded", "failed", "cancelled", "timed_out", "unknown")
MAX_OUTPUT = 6000
RESULT_RETENTION = 7 * 24 * 3600


def job_tool_schemas() -> list[types.Tool]:
    def tool(name, description, fields, required):
        return types.Tool(name=name, description=description, input_schema={
            "type": "object", "properties": fields, "required": required,
            "additionalProperties": False,
        })

    identifier = {"type": "string", "pattern": "^job_[a-f0-9]{32}$"}
    credential = {"type": "string", "minLength": 43, "maxLength": 43}
    credentials = {"job_id": identifier, "recovery_token": credential}
    return [
        tool("job_start", "Start a resumable, bounded native Hermes background terminal job. "
             "Returns a one-time recovery credential: save it securely. Do not retry an "
             "uncertain side-effectful request without checking external state. "
             "Concurrent agents must use separate Git worktrees; each MCP session "
             "may run at most two of the four global jobs.",
             {"command": {"type": "string", "minLength": 1, "maxLength": 8192},
              "workdir": {"type": "string", "minLength": 1, "maxLength": 1024},
              "max_runtime": {"type": "integer", "minimum": 1, "maximum": 7200, "default": 1800},
              "idempotency_key": {"type": "string", "minLength": 16, "maxLength": 128}},
             ["command", "workdir"]),
        tool("job_status", "Read a resumable job's state and bounded output from any MCP session "
             "with the matching recovery credential; never assumes a missing process succeeded.",
             credentials, ["job_id", "recovery_token"]),
        tool("job_cancel", "Cancel only the resumable job authorized by its recovery credential.",
             credentials, ["job_id", "recovery_token"]),
    ]


class ResumableJobs:
    def __init__(self, hermes, directory: Path, *, allowed_roots: tuple[str, ...],
                 max_active: int = 4, max_per_session: int = 2):
        self.hermes = hermes
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.directory / "gateway-jobs.sqlite3"
        if not self.path.exists():
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        self.allowed_roots = tuple(Path(root).resolve() for root in allowed_roots)
        self.max_active = max_active
        self.max_per_session = max_per_session
        self._job_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                credential_hash TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                idempotency_hash TEXT UNIQUE,
                native_id TEXT,
                state TEXT NOT NULL,
                created REAL NOT NULL,
                deadline REAL NOT NULL,
                updated REAL NOT NULL,
                exit_code INTEGER,
                output TEXT NOT NULL DEFAULT '',
                owner_hash TEXT,
                workdir TEXT
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            for column in ("owner_hash", "workdir"):
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")

    def _lock_for(self, job_id):
        if not isinstance(job_id, str):
            raise PermissionError("invalid job credential")
        lock = self._job_locks.get(job_id)
        if lock is None:
            lock = asyncio.Lock()
            self._job_locks[job_id] = lock
        return lock

    def _connect(self):
        return sqlite3.connect(self.path, timeout=3)

    def _get(self, job_id):
        with self._connect() as db:
            return db.execute(
                "SELECT job_id, credential_hash, request_hash, idempotency_hash, "
                "native_id, state, created, deadline, updated, exit_code, output "
                "FROM jobs WHERE job_id=?", (job_id,),
            ).fetchone()

    def _authorized(self, args):
        job_id = args.get("job_id")
        secret = args.get("recovery_token")
        if not isinstance(job_id, str) or not isinstance(secret, str):
            raise PermissionError("invalid job credential")
        row = self._get(job_id)
        if not row or not hmac.compare_digest(row[1], hashlib.sha256(secret.encode()).hexdigest()):
            raise PermissionError("invalid job credential")
        return row

    def _update(self, job_id, state, *, native_id=None, exit_code=None, output=None):
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET state=?, native_id=COALESCE(?, native_id), "
                "exit_code=COALESCE(?,exit_code), output=COALESCE(?,output), updated=? WHERE job_id=?",
                (state, native_id, exit_code, output, time.time(), job_id),
            )

    def _set_deadline_for_test(self, job_id, deadline):
        with self._connect() as db:
            db.execute("UPDATE jobs SET deadline=? WHERE job_id=?", (deadline, job_id))

    @staticmethod
    def _native_owner(job_id):
        return "gateway-job-" + job_id

    async def _native(self, name, args, job_id):
        result = await self.hermes.call_direct(
            name, args, task_id=self._native_owner(job_id),
        )
        if result.is_error or not result.content:
            raise RuntimeError("native job operation failed")
        payload = json.loads(result.content[0].text)
        if not isinstance(payload, dict) or payload.get("error") or payload.get("success") is False:
            raise RuntimeError("native job operation reported an error")
        return payload

    async def call(self, name, args, *, caller_id=None):
        if name == "job_start":
            return await self.start(args, caller_id=caller_id)
        if name == "job_status":
            return await self.status(args)
        if name == "job_cancel":
            return await self.cancel(args)
        raise ValueError("unknown job tool")

    async def start(self, args, *, caller_id=None):
        command = args.get("command")
        workdir = args.get("workdir")
        duration = args.get("max_runtime", 1800)
        idempotency = args.get("idempotency_key")
        if not isinstance(command, str) or not 0 < len(command) <= 8192:
            raise ValueError("invalid command")
        if not isinstance(workdir, str) or len(workdir) > 1024:
            raise ValueError("invalid workdir")
        path = Path(workdir).resolve()
        if not path.is_dir() or not any(path == root or root in path.parents for root in self.allowed_roots):
            raise ValueError("workdir is outside allowed roots")
        if type(duration) is not int or not 1 <= duration <= 7200:
            raise ValueError("invalid max_runtime")
        if idempotency is not None and (not isinstance(idempotency, str) or
                                       not 16 <= len(idempotency) <= 128):
            raise ValueError("invalid idempotency key")
        digest = hashlib.sha256(json.dumps([command, str(path), duration]).encode()).hexdigest()
        idem_digest = hashlib.sha256(idempotency.encode()).hexdigest() if idempotency else None
        # Only the Gateway supplies a verified session owner. A recovery token,
        # not the session owner, authorizes later status/cancel calls.
        owner_hash = hashlib.sha256(caller_id.encode()).hexdigest() if caller_id else None
        now = time.time()
        # BEGIN IMMEDIATE serializes admission across multiple gateway instances,
        # rather than relying on an in-process asyncio lock.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if idem_digest:
                previous = db.execute(
                    "SELECT job_id, request_hash, state FROM jobs WHERE idempotency_hash=?",
                    (idem_digest,),
                ).fetchone()
                if previous:
                    if previous[1] != digest:
                        raise ValueError("idempotency key reused with different request")
                    return {"job_id": previous[0], "state": previous[2],
                            "recovery_required": True, "replayed": False}
            active = db.execute(
                "SELECT owner_hash, workdir FROM jobs WHERE state IN ('starting','running') "
                "OR (state='unknown' AND deadline>?)", (now,),
            ).fetchall()
            if len(active) >= self.max_active:
                raise RuntimeError("resumable job capacity reached")
            if owner_hash and sum(row[0] == owner_hash for row in active) >= self.max_per_session:
                raise RuntimeError("resumable job session capacity reached")
            for _, existing_dir in active:
                if existing_dir:
                    other = Path(existing_dir)
                    if path == other or path in other.parents or other in path.parents:
                        raise RuntimeError("resumable job workspace busy; use a separate worktree")
            job_id = "job_" + secrets.token_hex(16)
            secret = secrets.token_urlsafe(32)
            db.execute(
                "INSERT INTO jobs (job_id,credential_hash,request_hash,idempotency_hash,native_id,"
                "state,created,deadline,updated,exit_code,owner_hash,workdir) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, hashlib.sha256(secret.encode()).hexdigest(),
                 digest, idem_digest, None, "starting", now, now + duration, now, None,
                 owner_hash, str(path)),
            )
        # Native work happens outside the SQLite transaction. A missing result
        # is unknown and must never trigger an automatic retry of a mutation.
        try:
            native = await asyncio.wait_for(
                self._native("terminal", {
                    "command": command,
                    "workdir": str(path),
                    "background": True,
                    "persist_on_release": True,
                    "notify": False,
                }, job_id), timeout=12,
            )
            native_id = native.get("session_id")
            if not isinstance(native_id, str) or not native_id.startswith("proc_"):
                raise RuntimeError("native terminal returned no process handle")
            self._update(job_id, "running", native_id=native_id)
            state = "running"
        except asyncio.CancelledError:
            self._update(job_id, "unknown")
            raise
        except Exception:
            self._update(job_id, "unknown")
            state = "unknown"
        return {"job_id": job_id, "state": state, "recovery_token": secret,
                "max_runtime": duration}

    async def _adopt(self, job_id):
        # One Hermes owner is created per job; query only that owner's processes.
        try:
            native = await asyncio.wait_for(
                self._native("process_manage", {"action": "list"}, job_id), timeout=12,
            )
            for proc in native.get("processes", []):
                ident = proc.get("session_id") if isinstance(proc, dict) else None
                if isinstance(ident, str) and ident.startswith("proc_"):
                    self._update(job_id, "running", native_id=ident)
                    return ident
        except Exception:
            pass
        return None

    async def _poll(self, job_id, native_id, last_output=""):
        native = await asyncio.wait_for(
            self._native("process_manage", {"action": "poll", "session_id": native_id}, job_id),
            timeout=12,
        )
        raw_state = native.get("status")
        exit_code = native.get("exit_code")
        if raw_state in ("exited", "completed"):
            state = "succeeded" if exit_code == 0 else "failed"
        elif raw_state in ("killed", "cancelled"):
            state = "cancelled"
        elif raw_state in ("running", "starting"):
            state = "running"
        else:
            state = "unknown"
        value = native.get("output", native.get("output_preview", ""))
        output = redact_sensitive_text(value[-MAX_OUTPUT:], force=True) if isinstance(value, str) and value else last_output
        self._update(job_id, state, exit_code=exit_code, output=output)
        return state, exit_code, output

    async def status(self, args):
        async with self._lock_for(args.get("job_id")):
            row = self._authorized(args)
            job_id, _, _, _, native_id, state, created, deadline, _, exit_code, output = row
            if state in ("starting", "unknown") and not native_id:
                native_id = await self._adopt(job_id)
                if native_id:
                    state = "running"
            if state == "running" and native_id:
                try:
                    state, exit_code, output = await self._poll(job_id, native_id, output)
                except Exception:
                    state = "unknown"
                    self._update(job_id, state)
            elif state == "starting":
                state = "unknown"
                self._update(job_id, state)
            return {"job_id": job_id, "state": state, "exit_code": exit_code,
                    "created": created, "deadline": deadline, "output": output[-MAX_OUTPUT:]}

    async def cancel(self, args):
        async with self._lock_for(args.get("job_id")):
            row = self._authorized(args)
            job_id, native_id, state = row[0], row[4], row[5]
            if state == "unknown" and not native_id:
                native_id = await self._adopt(job_id)
                if native_id:
                    state = "running"
            if state in FINAL:
                return {"job_id": job_id, "state": state}
            if not native_id:
                self._update(job_id, "unknown")
                return {"job_id": job_id, "state": "unknown"}
            try:
                await asyncio.wait_for(self._native("process_manage", {
                    "action": "kill", "session_id": native_id,
                }, job_id), timeout=12)
            except Exception:
                self._update(job_id, "unknown")
                return {"job_id": job_id, "state": "unknown"}
            self._update(job_id, "cancelled")
            return {"job_id": job_id, "state": "cancelled"}

    async def maintain(self):
        now = time.time()
        with self._connect() as db:
            active = db.execute(
                "SELECT job_id, native_id, deadline, output, updated FROM jobs "
                "WHERE state IN ('running','starting','unknown')",
            ).fetchall()
            db.execute("DELETE FROM jobs WHERE state IN ('succeeded','failed','cancelled',"
                       "'timed_out','unknown') AND updated<?", (now - RESULT_RETENTION,))
        for job_id, native_id, deadline, last_output, updated in active:
            async with self._lock_for(job_id):
                # Status/cancel may have completed this job after the initial query.
                current = self._get(job_id)
                if current is None or current[5] not in ("running", "starting", "unknown"):
                    continue
                native_id, last_output = current[4], current[10]
                if not native_id and now - current[8] < 60:
                    continue
                if not native_id:
                    native_id = await self._adopt(job_id)
                if not native_id:
                    if deadline < now:
                        self._update(job_id, "unknown")
                    continue
                try:
                    state, _, _ = await self._poll(job_id, native_id, last_output)
                    if state != "running" or deadline >= now:
                        continue
                    await asyncio.wait_for(self._native("process_manage", {
                        "action": "kill", "session_id": native_id,
                    }, job_id), timeout=12)
                    self._update(job_id, "timed_out")
                except Exception:
                    self._update(job_id, "unknown")
