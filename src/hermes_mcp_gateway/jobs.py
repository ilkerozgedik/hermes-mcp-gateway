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
import stat
import time
from pathlib import Path
from weakref import WeakValueDictionary

from mcp import types
from agent.redact import redact_sensitive_text

ACTIVE = ("starting", "running", "unknown")
FINAL = ("succeeded", "failed", "cancelled", "timed_out", "expired")
MAX_OUTPUT = 6000
RESULT_RETENTION = 7 * 24 * 3600
MAX_PREPARED = 64
MAX_PREPARED_PER_OWNER = 8
PREPARED_TTL = 15 * 60
NATIVE_TIMEOUT = 12
_UNSET = object()


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
        tool("job_prepare", "Durably prepare and freeze a resumable job request, returning its "
             "job ID and one-time recovery credential. This does not start native work. Save "
             "the credential securely before calling job_start.",
             {"command": {"type": "string", "minLength": 1, "maxLength": 8192},
              "workdir": {"type": "string", "minLength": 1, "maxLength": 1024},
              "max_runtime": {"type": "integer", "minimum": 1, "maximum": 7200, "default": 1800},
              "idempotency_key": {"type": "string", "minLength": 16, "maxLength": 128}},
             ["command", "workdir"]),
        tool("job_start", "Start a previously prepared job exactly once using its recovery "
             "credential. Repeating a valid start returns the existing state and never relaunches "
             "native work. An uncertain response must be recovered with the same credential.",
             credentials, ["job_id", "recovery_token"]),
        tool("job_status", "Read a resumable job's state and bounded output from any MCP session "
             "with the matching recovery credential; a missing process is never assumed successful.",
             credentials, ["job_id", "recovery_token"]),
        tool("job_cancel", "Cancel only the resumable job authorized by its recovery credential.",
             credentials, ["job_id", "recovery_token"]),
    ]


class ResumableJobs:
    def __init__(self, hermes, directory: Path, *, allowed_roots: tuple[str, ...],
                 max_active: int = 4, max_per_session: int = 2,
                 max_prepared: int = MAX_PREPARED,
                 max_prepared_per_owner: int = MAX_PREPARED_PER_OWNER,
                 prepared_ttl: int = PREPARED_TTL):
        self.hermes = hermes
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.directory / "gateway-jobs.sqlite3"
        if self.path.is_symlink():
            raise ValueError("job database must not be a symlink")
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("job database must be a regular file")
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        self.allowed_roots = tuple(Path(root).resolve() for root in allowed_roots)
        self.max_active = max_active
        self.max_per_session = max_per_session
        self.max_prepared = max_prepared
        self.max_prepared_per_owner = max_prepared_per_owner
        self.prepared_ttl = prepared_ttl
        if type(max_prepared) is not int or max_prepared < 1:
            raise ValueError("invalid max_prepared")
        if type(max_prepared_per_owner) is not int or max_prepared_per_owner < 1:
            raise ValueError("invalid max_prepared_per_owner")
        if type(prepared_ttl) is not int or prepared_ttl < 1:
            raise ValueError("invalid prepared_ttl")
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
                workdir TEXT,
                prepared_command TEXT,
                prepared_max_runtime INTEGER,
                prepared_expires REAL
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            for column, declaration in (
                ("owner_hash", "TEXT"),
                ("workdir", "TEXT"),
                ("prepared_command", "TEXT"),
                ("prepared_max_runtime", "INTEGER"),
                ("prepared_expires", "REAL"),
            ):
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} {declaration}")
            db.execute("""CREATE TABLE IF NOT EXISTS gateway_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )""")

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
                "native_id, state, created, deadline, updated, exit_code, output, "
                "owner_hash, workdir, prepared_command, prepared_max_runtime, prepared_expires "
                "FROM jobs WHERE job_id=?", (job_id,),
            ).fetchone()

    @staticmethod
    def _caller_hash(caller_id):
        if not isinstance(caller_id, str) or not caller_id.strip():
            raise PermissionError("verified caller identity required")
        return hashlib.sha256(caller_id.encode()).hexdigest()

    def _authorized(self, args):
        if not isinstance(args, dict) or set(args) != {"job_id", "recovery_token"}:
            raise PermissionError("invalid job credential")
        job_id = args.get("job_id")
        secret = args.get("recovery_token")
        if not isinstance(job_id, str) or not isinstance(secret, str):
            raise PermissionError("invalid job credential")
        row = self._get(job_id)
        expected = hashlib.sha256(secret.encode()).hexdigest()
        if not row or not hmac.compare_digest(row[1], expected):
            raise PermissionError("invalid job credential")
        return row

    def _update(self, job_id, state, *, native_id=_UNSET, exit_code=_UNSET, output=_UNSET,
                clear_prepared=False):
        fields = ["state=?", "updated=?"]
        values = [state, time.time()]
        if native_id is not _UNSET:
            fields.append("native_id=?")
            values.append(native_id)
        if exit_code is not _UNSET:
            fields.append("exit_code=?")
            values.append(exit_code)
        if output is not _UNSET:
            fields.append("output=?")
            values.append(output)
        if clear_prepared:
            fields.append("prepared_command=NULL")
        values.append(job_id)
        with self._connect() as db:
            db.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id=?", values)

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
        if name == "job_prepare":
            return await self.prepare(args, caller_id=caller_id)
        if name == "job_start":
            return await self.start(args, caller_id=caller_id)
        if name == "job_status":
            return await self.status(args)
        if name == "job_cancel":
            return await self.cancel(args)
        raise ValueError("unknown job tool")

    def _validated_request(self, args):
        if not isinstance(args, dict) or set(args) - {
            "command", "workdir", "max_runtime", "idempotency_key",
        }:
            raise ValueError("invalid job_prepare arguments")
        command = args.get("command")
        workdir = args.get("workdir")
        duration = args.get("max_runtime", 1800)
        idempotency = args.get("idempotency_key")
        if not isinstance(command, str) or not 0 < len(command) <= 8192:
            raise ValueError("invalid command")
        if not isinstance(workdir, str) or not 0 < len(workdir) <= 1024:
            raise ValueError("invalid workdir")
        path = Path(workdir).resolve()
        if not path.is_dir() or not any(path == root or root in path.parents for root in self.allowed_roots):
            raise ValueError("workdir is outside allowed roots")
        if type(duration) is not int or not 1 <= duration <= 7200:
            raise ValueError("invalid max_runtime")
        if idempotency is not None and (
            not isinstance(idempotency, str) or not 16 <= len(idempotency) <= 128
        ):
            raise ValueError("invalid idempotency key")
        return command, path, duration, idempotency

    def _revalidate_workdir(self, frozen_path):
        """Ensure the frozen directory still resolves to the admitted path."""
        try:
            path = Path(frozen_path)
            resolved = path.resolve(strict=True)
            if path.is_symlink() or resolved != path or not resolved.is_dir():
                raise ValueError
            if not any(resolved == root or root in resolved.parents for root in self.allowed_roots):
                raise ValueError
            return resolved
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("prepared workdir changed or is unavailable") from exc

    async def prepare(self, args, *, caller_id=None):
        owner_hash = self._caller_hash(caller_id)
        command, path, duration, idempotency = self._validated_request(args)
        # Preserve the historical encoding so existing idempotency rows still deduplicate.
        digest = hashlib.sha256(json.dumps([command, str(path), duration]).encode()).hexdigest()
        idem_digest = hashlib.sha256(idempotency.encode()).hexdigest() if idempotency else None
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Expired prepared payloads are safe to discard: no native work is
            # permitted until the durable prepared->starting transition.
            db.execute("DELETE FROM jobs WHERE state='prepared' AND prepared_expires<=?", (now,))
            if idem_digest:
                previous = db.execute(
                    "SELECT job_id, request_hash, state FROM jobs WHERE idempotency_hash=?",
                    (idem_digest,),
                ).fetchone()
                if previous:
                    if previous[1] != digest:
                        raise ValueError("idempotency key reused with different request")
                    return {"job_id": previous[0], "state": previous[2],
                            "recovery_required": True, "replayed": True}
            prepared_count = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state='prepared'",
            ).fetchone()[0]
            if prepared_count >= self.max_prepared:
                raise RuntimeError("prepared capacity reached")
            owner_prepared_count = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state='prepared' AND owner_hash=?",
                (owner_hash,),
            ).fetchone()[0]
            if owner_prepared_count >= self.max_prepared_per_owner:
                raise RuntimeError("prepared session capacity reached")
            job_id = "job_" + secrets.token_hex(16)
            secret = secrets.token_urlsafe(32)
            db.execute(
                "INSERT INTO jobs (job_id,credential_hash,request_hash,idempotency_hash,native_id,"
                "state,created,deadline,updated,exit_code,owner_hash,workdir,prepared_command,"
                "prepared_max_runtime,prepared_expires) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, hashlib.sha256(secret.encode()).hexdigest(), digest, idem_digest,
                 None, "prepared", now, now + duration, now, None, owner_hash, str(path),
                 command, duration, now + self.prepared_ttl),
            )
        # Preparation is durable before the credential is disclosed.  The
        # command remains private in the 0600 SQLite file only until start.
        return {"job_id": job_id, "state": "prepared", "recovery_token": secret,
                "max_runtime": duration, "expires": now + self.prepared_ttl}

    def is_draining(self):
        """Return the local operator admission gate (missing row means accepting)."""
        with self._connect() as db:
            row = db.execute(
                "SELECT value FROM gateway_settings WHERE key='draining'",
            ).fetchone()
        return self._draining_value(row[0] if row else None)

    @staticmethod
    def _draining_value(value):
        if value is None:
            return False
        try:
            return int(value) != 0
        except (TypeError, ValueError):
            # Unknown settings fail closed rather than reopening job admission.
            return True

    def _start_existing(self, row):
        return {"job_id": row[0], "state": row[5],
                "max_runtime": row[14] if row[14] is not None else None,
                "replayed": True}

    async def start(self, args, *, caller_id=None):
        caller_hash = self._caller_hash(caller_id)
        if not isinstance(args, dict) or set(args) != {"job_id", "recovery_token"}:
            raise ValueError("job_start requires only job_id and recovery_token")
        job_id, secret = args.get("job_id"), args.get("recovery_token")
        if not isinstance(job_id, str) or not isinstance(secret, str):
            raise PermissionError("invalid job credential")
        now = time.time()
        expected = hashlib.sha256(secret.encode()).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT job_id, credential_hash, request_hash, idempotency_hash, native_id, "
                "state, created, deadline, updated, exit_code, output, owner_hash, workdir, "
                "prepared_command, prepared_max_runtime, prepared_expires "
                "FROM jobs WHERE job_id=?", (job_id,),
            ).fetchone()
            if not row or not hmac.compare_digest(row[1], expected):
                raise PermissionError("invalid job credential")
            if row[5] != "prepared":
                return self._start_existing(row)
            if row[15] is None or row[15] <= now:
                db.execute(
                    "UPDATE jobs SET state='expired', prepared_command=NULL, updated=? WHERE job_id=?",
                    (now, job_id),
                )
                return {"job_id": job_id, "state": "expired", "replayed": True}
            draining = db.execute(
                "SELECT value FROM gateway_settings WHERE key='draining'",
            ).fetchone()
            if self._draining_value(draining[0] if draining else None):
                raise RuntimeError("gateway is draining; fresh job starts are disabled")

            active = db.execute(
                "SELECT owner_hash, workdir FROM jobs WHERE state IN ('starting','running','unknown')",
            ).fetchall()
            if len(active) >= self.max_active:
                raise RuntimeError("resumable job capacity reached")
            if sum(owner == caller_hash for owner, _ in active) >= self.max_per_session:
                raise RuntimeError("resumable job session capacity reached")
            path = self._revalidate_workdir(row[12])
            for _, existing_dir in active:
                if existing_dir:
                    other = Path(existing_dir)
                    if path == other or path in other.parents or other in path.parents:
                        raise RuntimeError("resumable job workspace busy; use a separate worktree")
            command = row[13]
            duration = row[14]
            if not isinstance(command, str) or type(duration) is not int:
                # A prepared row without its frozen payload cannot be launched.
                raise RuntimeError("prepared job payload is unavailable")
            db.execute(
                "UPDATE jobs SET state='starting', owner_hash=?, deadline=?, updated=?, "
                "prepared_command=NULL WHERE job_id=? AND state='prepared'",
                (caller_hash, now + duration, now, job_id),
            )

        # This is deliberately after the durable state transition.  If the
        # acknowledgement is lost, recovery inspects the owned native process;
        # it never replays this side-effectful request.
        try:
            # The path can be replaced while admission is being committed.
            # Recheck immediately before dispatching native work.
            path = self._revalidate_workdir(path)
        except ValueError:
            self._update(job_id, "failed", clear_prepared=True)
            raise
        try:
            native = await asyncio.wait_for(
                self._native("terminal", {
                    "command": command,
                    "workdir": str(path),
                    "background": True,
                    "persist_on_release": True,
                    "notify": False,
                }, job_id), timeout=NATIVE_TIMEOUT,
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
        return {"job_id": job_id, "state": state, "max_runtime": duration,
                "replayed": False}

    async def _adopt(self, job_id):
        # One Hermes owner is created per job; query only that owner's processes.
        try:
            native = await asyncio.wait_for(
                self._native("process_manage", {"action": "list"}, job_id), timeout=NATIVE_TIMEOUT,
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
            timeout=NATIVE_TIMEOUT,
        )
        raw_state = native.get("status")
        exit_code = native.get("exit_code")
        explicitly_killed = (
            raw_state in ("killed", "cancelled")
            or native.get("completion_reason") == "killed"
            or native.get("termination_source") == "process.kill"
        )
        if explicitly_killed:
            state = "cancelled"
        elif raw_state in ("exited", "completed"):
            state = "succeeded" if exit_code == 0 else "failed"
        elif raw_state in ("running", "starting"):
            state = "running"
        else:
            state = "unknown"
        value = native.get("output", native.get("output_preview", ""))
        output = (redact_sensitive_text(value[-MAX_OUTPUT:], force=True)
                  if isinstance(value, str) and value else last_output)
        self._update(job_id, state, exit_code=exit_code, output=output)
        return state, exit_code, output

    async def status(self, args):
        async with self._lock_for(args.get("job_id") if isinstance(args, dict) else None):
            row = self._authorized(args)
            job_id, _, _, _, native_id, state, created, deadline, _, exit_code, output = row[:11]
            expires = row[15]
            if state == "prepared" and (expires is None or expires <= time.time()):
                self._update(job_id, "expired", clear_prepared=True)
                state = "expired"
            elif state == "prepared":
                return {"job_id": job_id, "state": state, "exit_code": exit_code,
                        "created": created, "deadline": deadline, "output": output[-MAX_OUTPUT:]}
            if state in ("starting", "unknown") and not native_id:
                native_id = await self._adopt(job_id)
                if native_id:
                    state = "running"
            if state in ("running", "unknown") and native_id:
                try:
                    state, exit_code, output = await self._poll(job_id, native_id, output)
                except Exception:
                    state = "unknown"
                    self._update(job_id, state)
            return {"job_id": job_id, "state": state, "exit_code": exit_code,
                    "created": created, "deadline": deadline, "output": output[-MAX_OUTPUT:]}

    async def cancel(self, args):
        async with self._lock_for(args.get("job_id") if isinstance(args, dict) else None):
            row = self._authorized(args)
            job_id, native_id, state = row[0], row[4], row[5]
            if state == "prepared":
                self._update(job_id, "cancelled", clear_prepared=True)
                return {"job_id": job_id, "state": "cancelled"}
            if state in FINAL:
                return {"job_id": job_id, "state": state}
            if state in ("starting", "unknown") and not native_id:
                native_id = await self._adopt(job_id)
                if native_id:
                    state = "running"
            if not native_id:
                self._update(job_id, "unknown")
                return {"job_id": job_id, "state": "unknown"}
            try:
                await asyncio.wait_for(self._native("process_manage", {
                    "action": "kill", "session_id": native_id,
                }, job_id), timeout=NATIVE_TIMEOUT)
                state, _, _ = await self._poll(job_id, native_id, row[10])
            except Exception:
                self._update(job_id, "unknown")
                return {"job_id": job_id, "state": "unknown"}
            return {"job_id": job_id, "state": state}

    async def _maintain_one(self, job_id, deadline, last_output, updated, now):
        async with self._lock_for(job_id):
            current = self._get(job_id)
            if current is None or current[5] not in ACTIVE:
                return
            native_id, last_output = current[4], current[10]
            deadline, updated = current[7], current[8]
            if not native_id and current[5] == "starting" and now - updated < 60:
                return
            if not native_id:
                native_id = await self._adopt(job_id)
            if not native_id:
                if current[5] == "starting" and now - updated >= 60:
                    self._update(job_id, "unknown")
                # Unknown without a verifiably terminal process remains an
                # admission/workspace quarantine regardless of its deadline.
                return
            try:
                state, _, _ = await self._poll(job_id, native_id, last_output)
                if state != "running" or deadline >= now:
                    return
                await asyncio.wait_for(self._native("process_manage", {
                    "action": "kill", "session_id": native_id,
                }, job_id), timeout=NATIVE_TIMEOUT)
                state, _, _ = await self._poll(job_id, native_id, last_output)
                if state == "cancelled":
                    self._update(job_id, "timed_out")
            except Exception:
                self._update(job_id, "unknown")

    async def maintain(self):
        now = time.time()
        with self._connect() as db:
            active = db.execute(
                "SELECT job_id, native_id, deadline, output, updated FROM jobs "
                "WHERE state IN ('starting','running','unknown')",
            ).fetchall()
            db.execute("DELETE FROM jobs WHERE state='prepared' AND prepared_expires<=?", (now,))
            db.execute(
                "DELETE FROM jobs WHERE state IN ('succeeded','failed','cancelled','timed_out','expired') "
                "AND updated<?", (now - RESULT_RETENTION,),
            )
        # There are at most max_active active records (including quarantined
        # unknown ones); reconcile concurrently so one slow native poll does not
        # serialize the whole reaper. Hermes may still exceed this bound.
        await asyncio.gather(*(
            self._maintain_one(job_id, deadline, output, updated, now)
            for job_id, _, deadline, output, updated in active
        ), return_exceptions=True)
