#!/usr/bin/env python3
"""Local operator admission gate over the Gateway's existing SQLite receipts.

No MCP endpoint, command execution, process killing, or credential access.
A timed-out drain deliberately leaves admission paused; the deploy owner must
resume it on abort. Unknown jobs require reconciliation rather than a restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import sys
import time
from pathlib import Path
from typing import cast

ACTIVE = frozenset({"starting", "running", "unknown"})
KNOWN = ACTIVE | {"prepared", "succeeded", "failed", "cancelled", "timed_out", "expired"}
OWNER_TOKEN_RE = re.compile(r"[A-Za-z0-9._~-]{32,128}\Z", re.ASCII)


def _protected_path(path: Path, *, require_existing: bool) -> Path:
    """Reject symlinks and untrusted or writable parent directories.

    Only root-owned and effective-UID-owned directories with no group/world
    write bit are trusted. This makes pathname checks stable against other-UID
    replacement; same-UID processes are outside this tool's trust boundary.
    """
    absolute = Path(os.path.abspath(os.fspath(path)))
    components = absolute.parts
    current = Path(components[0])
    trusted_owners = {0, os.geteuid()}

    for component in components[1:-1]:
        current = current / component
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("Path ancestors must not be symlinks")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("Path ancestors must be directories")
        if info.st_mode & 0o022:
            raise ValueError("Path ancestors must not be group/world writable")
        if info.st_uid not in trusted_owners:
            raise ValueError("Path ancestors must be owned by root or the current user")

    root_info = Path(components[0]).lstat()
    if (not stat.S_ISDIR(root_info.st_mode) or root_info.st_mode & 0o022
            or root_info.st_uid not in trusted_owners):
        raise ValueError("Path ancestors must be protected directories owned by root or the current user")

    try:
        target = absolute.lstat()
    except FileNotFoundError:
        if require_existing:
            raise
    else:
        if stat.S_ISLNK(target.st_mode):
            raise ValueError("Path must not be a symlink")
        if require_existing:
            if not stat.S_ISREG(target.st_mode):
                raise ValueError("The job database must be a regular file")
            if target.st_mode & 0o022:
                raise ValueError("The job database must not be group/world writable")
        elif not stat.S_ISREG(target.st_mode):
            raise ValueError("An existing snapshot destination must be a regular file")

    return absolute


def connect(path: Path, *, write: bool = False) -> sqlite3.Connection:
    path = _protected_path(path, require_existing=True)
    before = path.stat()
    db = sqlite3.connect(path.as_uri() + ("?mode=rw" if write else "?mode=ro"),
                         uri=True, timeout=3)
    try:
        _protected_path(path, require_existing=True)
        after = path.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("The job database changed during admission control")
        return db
    except BaseException:
        db.close()
        raise


def _validated_owner_hash(owner_token: str) -> str:
    if not isinstance(owner_token, str) or OWNER_TOKEN_RE.fullmatch(owner_token) is None:
        raise ValueError("Owner token must contain 32..128 safe printable ASCII characters")
    return hashlib.sha256(owner_token.encode("ascii")).hexdigest()


def set_draining(path: Path, enabled: bool, *, owner_token: str) -> None:
    owner_hash = _validated_owner_hash(owner_token)
    db = connect(path, write=True)
    try:
        with db:
            db.execute("BEGIN IMMEDIATE")
            # Also supports the first upgrade from the legacy Gateway. The
            # deploy must stop legacy ingress before using this additive gate.
            db.execute("CREATE TABLE IF NOT EXISTS gateway_settings "
                       "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            draining = db.execute(
                "SELECT value FROM gateway_settings WHERE key='draining'"
            ).fetchone()
            current_owner = db.execute(
                "SELECT value FROM gateway_settings WHERE key='drain_owner'"
            ).fetchone()
            is_paused = draining is not None and draining[0] != "0"

            if enabled:
                if is_paused and (current_owner is None or current_owner[0] != owner_hash):
                    raise ValueError("The admission pause is owned by another operator")
                db.execute("INSERT INTO gateway_settings(key,value) VALUES('draining','1') "
                           "ON CONFLICT(key) DO UPDATE SET value=excluded.value")
                db.execute("INSERT INTO gateway_settings(key,value) VALUES('drain_owner',?) "
                           "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                           (owner_hash,))
            else:
                # Resuming an already-open gate is harmless and idempotent.
                if not is_paused:
                    return
                if current_owner is None or current_owner[0] != owner_hash:
                    raise ValueError("The admission pause is owned by another operator")
                db.execute("UPDATE gateway_settings SET value='0' WHERE key='draining'")
                db.execute("DELETE FROM gateway_settings WHERE key='drain_owner'")
    finally:
        db.close()


def status(path: Path) -> dict:
    db = connect(path)
    try:
        with db:
            db.execute("BEGIN")
            has_settings = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                      "AND name='gateway_settings'").fetchone()
            value = db.execute("SELECT value FROM gateway_settings WHERE key='draining'").fetchone() if has_settings else None
            rows = db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state").fetchall()
            states = {state: count for state, count in rows if state in KNOWN}
            unrecognized = sum(count for state, count in rows if state not in KNOWN)
            return {"draining": bool(value and value[0] != "0"), "states": states,
                    "active_count": sum(states.get(state, 0) for state in ACTIVE) + unrecognized,
                    "unrecognized_count": unrecognized}
    finally:
        db.close()


def snapshot(path: Path, destination: Path) -> None:
    destination = _protected_path(destination, require_existing=False)
    source = connect(path)
    created = False
    try:
        fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        created = True
        created_info = os.fstat(fd)
        os.close(fd)
        target = sqlite3.connect(destination, timeout=3)
        try:
            current_info = _protected_path(destination, require_existing=True).stat()
            if (created_info.st_dev, created_info.st_ino) != (current_info.st_dev, current_info.st_ino):
                raise ValueError("The snapshot destination changed during creation")
            deadline = time.monotonic() + 10

            def progress(*_):
                if time.monotonic() > deadline:
                    raise ValueError("SQLite snapshot exceeded its deadline")

            source.backup(target, pages=100, progress=progress, sleep=0.05)
        finally:
            target.close()
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    finally:
        source.close()


def bounded_seconds(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= 7200:
        raise argparse.ArgumentTypeError("Seconds must be finite and within 0..7200")
    return value


def owner_token_arg(raw: str) -> str:
    try:
        _validated_owner_hash(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "drain", "resume", "snapshot"))
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--owner-token", type=owner_token_arg)
    parser.add_argument("--timeout", type=bounded_seconds, default=120.0)
    parser.add_argument("--interval", type=bounded_seconds, default=1.0)
    args = parser.parse_args(argv)
    if not 0 < args.interval <= 10:
        parser.error("Interval must be positive and at most 10 seconds")
    if args.action in ("drain", "resume") and args.owner_token is None:
        parser.error("Drain and resume require --owner-token")
    if args.action not in ("drain", "resume") and args.owner_token is not None:
        parser.error("--owner-token is only valid with drain or resume")
    try:
        if args.action == "snapshot":
            if args.destination is None:
                parser.error("Snapshot requires --destination")
            snapshot(args.database, args.destination)
            print(json.dumps({"snapshot_created": True}))
            return 0
        if args.action != "status":
            if args.owner_token is None:
                raise ValueError("Drain and resume require --owner-token")
            set_draining(args.database, args.action == "drain",
                         owner_token=cast(str, args.owner_token))
        deadline = time.monotonic() + args.timeout
        while True:
            result = status(args.database)
            if args.action != "drain" or result["active_count"] == 0:
                print(json.dumps(result, sort_keys=True))
                return 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(json.dumps({**result, "error": "drain_blocked"}, sort_keys=True))
                return 3
            time.sleep(min(args.interval, remaining))
    except (OSError, ValueError, sqlite3.Error):
        print(json.dumps({"error": "job_database_unavailable_or_incompatible"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
