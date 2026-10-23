#!/usr/bin/env python3
"""Local operator admission gate over the Gateway's existing SQLite receipts.

No MCP endpoint, command execution, process killing, or credential access.
A timed-out drain deliberately leaves admission paused; the deploy owner must
resume it on abort. Unknown jobs require reconciliation rather than a restart.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import stat
import sys
import time

ACTIVE = frozenset({"starting", "running", "unknown"})
KNOWN = ACTIVE | {"prepared", "succeeded", "failed", "cancelled", "timed_out", "expired"}


def connect(path: Path, *, write: bool = False) -> sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise ValueError("An existing, non-symlink job database is required")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_mode & 0o022:
        raise ValueError("The job database must not be group/world writable")
    db = sqlite3.connect(path.absolute().as_uri() + ("?mode=rw" if write else "?mode=ro"),
                         uri=True, timeout=3)
    after = path.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        db.close()
        raise ValueError("The job database changed during admission control")
    return db


def set_draining(path: Path, enabled: bool) -> None:
    db = connect(path, write=True)
    try:
        with db:
            db.execute("BEGIN IMMEDIATE")
            # Also supports the first upgrade from the legacy Gateway. The
            # deploy must stop legacy ingress before using this additive gate.
            db.execute("CREATE TABLE IF NOT EXISTS gateway_settings "
                       "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("INSERT INTO gateway_settings(key,value) VALUES('draining',?) "
                       "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       ("1" if enabled else "0",))
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


def bounded_seconds(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= 7200:
        raise argparse.ArgumentTypeError("Seconds must be finite and within 0..7200")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "drain", "resume"))
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--timeout", type=bounded_seconds, default=120.0)
    parser.add_argument("--interval", type=bounded_seconds, default=1.0)
    args = parser.parse_args(argv)
    if not 0 < args.interval <= 10:
        parser.error("Interval must be positive and at most 10 seconds")
    try:
        if args.action != "status":
            set_draining(args.database, args.action == "drain")
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
