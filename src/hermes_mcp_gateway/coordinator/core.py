"""Durable, fail-closed task ownership for agent clients.

GitHub is the backlog, SQLite is the authority for ownership. A lease expiry
marks work for recovery; it never silently grants ownership to another agent.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
STATES = {"preparing", "active", "review", "blocked", "done", "released"}
LABELS = {
    "active": "agent:active",
    "review": "agent:review",
    "blocked": "agent:blocked",
    "done": "agent:done",
}


def run_command(
    command: list[str], *, cwd: Path | None = None, timeout: int = 25
) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} command failed: {result.stderr.strip()[:400]}"
        )
    return result.stdout.strip()


class Coordinator:
    def __init__(
        self,
        *,
        db_path: str | Path,
        work_root: str | Path,
        repos_root: str | Path,
        owner: str,
        repositories: tuple[str, ...],
        ttl_seconds: int = 3600,
    ):
        self.db_path = Path(db_path)
        self.work_root = Path(work_root).resolve()
        self.repos_root = Path(repos_root).resolve()
        self.owner = owner
        self.repositories = frozenset(repositories)
        self.ttl = ttl_seconds
        if not self.repositories or not all(
            REPO_RE.fullmatch(r) for r in self.repositories
        ):
            raise ValueError("invalid repository allowlist")
        if not REPO_RE.fullmatch(owner):
            raise ValueError("invalid GitHub owner")
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.work_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.db_path.is_symlink():
            raise ValueError("refusing symlinked state database")
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                repo TEXT NOT NULL, issue INTEGER NOT NULL,
                agent TEXT NOT NULL, token_hash TEXT NOT NULL,
                status TEXT NOT NULL, branch TEXT NOT NULL, worktree TEXT NOT NULL,
                lease_until INTEGER NOT NULL, pr_url TEXT,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY (repo, issue)
            )""")
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_lease ON tasks(status, lease_until)"
            )
        os.chmod(self.db_path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def _repo_path(self, repo: str) -> Path:
        if repo not in self.repositories:
            raise ValueError("repository not allowlisted")
        path = self.repos_root / repo
        if (
            path.is_symlink()
            or not path.is_dir()
            or path.resolve().parent != self.repos_root
        ):
            raise ValueError("repository path not safe")
        if run_command(["git", "rev-parse", "--show-toplevel"], cwd=path) != str(path):
            raise ValueError("repository must be a top-level worktree")
        return path

    def _issue(self, issue: int) -> int:
        if type(issue) is not int or issue < 1:
            raise ValueError("invalid issue number")
        return issue

    def _github(self, repo: str, *args: str) -> Any:
        output = run_command(["gh", *args, "-R", f"{self.owner}/{repo}"], timeout=25)
        return json.loads(output) if output else None

    def _github_issue(self, repo: str, issue: int) -> dict:
        result = self._github(
            repo, "issue", "view", str(issue), "--json", "number,state,labels,title"
        )
        if not isinstance(result, dict) or result.get("state") != "OPEN":
            raise ValueError("issue is not open")
        if "agent:ready" not in {label["name"] for label in result.get("labels", [])}:
            raise ValueError("issue is not labeled agent:ready")
        return result

    @staticmethod
    def _public(row: sqlite3.Row) -> dict:
        return {k: row[k] for k in row.keys() if k != "token_hash"}  # noqa: SIM118 - sqlite Row iterates values

    def tasks_list(self, repo: str) -> dict:
        self._repo_path(repo)
        self.reconcile()
        pending = (
            self._github(
                repo,
                "issue",
                "list",
                "--state",
                "open",
                "--label",
                "agent:ready",
                "--limit",
                "100",
                "--json",
                "number,title,labels",
            )
            or []
        )
        with self._connect() as db:
            rows = {
                r["issue"]: r
                for r in db.execute("SELECT * FROM tasks WHERE repo = ?", (repo,))
            }
        return {
            "ready": [
                {"number": item["number"], "title": item["title"]}
                for item in pending
                if item["number"] not in rows
                or rows[item["number"]]["status"] == "released"
            ],
            "tracked": [self._public(row) for row in rows.values()],
        }

    def claim(self, repo: str, issue: int, agent: str) -> dict:
        source = self._repo_path(repo)
        issue = self._issue(issue)
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{2,100}", agent):
            raise ValueError("invalid agent identity")
        self._github_issue(repo, issue)
        branch = f"agent/issue-{issue}"
        directory = self.work_root / repo / f"issue-{issue}"
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT status FROM tasks WHERE repo=? AND issue=?", (repo, issue)
            ).fetchone()
            if current and current["status"] != "released":
                db.execute("ROLLBACK")
                raise ValueError("issue already claimed or requires manual recovery")
            if directory.exists() or directory.is_symlink():
                db.execute("ROLLBACK")
                raise ValueError("existing worktree requires manual recovery")
            db.execute(
                """INSERT OR REPLACE INTO tasks
                (repo,issue,agent,token_hash,status,branch,worktree,lease_until,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    repo,
                    issue,
                    agent,
                    hashlib.sha256(token.encode()).hexdigest(),
                    "preparing",
                    branch,
                    str(directory),
                    now + self.ttl,
                    now,
                    now,
                ),
            )
            db.execute("COMMIT")
        try:
            directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if (
                directory.parent.is_symlink()
                or directory.parent.resolve().parent != self.work_root
            ):
                raise ValueError("worktree parent path is not safe")
            lock_path = self.db_path.parent / f"git-{repo}.lock"
            with lock_path.open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                run_command(["git", "fetch", "origin", "main"], cwd=source, timeout=45)
                run_command(
                    [
                        "git",
                        "worktree",
                        "add",
                        "-b",
                        branch,
                        str(directory),
                        "origin/main",
                    ],
                    cwd=source,
                    timeout=45,
                )
        except Exception:
            self._set_status(repo, issue, "blocked")
            raise
        # A delayed Git fetch must not reactivate a claim that expired during setup.
        activated_at = int(time.time())
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                "UPDATE tasks SET status='active',lease_until=?,updated_at=? "
                "WHERE repo=? AND issue=? AND status='preparing' AND lease_until>?",
                (activated_at + self.ttl, activated_at, repo, issue, activated_at),
            )
            db.execute("COMMIT")
        if updated.rowcount != 1:
            self._set_status(repo, issue, "blocked")
            raise ValueError(
                "lease expired during worktree preparation; manual recovery required"
            )
        # GitHub labels are a projection; an API error must not revoke a real claim.
        synced = self._label(repo, issue, "active")
        return {
            "repo": repo,
            "issue": issue,
            "agent": agent,
            "claim_id": token,
            "branch": branch,
            "worktree": str(directory),
            "lease_until": activated_at + self.ttl,
            "github_synced": synced,
        }

    def _set_status(self, repo: str, issue: int, status: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE tasks SET status=?,updated_at=? WHERE repo=? AND issue=?",
                (status, int(time.time()), repo, issue),
            )

    def _label(self, repo: str, issue: int, status: str) -> bool:
        try:
            data = self._github(repo, "issue", "view", str(issue), "--json", "labels")
            present = {item["name"] for item in data.get("labels", [])}
            desired = LABELS[status]
            command = ["gh", "issue", "edit", str(issue), "-R", f"{self.owner}/{repo}"]
            for previous in sorted(
                (set(LABELS.values()) | {"agent:ready"}) & present - {desired}
            ):
                command.extend(["--remove-label", previous])
            if desired not in present:
                command.extend(["--add-label", desired])
            if len(command) > 6:
                run_command(command, timeout=25)
            return True
        except (RuntimeError, subprocess.TimeoutExpired, ValueError, TypeError):
            return False

    def _owned(
        self, db: sqlite3.Connection, repo: str, issue: int, token: str
    ) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM tasks WHERE repo=? AND issue=?", (repo, issue)
        ).fetchone()
        if row is None or not secrets.compare_digest(
            row["token_hash"], hashlib.sha256(token.encode()).hexdigest()
        ):
            raise PermissionError("invalid ownership capability")
        if row["status"] not in ("preparing", "active", "review") or row[
            "lease_until"
        ] <= int(time.time()):
            raise ValueError(
                "task is not active or lease expired; manual recovery required"
            )
        return row

    def status(self, repo: str, issue: int) -> dict:
        self._repo_path(repo)
        self.reconcile()
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM tasks WHERE repo=? AND issue=?",
                (repo, self._issue(issue)),
            ).fetchone()
        if not row:
            return {"repo": repo, "issue": issue, "status": "unclaimed"}
        result = self._public(row)
        result["lease_expired"] = row["lease_until"] <= int(time.time())
        return result

    def heartbeat(self, repo: str, issue: int, claim_id: str) -> dict:
        now = int(time.time())
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owned(db, repo, self._issue(issue), claim_id)
            until = now + self.ttl
            db.execute(
                "UPDATE tasks SET lease_until=?,updated_at=? WHERE repo=? AND issue=?",
                (until, now, repo, issue),
            )
            db.execute("COMMIT")
        return {"repo": repo, "issue": issue, "lease_until": until}

    def finish(self, repo: str, issue: int, claim_id: str, pr_number: int) -> dict:
        self._repo_path(repo)
        self._issue(issue)
        self._issue(pr_number)
        with self._connect() as db:
            self._owned(db, repo, issue, claim_id)
        pr = self._github(
            repo,
            "pr",
            "view",
            str(pr_number),
            "--json",
            "url,state,headRefName,baseRefName",
        )
        if (
            pr["state"] != "OPEN"
            or pr["headRefName"] != f"agent/issue-{issue}"
            or pr["baseRefName"] != "main"
        ):
            raise ValueError("PR does not match the assigned branch targeting main")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owned(db, repo, issue, claim_id)
            db.execute(
                "UPDATE tasks SET status='review',pr_url=?,updated_at=? WHERE repo=? AND issue=?",
                (pr["url"], int(time.time()), repo, issue),
            )
            db.execute("COMMIT")
        return {
            "repo": repo,
            "issue": issue,
            "status": "review",
            "pr_url": pr["url"],
            "github_synced": self._label(repo, issue, "review"),
        }

    def release(
        self, repo: str, issue: int, claim_id: str, *, blocked: bool = True
    ) -> dict:
        self._repo_path(repo)
        self._issue(issue)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._owned(db, repo, issue, claim_id)
            if not blocked and Path(row["worktree"]).exists():
                raise ValueError(
                    "worktree still exists; refusing reuse without manual recovery"
                )
            new = "blocked" if blocked else "released"
            db.execute(
                "UPDATE tasks SET status=?,updated_at=? WHERE repo=? AND issue=?",
                (new, int(time.time()), repo, issue),
            )
            db.execute("COMMIT")
        return {
            "repo": repo,
            "issue": issue,
            "status": new,
            "github_synced": self._label(repo, issue, "blocked") if blocked else False,
        }

    def reconcile(self) -> list[dict]:
        now = int(time.time())
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            stale = db.execute(
                "SELECT repo,issue FROM tasks WHERE status IN ('preparing','active') AND lease_until<=?",
                (now,),
            ).fetchall()
            for row in stale:
                db.execute(
                    "UPDATE tasks SET status='blocked',updated_at=? WHERE repo=? AND issue=?",
                    (now, row["repo"], row["issue"]),
                )
            db.execute("COMMIT")
        changes = [
            {"repo": r["repo"], "issue": r["issue"], "status": "blocked"} for r in stale
        ]
        for row in stale:
            self._label(row["repo"], row["issue"], "blocked")
        with self._connect() as db:
            reviewing = db.execute(
                "SELECT repo,issue,pr_url FROM tasks WHERE status='review' AND pr_url IS NOT NULL"
            ).fetchall()
        for row in reviewing:
            try:
                pr_number = str(int(row["pr_url"].rstrip("/").rsplit("/", 1)[-1]))
                pr = self._github(
                    row["repo"], "pr", "view", pr_number, "--json", "state,mergedAt"
                )
            except (RuntimeError, ValueError, subprocess.TimeoutExpired):
                continue
            if not pr.get("mergedAt"):
                continue
            with self._connect() as db:
                db.execute(
                    "UPDATE tasks SET status='done',updated_at=? WHERE repo=? AND issue=? AND status='review'",
                    (now, row["repo"], row["issue"]),
                )
            self._label(row["repo"], row["issue"], "done")
            changes.append(
                {"repo": row["repo"], "issue": row["issue"], "status": "done"}
            )
        return changes
