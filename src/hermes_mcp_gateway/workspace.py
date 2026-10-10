"""Minimal native Git worktree lifecycle for ChatGPT coding agents."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from uuid import uuid4

from mcp import types

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
_ID = re.compile(r"[0-9a-f]{12}\Z")


def workspace_tool_schemas() -> list[types.Tool]:
    return [
        types.Tool(
            name="workspace_open",
            description="Create a unique native Git branch and isolated worktree for a repository under the work root. Non-Git directories return not_git.",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository directory name under /home/hermes/work, e.g. peacify",
                    }
                },
                "required": ["repo"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="workspace_close",
            description="Safely remove a clean managed worktree only when its HEAD is retained by another branch or remote-tracking ref; never force or delete the branch.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Exact worktree_path returned by workspace_open",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
    ]


class Workspaces:
    def __init__(self, repos_root: str | Path, worktrees_root: str | Path):
        self.repos_root = Path(repos_root).resolve()
        self.worktrees_root = Path(worktrees_root).resolve()

    @staticmethod
    def _git(
        cwd: Path, *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        if check and result.returncode:
            raise ValueError(
                f"git {args[0]} failed: {(result.stderr or result.stdout).strip()[:400]}"
            )
        return result

    def _repo(self, name: str) -> Path:
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or name in {".", ".."}
        ):
            raise ValueError("invalid repository name")
        repo = (self.repos_root / name).resolve()
        if repo.parent != self.repos_root or not repo.is_dir():
            raise ValueError("repository path not allowed or missing")
        return repo

    def open(self, repo: str) -> dict[str, str]:
        source = self._repo(repo)
        top = self._git(source, "rev-parse", "--show-toplevel", check=False)
        if top.returncode:
            return {"status": "not_git", "repo": repo}
        if Path(top.stdout.strip()).resolve() != source:
            raise ValueError("repository must be a top-level Git checkout")
        uid = uuid4().hex[:12]
        path = self.worktrees_root / repo / uid
        branch = f"agent/{uid}"
        # Git creates the parent worktree directory; only the owned path is mutated.
        self._git(source, "worktree", "add", "-b", branch, str(path), "HEAD")
        return {
            "status": "opened",
            "repo": repo,
            "worktree_path": str(path),
            "branch": branch,
            "base_sha": self._git(path, "rev-parse", "HEAD").stdout.strip(),
        }

    def close(self, path: str) -> dict[str, str]:
        if not isinstance(path, str):
            raise TypeError("invalid workspace path")
        candidate = Path(path)
        if not candidate.is_absolute() or candidate.is_symlink():
            raise ValueError("workspace path must be absolute and not a symlink")
        worktree = candidate.resolve()
        try:
            relative = worktree.relative_to(self.worktrees_root)
        except ValueError as exc:
            raise ValueError("workspace is outside managed worktrees") from exc
        if len(relative.parts) != 2 or not _ID.fullmatch(relative.parts[1]):
            raise ValueError("workspace is not a managed agent worktree")
        repo_name, uid = relative.parts
        source = self._repo(repo_name)
        if worktree != candidate or not worktree.is_dir():
            raise ValueError("workspace path has changed or is missing")
        known = self._git(source, "worktree", "list", "--porcelain").stdout.splitlines()
        if f"worktree {worktree}" not in known:
            raise ValueError("workspace is not registered with Git")
        branch = self._git(worktree, "symbolic-ref", "--short", "HEAD").stdout.strip()
        if branch != f"agent/{uid}":
            raise ValueError("workspace branch does not match")
        if self._git(
            worktree, "status", "--porcelain", "--untracked-files=all", "--ignored"
        ).stdout.strip():
            raise ValueError("workspace contains modified, untracked, or ignored files")
        head = self._git(worktree, "rev-parse", "HEAD").stdout.strip()
        refs = self._git(
            source,
            "for-each-ref",
            "--format=%(refname)",
            f"--contains={head}",
            "refs/heads",
            "refs/remotes",
        ).stdout.splitlines()
        if not any(
            ref != f"refs/heads/{branch}" and ref != "refs/remotes/origin/HEAD"
            for ref in refs
        ):
            raise ValueError(
                "workspace HEAD is not retained by another branch or remote-tracking ref"
            )
        self._git(source, "worktree", "remove", str(worktree))
        return {"status": "closed", "worktree_path": str(worktree), "branch": branch}
