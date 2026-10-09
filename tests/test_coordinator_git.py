"""Real Git worktrees on a disposable local repository (no GitHub writes)."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_mcp_gateway.coordinator.core import Coordinator


def git(path, *args):
    proc = subprocess.run(
        ["git", *args], cwd=path, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


class WorktreeIntegrationTests(unittest.TestCase):
    def test_two_issues_have_isolated_commits_and_branches(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            bare = root / "remote.git"
            bare.mkdir()
            git(bare, "init", "--bare", "-q")
            sources = root / "sources"
            sources.mkdir()
            repo = sources / "demo"
            repo.mkdir()
            git(repo, "init", "-q", "-b", "main")
            git(repo, "config", "user.email", "agent@example.invalid")
            git(repo, "config", "user.name", "Test Agent")
            (repo / "README.md").write_text("base\n")
            git(repo, "add", ".")
            git(repo, "commit", "-qm", "initial")
            git(repo, "remote", "add", "origin", str(bare))
            git(repo, "push", "-q", "-u", "origin", "main")
            coordinator = Coordinator(
                db_path=root / "state" / "db.sqlite",
                work_root=root / "worktrees",
                repos_root=sources,
                owner="test-owner",
                repositories=("demo",),
            )
            with (
                patch.object(
                    coordinator, "_github_issue", return_value={"state": "OPEN"}
                ),
                patch.object(coordinator, "_label", return_value=True),
            ):
                first = coordinator.claim("demo", 101, "hermes")
                second = coordinator.claim("demo", 102, "chatgpt")
            first_dir, second_dir = Path(first["worktree"]), Path(second["worktree"])
            self.assertNotEqual(first_dir, second_dir)
            self.assertEqual(
                git(first_dir, "branch", "--show-current"), "agent/issue-101"
            )
            self.assertEqual(
                git(second_dir, "branch", "--show-current"), "agent/issue-102"
            )
            (first_dir / "backend.txt").write_text("first agent\n")
            git(first_dir, "add", ".")
            git(
                first_dir,
                "-c",
                "user.name=Agent",
                "-c",
                "user.email=agent@example.invalid",
                "commit",
                "-qm",
                "backend",
            )
            self.assertFalse((second_dir / "backend.txt").exists())
            self.assertFalse((repo / "backend.txt").exists())
            self.assertEqual(coordinator.status("demo", 101)["status"], "active")
            self.assertEqual(coordinator.status("demo", 102)["status"], "active")


if __name__ == "__main__":
    unittest.main()
