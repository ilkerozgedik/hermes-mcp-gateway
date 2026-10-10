import concurrent.futures
import subprocess
import tempfile
import unittest
from pathlib import Path

from hermes_mcp_gateway.workspace import Workspaces, workspace_tool_schemas


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repos = self.root / "repos"
        self.repos.mkdir()
        self.project = self.repos / "demo"
        self.project.mkdir()
        self.workspaces = Workspaces(self.repos, self.root / "worktrees")
        self.git(self.project, "init", "-q")
        self.git(self.project, "config", "user.name", "Test Agent")
        self.git(self.project, "config", "user.email", "test@example.invalid")
        (self.project / "original.txt").write_text("main\n")
        self.git(self.project, "add", "original.txt")
        self.git(self.project, "commit", "-qm", "initial")

    @staticmethod
    def git(path, *args):
        return subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def test_tool_contract(self):
        self.assertEqual(
            [x.name for x in workspace_tool_schemas()],
            ["workspace_open", "workspace_close"],
        )

    def test_parallel_open_and_safe_close(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            a, b = list(pool.map(self.workspaces.open, ["demo", "demo"]))
        self.assertNotEqual(a["branch"], b["branch"])
        self.assertNotEqual(a["worktree_path"], b["worktree_path"])
        self.assertEqual(self.git(self.project, "status", "--porcelain"), "")
        self.assertTrue((self.project / "original.txt").exists())
        self.assertEqual(self.workspaces.close(a["worktree_path"])["status"], "closed")
        self.assertTrue(Path(b["worktree_path"]).is_dir())
        self.assertEqual(self.workspaces.close(b["worktree_path"])["status"], "closed")

    def test_non_git_and_invalid_repo(self):
        (self.repos / "plain").mkdir()
        self.assertEqual(self.workspaces.open("plain")["status"], "not_git")
        for name in ("../etc", "/tmp", "demo/sub", ".."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.workspaces.open(name)
        self.assertFalse((self.repos / "plain" / ".git").exists())

    def test_dirty_or_unpushed_commit_preserved(self):
        opened = self.workspaces.open("demo")
        folder = Path(opened["worktree_path"])
        (folder / "untracked.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.workspaces.close(str(folder))
        (folder / "untracked.txt").unlink()
        (folder / "original.txt").write_text("new")
        self.git(folder, "add", "original.txt")
        self.git(folder, "commit", "-qm", "private agent commit")
        with self.assertRaisesRegex(ValueError, "not retained"):
            self.workspaces.close(str(folder))
        self.assertTrue(folder.is_dir())

    def test_ignored_files_and_paths_are_protected(self):
        opened = self.workspaces.open("demo")
        folder = Path(opened["worktree_path"])
        (folder / ".gitignore").write_text("secret-cache\n")
        (folder / "secret-cache").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.workspaces.close(str(folder))
        for path in (str(self.project), str(folder / ".."), "/tmp/arbitrary"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.workspaces.close(path)

    def test_pushed_commit_can_close(self):
        opened = self.workspaces.open("demo")
        folder = Path(opened["worktree_path"])
        (folder / "original.txt").write_text("new\n")
        self.git(folder, "add", "original.txt")
        self.git(folder, "commit", "-qm", "change")
        # A remote-tracking ref conservatively represents protected/pushed work.
        self.git(self.project, "update-ref", "refs/remotes/origin/agent/test", "HEAD")
        # HEAD above is main; point the retained ref to the agent commit instead.
        head = self.git(folder, "rev-parse", "HEAD")
        self.git(self.project, "update-ref", "refs/remotes/origin/agent/test", head)
        self.assertEqual(self.workspaces.close(str(folder))["status"], "closed")
        self.assertEqual(
            self.git(
                self.project, "show-ref", "--verify", f"refs/heads/{opened['branch']}"
            )
            != "",
            True,
        )

    def test_two_independent_agents_commit_without_touching_main(self):
        opened = [self.workspaces.open("demo") for _ in range(2)]
        for index, workspace in enumerate(opened):
            folder = Path(workspace["worktree_path"])
            (folder / f"agent-{index}.txt").write_text(f"agent {index}\n")
            self.git(folder, "add", f"agent-{index}.txt")
            self.git(folder, "commit", "-qm", f"agent {index}")
        self.assertEqual(self.git(self.project, "status", "--porcelain"), "")
        self.assertEqual(
            sorted(p.name for p in self.project.iterdir() if p.is_file()),
            ["original.txt"],
        )
        for index, workspace in enumerate(opened):
            folder = Path(workspace["worktree_path"])
            self.assertEqual(
                (folder / f"agent-{index}.txt").read_text(), f"agent {index}\n"
            )
            with self.assertRaisesRegex(ValueError, "not retained"):
                self.workspaces.close(str(folder))


if __name__ == "__main__":
    unittest.main()
