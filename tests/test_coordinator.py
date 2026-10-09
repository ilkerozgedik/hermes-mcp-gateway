"""Coordinator isolation, ownership, recovery, and MCP schema contracts."""

from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from hermes_mcp_gateway.coordinator.core import Coordinator
from hermes_mcp_gateway.coordinator.server import TOOLS, tool_schemas


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.repo_root = root / "repos"
        self.repo_root.mkdir()
        self.repo = self.repo_root / "demo"
        self.repo.mkdir()
        self.coordinator = Coordinator(
            db_path=root / "state" / "tasks.sqlite",
            work_root=root / "worktrees",
            repos_root=self.repo_root,
            owner="demo-owner",
            repositories=("demo",),
            ttl_seconds=3600,
        )
        self.commands = []
        self.lock = threading.Lock()

        def runner(command, *, cwd=None, timeout=25):
            with self.lock:
                self.commands.append(command)
            if command[0:2] == ["git", "rev-parse"]:
                return str(self.repo)
            if command[:3] == ["gh", "issue", "view"]:
                return '{"state":"OPEN","labels":[{"name":"agent:ready"}],"number":12,"title":"Demo"}'
            if command[:3] == ["gh", "issue", "list"]:
                return '[{"number":12,"title":"Demo"}]'
            if command[:3] == ["gh", "pr", "view"]:
                return '{"state":"OPEN","headRefName":"agent/issue-12","baseRefName":"main","url":"https://github.com/demo-owner/demo/pull/42"}'
            if command[:3] == ["git", "worktree", "add"]:
                Path(command[5]).mkdir(parents=True)
            return ""

        self.runner = patch("hermes_mcp_gateway.coordinator.core.run_command", runner)
        self.runner.start()
        self.addCleanup(self.runner.stop)

    def test_only_one_agent_can_claim_same_issue_concurrently(self):
        def claim(index):
            try:
                return self.coordinator.claim("demo", 12, f"agent-{index}")
            except ValueError:
                return None

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(claim, range(10)))
        success = [result for result in results if result]
        self.assertEqual(len(success), 1)
        self.assertEqual(self.coordinator.status("demo", 12)["status"], "active")
        self.assertEqual(
            len([c for c in self.commands if c[:3] == ["git", "worktree", "add"]]), 1
        )

    def test_token_required_and_state_survives_reopen(self):
        task = self.coordinator.claim("demo", 12, "hermes")
        reopened = Coordinator(
            db_path=self.coordinator.db_path,
            work_root=self.coordinator.work_root,
            repos_root=self.repo_root,
            owner="demo-owner",
            repositories=("demo",),
        )
        self.assertEqual(reopened.status("demo", 12)["status"], "active")
        self.assertNotIn("token_hash", reopened.status("demo", 12))
        with self.assertRaises(PermissionError):
            reopened.heartbeat("demo", 12, "wrong-token")
        self.assertGreater(
            reopened.heartbeat("demo", 12, task["claim_id"])["lease_until"], 0
        )
        self.assertEqual(
            reopened.finish("demo", 12, task["claim_id"], 42)["status"], "review"
        )
        self.assertEqual(
            reopened.release("demo", 12, task["claim_id"])["status"], "blocked"
        )
        with self.assertRaises(ValueError):
            reopened.claim("demo", 12, "agent-two")

    def test_expired_preparing_lease_cannot_be_reactivated(self):
        with (
            patch(
                "hermes_mcp_gateway.coordinator.core.time.time",
                side_effect=[1000, 5000, 5001],
            ),
            self.assertRaisesRegex(
                ValueError, "lease expired during worktree preparation"
            ),
        ):
            self.coordinator.claim("demo", 12, "slow-agent")
        self.assertEqual(self.coordinator.status("demo", 12)["status"], "blocked")

    def test_expiry_is_blocked_and_never_reassigned(self):
        task = self.coordinator.claim("demo", 12, "hermes")
        with self.coordinator._connect() as db:
            db.execute("UPDATE tasks SET lease_until=1")
        self.assertEqual(
            self.coordinator.reconcile(),
            [{"repo": "demo", "issue": 12, "status": "blocked"}],
        )
        with self.assertRaises(ValueError):
            self.coordinator.heartbeat("demo", 12, task["claim_id"])
        with self.assertRaises(ValueError):
            self.coordinator.claim("demo", 12, "second")

    def test_repository_not_allowlisted(self):
        with self.assertRaises(ValueError):
            self.coordinator.tasks_list("other")
        with self.assertRaises(ValueError):
            self.coordinator.claim("../demo", 12, "agent")

    def test_existing_worktree_is_not_overwritten(self):
        (self.coordinator.work_root / "demo" / "issue-12").mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.coordinator.claim("demo", 12, "hermes")
        self.assertEqual(self.coordinator.status("demo", 12)["status"], "unclaimed")

    def test_ready_issues_and_claimed_separated(self):
        self.assertEqual(
            [x["number"] for x in self.coordinator.tasks_list("demo")["ready"]], [12]
        )
        self.coordinator.claim("demo", 12, "hermes")
        self.assertEqual(self.coordinator.tasks_list("demo")["ready"], [])

    def test_pr_wrong_branch_rejected(self):
        task = self.coordinator.claim("demo", 12, "hermes")
        with (
            patch.object(
                self.coordinator,
                "_github",
                return_value={
                    "state": "OPEN",
                    "headRefName": "main",
                    "baseRefName": "main",
                    "url": "x",
                },
            ),
            self.assertRaises(ValueError),
        ):
            self.coordinator.finish("demo", 12, task["claim_id"], 42)

    def test_merged_pull_request_updates_task(self):
        task = self.coordinator.claim("demo", 12, "hermes")
        self.coordinator.finish("demo", 12, task["claim_id"], 42)
        with patch.object(
            self.coordinator, "_github", return_value={"mergedAt": "now", "labels": []}
        ):
            self.assertEqual(self.coordinator.status("demo", 12)["status"], "done")

    def test_gateway_preserves_claim_id(self):
        import json

        from mcp import types

        from hermes_mcp_gateway.server import normalize_result

        claim = self.coordinator.claim("demo", 12, "hermes")
        encoded = json.dumps(claim)
        result = normalize_result(
            types.CallToolResult(
                content=[types.TextContent(text=encoded)], structured_content=claim
            ),
            limit=65536,
        )
        text_body = next(
            x.text for x in result.content if isinstance(x, types.TextContent)
        )
        self.assertEqual(json.loads(text_body)["claim_id"], claim["claim_id"])
        self.assertEqual(result.structured_content["claim_id"], claim["claim_id"])

    def test_tool_contract(self):
        self.assertEqual({x.name for x in tool_schemas()}, set(TOOLS))
        for schema in tool_schemas():
            self.assertFalse(schema.input_schema["additionalProperties"])
            self.assertEqual(
                set(schema.input_schema["required"]), set(TOOLS[schema.name])
            )


if __name__ == "__main__":
    unittest.main()
