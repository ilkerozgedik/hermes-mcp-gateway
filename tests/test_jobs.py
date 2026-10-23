import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from mcp import types

from hermes_mcp_gateway.jobs import ResumableJobs, job_tool_schemas


def tool_reply(payload):
    return types.CallToolResult(content=[types.TextContent(text=json.dumps(payload))])


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.hermes = AsyncMock()
        self.hermes.call_direct.side_effect = self.dispatch
        self.outcome = {"output": "Background process started", "session_id": "proc_1234", "pid": 321}
        self.poll_outcome = {"status": "exited", "exit_code": 0, "output": "all green"}
        self.jobs = ResumableJobs(self.hermes, Path(self.temp.name), allowed_roots=(self.temp.name,))
        self.workdir = self.temp.name

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def dispatch(self, name, args, *, task_id):
        if name == "terminal":
            return tool_reply(self.outcome)
        if name == "process_manage":
            return tool_reply(self.poll_outcome)
        raise AssertionError(name)

    async def test_start_returns_recovery_secret_and_runs_existing_background_terminal(self):
        r = await self.jobs.call("job_start", {"command": "printf okay", "workdir": self.workdir})
        self.assertEqual(r["state"], "running")
        self.assertTrue(r["job_id"].startswith("job_"))
        self.assertGreaterEqual(len(r["recovery_token"]), 40)
        name, args = self.hermes.call_direct.call_args.args
        self.assertEqual(name, "terminal")
        self.assertTrue(args["background"])
        self.assertTrue(args["persist_on_release"])
        self.assertEqual(self.hermes.call_direct.call_args.kwargs["task_id"], "gateway-job-" + r["job_id"])

    async def test_new_session_can_read_and_cancel_with_credential_but_not_without(self):
        r = await self.jobs.call("job_start", {"command": "printf okay", "workdir": self.workdir})
        second = ResumableJobs(self.hermes, Path(self.temp.name), allowed_roots=(self.temp.name,))
        with self.assertRaisesRegex(PermissionError, "job credential"):
            await second.call("job_status", {"job_id": r["job_id"], "recovery_token": "x" * 43})
        status = await second.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(status["state"], "succeeded")
        self.assertEqual(status["exit_code"], 0)
        self.assertEqual(status["output"], "all green")
        c = await second.call("job_cancel", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(c["state"], "succeeded")

    async def test_duplicate_idempotency_does_not_spawn_again(self):
        args = {"command": "printf okay", "workdir": self.workdir, "idempotency_key": "nonce" * 8}
        r = await self.jobs.call("job_start", args)
        again = await self.jobs.call("job_start", args)
        self.assertEqual(r["job_id"], again["job_id"])
        self.assertNotIn("recovery_token", again)
        self.assertEqual(self.hermes.call_direct.call_count, 1)
        with self.assertRaisesRegex(ValueError, "idempotency"):
            await self.jobs.call("job_start", {**args, "command": "printf changed"})

    async def test_rejects_workdirs_outside_root_and_concurrency_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "workdir"):
            await self.jobs.call("job_start", {"command": "pwd", "workdir": "/tmp"})
        jobs = ResumableJobs(self.hermes, Path(self.temp.name), allowed_roots=(self.temp.name,), max_active=1)
        await jobs.call("job_start", {"command": "pwd", "workdir": self.workdir})
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            await jobs.call("job_start", {"command": "pwd", "workdir": self.workdir})

    async def test_unknown_on_lost_process_never_replays_command(self):
        r = await self.jobs.call("job_start", {"command": "printf okay", "workdir": self.workdir})
        self.poll_outcome = {"error": "process not found"}
        result = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(self.hermes.call_direct.call_count, 2)

    async def test_cancellation_targets_only_job_owner(self):
        r = await self.jobs.call("job_start", {"command": "printf okay", "workdir": self.workdir})
        self.poll_outcome = {"status": "killed", "termination_source": "process.kill"}
        cancelled = await self.jobs.call("job_cancel", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(cancelled["state"], "cancelled")
        _, args = self.hermes.call_direct.call_args.args
        self.assertEqual(args["action"], "kill")
        self.assertEqual(args["session_id"], "proc_1234")
        self.assertEqual(self.hermes.call_direct.call_args.kwargs["task_id"], "gateway-job-" + r["job_id"])

    async def test_expired_job_is_killed_by_reaper(self):
        r = await self.jobs.call("job_start", {"command": "printf okay", "workdir": self.workdir, "max_runtime": 1})
        self.jobs._set_deadline_for_test(r["job_id"], 0)
        async def running_then_kill(name, args, *, task_id):
            if args.get("action") == "poll":
                return tool_reply({"status": "running"})
            if args.get("action") == "kill":
                return tool_reply({"status": "killed"})
            raise AssertionError(name)
        self.hermes.call_direct.side_effect = running_then_kill
        await self.jobs.maintain()
        status = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(status["state"], "timed_out")

    async def test_result_and_output_are_retained_after_reconnect(self):
        r = await self.jobs.call("job_start", {"command": "printf okay", "workdir": self.workdir})
        first = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(first["output"], "all green")
        second = ResumableJobs(self.hermes, Path(self.temp.name), allowed_roots=(self.temp.name,))
        self.hermes.call_direct.reset_mock()
        receipt = await second.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(receipt["output"], "all green")
        self.assertEqual(receipt["state"], "succeeded")
        self.hermes.call_direct.assert_not_called()

    async def test_nonzero_exit_is_a_failure(self):
        r = await self.jobs.call("job_start", {"command": "false", "workdir": self.workdir})
        self.poll_outcome = {"status": "exited", "exit_code": 23, "output": "failed"}
        receipt = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(receipt["state"], "failed")
        self.assertEqual(receipt["exit_code"], 23)

    def test_database_contains_only_hashed_job_credentials(self):
        import sqlite3
        import os
        self.assertEqual(os.stat(self.jobs.path).st_mode & 0o777, 0o600)
        with sqlite3.connect(self.jobs.path) as db:
            self.assertTrue(db.execute("SELECT name FROM sqlite_master WHERE name='jobs'").fetchone())

    async def test_maintenance_reconciles_completed_jobs_to_free_capacity(self):
        only_one = ResumableJobs(self.hermes, Path(self.temp.name),
                                 allowed_roots=(self.temp.name,), max_active=1)
        await only_one.call("job_start", {"command": "printf first", "workdir": self.workdir})
        await only_one.maintain()
        fresh = await only_one.call("job_start", {"command": "printf second", "workdir": self.workdir})
        self.assertEqual(fresh["state"], "running")

    async def test_maintenance_preserves_success_after_deadline_instead_of_killing(self):
        r = await self.jobs.call("job_start", {"command": "printf done", "workdir": self.workdir})
        self.jobs._set_deadline_for_test(r["job_id"], 0)
        await self.jobs.maintain()
        receipt = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(receipt["state"], "succeeded")
        self.assertEqual(receipt["output"], "all green")

    async def test_ambiguous_start_never_retries_mutating_command(self):
        self.hermes.call_direct.side_effect = RuntimeError("connection dropped")
        args = {"command": "git status", "workdir": self.workdir, "idempotency_key": "recovery-nonce-" + "x" * 18}
        first = await self.jobs.call("job_start", args)
        self.assertEqual(first["state"], "unknown")
        retry = await self.jobs.call("job_start", args)
        self.assertEqual(retry["job_id"], first["job_id"])
        self.assertNotIn("recovery_token", retry)
        self.assertEqual(self.hermes.call_direct.call_count, 1)

    async def test_orphan_can_be_adopted_via_scoped_native_list(self):
        self.hermes.call_direct.side_effect = RuntimeError("connection dropped")
        r = await self.jobs.call("job_start", {"command": "printf ok", "workdir": self.workdir})
        self.assertEqual(r["state"], "unknown")
        async def retry_native(name, args, *, task_id):
            if args.get("action") == "list":
                return tool_reply({"processes": [{"session_id": "proc_recovered"}]})
            if args.get("action") == "poll":
                return tool_reply({"status": "exited", "exit_code": 0, "output": "recovered"})
            raise AssertionError(name)
        self.hermes.call_direct.side_effect = retry_native
        receipt = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(receipt["state"], "succeeded")
        self.assertEqual(receipt["output"], "recovered")

    async def test_mcp_gateway_requires_session_identity_for_start_not_for_recovery(self):
        from hermes_mcp_gateway.server import Gateway, build_catalog
        gateway = Gateway()
        gateway.jobs = AsyncMock()
        gateway.jobs.call.return_value = {"job_id": "job_demo", "state": "running", "recovery_token": "a" * 43}
        gateway.catalog = build_catalog([], [], job_tool_schemas())
        denied = await gateway.call("job_start", {"command": "printf ok", "workdir": self.workdir})
        self.assertTrue(denied.is_error)
        accepted = await gateway.call("job_start", {"command": "printf ok", "workdir": self.workdir},
                                      task_id="gateway-session-test")
        self.assertFalse(accepted.is_error)
        self.assertIsInstance(accepted.content[0], types.TextContent)
        self.assertIn("recovery_token", accepted.content[0].text)
        recovered = await gateway.call("job_status", {"job_id": "job_demo", "recovery_token": "a" * 43})
        self.assertFalse(recovered.is_error)
        await gateway.execution.close()

    def test_catalog_contains_job_tools(self):
        self.assertEqual({t.name for t in job_tool_schemas()}, {"job_start", "job_status", "job_cancel"})
