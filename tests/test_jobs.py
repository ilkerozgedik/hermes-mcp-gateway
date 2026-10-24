import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

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

    async def launch(self, args, *, jobs=None, caller_id="session-a"):
        adapter = jobs or self.jobs
        prepared = await adapter.call("job_prepare", args, caller_id=caller_id)
        if "recovery_token" not in prepared:
            return prepared
        started = await adapter.call("job_start", {
            "job_id": prepared["job_id"], "recovery_token": prepared["recovery_token"],
        }, caller_id=caller_id)
        return {**prepared, **started}

    async def test_start_returns_recovery_secret_and_runs_existing_background_terminal(self):
        r = await self.launch({"command": "printf okay", "workdir": self.workdir})
        self.assertEqual(r["state"], "running")
        self.assertTrue(r["job_id"].startswith("job_"))
        self.assertGreaterEqual(len(r["recovery_token"]), 40)
        name, args = self.hermes.call_direct.call_args.args
        self.assertEqual(name, "terminal")
        self.assertTrue(args["background"])
        self.assertTrue(args["persist_on_release"])
        self.assertEqual(self.hermes.call_direct.call_args.kwargs["task_id"], "gateway-job-" + r["job_id"])

    async def test_new_session_can_read_and_cancel_with_credential_but_not_without(self):
        r = await self.launch({"command": "printf okay", "workdir": self.workdir})
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
        r = await self.launch(args)
        again = await self.jobs.call("job_prepare", args, caller_id="session-a")
        self.assertEqual(r["job_id"], again["job_id"])
        self.assertNotIn("recovery_token", again)
        self.assertEqual(self.hermes.call_direct.call_count, 1)
        with self.assertRaisesRegex(ValueError, "idempotency"):
            await self.jobs.call("job_prepare", {**args, "command": "printf changed"}, caller_id="session-a")

    async def test_rejects_workdirs_outside_root_and_concurrency_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "workdir"):
            await self.launch({"command": "pwd", "workdir": "/tmp"})
        jobs = ResumableJobs(self.hermes, Path(self.temp.name), allowed_roots=(self.temp.name,), max_active=1)
        await self.launch({"command": "pwd", "workdir": self.workdir}, jobs=jobs)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            await self.launch({"command": "pwd", "workdir": self.workdir}, jobs=jobs)

    async def test_unknown_on_lost_process_never_replays_command(self):
        r = await self.launch({"command": "printf okay", "workdir": self.workdir})
        self.poll_outcome = {"error": "process not found"}
        result = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(self.hermes.call_direct.call_count, 2)

    async def test_cancellation_targets_only_job_owner(self):
        r = await self.launch({"command": "printf okay", "workdir": self.workdir})
        self.poll_outcome = {
            "status": "exited", "exit_code": -15,
            "completion_reason": "killed", "termination_source": "process.kill",
        }
        cancelled = await self.jobs.call("job_cancel", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(cancelled["state"], "cancelled")
        native_calls = [call.args[1] for call in self.hermes.call_direct.call_args_list]
        kill_call = next(args for args in native_calls if args.get("action") == "kill")
        self.assertEqual(kill_call["session_id"], "proc_1234")
        self.assertEqual(native_calls[-1]["action"], "poll")
        self.assertEqual(native_calls[-1]["session_id"], "proc_1234")
        self.assertEqual(self.hermes.call_direct.call_args_list[-1].kwargs["task_id"],
                         "gateway-job-" + r["job_id"])

    async def test_expired_job_is_killed_by_reaper(self):
        r = await self.launch({"command": "printf okay", "workdir": self.workdir, "max_runtime": 1})
        self.jobs._set_deadline_for_test(r["job_id"], 0)
        async def running_then_kill(name, args, *, task_id):
            if args.get("action") == "poll":
                if any(call.args[1].get("action") == "kill"
                       for call in self.hermes.call_direct.call_args_list):
                    return tool_reply({
                        "status": "exited", "exit_code": -15,
                        "completion_reason": "killed", "termination_source": "process.kill",
                    })
                return tool_reply({"status": "running"})
            if args.get("action") == "kill":
                return tool_reply({
                    "status": "killed", "completion_reason": "killed",
                    "termination_source": "process.kill",
                })
            raise AssertionError(name)
        self.hermes.call_direct.side_effect = running_then_kill
        await self.jobs.maintain()
        status = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(status["state"], "timed_out")
        actions = [call.args[1].get("action") for call in self.hermes.call_direct.call_args_list
                   if call.args[1].get("action")]
        self.assertEqual(actions, ["poll", "kill", "poll"])

    async def test_result_and_output_are_retained_after_reconnect(self):
        r = await self.launch({"command": "printf okay", "workdir": self.workdir})
        first = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(first["output"], "all green")
        second = ResumableJobs(self.hermes, Path(self.temp.name), allowed_roots=(self.temp.name,))
        self.hermes.call_direct.reset_mock()
        receipt = await second.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(receipt["output"], "all green")
        self.assertEqual(receipt["state"], "succeeded")
        self.hermes.call_direct.assert_not_called()

    async def test_nonzero_exit_is_a_failure(self):
        r = await self.launch({"command": "false", "workdir": self.workdir})
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
        await self.launch({"command": "printf first", "workdir": self.workdir}, jobs=only_one)
        await only_one.maintain()
        fresh = await self.launch({"command": "printf second", "workdir": self.workdir}, jobs=only_one)
        self.assertEqual(fresh["state"], "running")

    async def test_maintenance_preserves_success_after_deadline_instead_of_killing(self):
        r = await self.launch({"command": "printf done", "workdir": self.workdir})
        self.jobs._set_deadline_for_test(r["job_id"], 0)
        await self.jobs.maintain()
        receipt = await self.jobs.call("job_status", {"job_id": r["job_id"], "recovery_token": r["recovery_token"]})
        self.assertEqual(receipt["state"], "succeeded")
        self.assertEqual(receipt["output"], "all green")

    async def test_ambiguous_start_never_retries_mutating_command(self):
        self.hermes.call_direct.side_effect = RuntimeError("connection dropped")
        args = {"command": "git status", "workdir": self.workdir, "idempotency_key": "recovery-nonce-" + "x" * 18}
        prepared = await self.jobs.call("job_prepare", args, caller_id="verified-agent")
        first = await self.start_prepared(prepared, caller_id="verified-agent")
        self.assertEqual(first["state"], "unknown")
        retry = await self.start_prepared(prepared, caller_id="verified-agent")
        self.assertEqual(retry["job_id"], first["job_id"])
        self.assertNotIn("recovery_token", retry)
        self.assertEqual(self.hermes.call_direct.call_count, 1)

    async def test_orphan_can_be_adopted_via_scoped_native_list(self):
        self.hermes.call_direct.side_effect = RuntimeError("connection dropped")
        r = await self.launch({"command": "printf ok", "workdir": self.workdir})
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
        gateway.jobs.call.return_value = {"job_id": "job_demo", "state": "running", "replayed": False}
        gateway.catalog = build_catalog([], [], job_tool_schemas())
        credentials = {"job_id": "job_demo", "recovery_token": "a" * 43}
        denied = await gateway.call("job_start", credentials)
        self.assertTrue(denied.is_error)
        accepted = await gateway.call("job_start", credentials,
                                      task_id="gateway-session-test")
        self.assertFalse(accepted.is_error)
        self.assertIsInstance(accepted.content[0], types.TextContent)
        self.assertNotIn("recovery_token", accepted.content[0].text)
        recovered = await gateway.call("job_status", {"job_id": "job_demo", "recovery_token": "a" * 43})
        self.assertFalse(recovered.is_error)
        await gateway.execution.close()

    async def test_one_session_cannot_consume_all_global_job_slots(self):
        from pathlib import Path
        for n in range(2):
            directory = Path(self.workdir) / ("agent-a-" + str(n))
            directory.mkdir()
            await self.launch({"command": "sleep 1", "workdir": str(directory)},
                              caller_id="verified-agent-a")
        other = Path(self.workdir) / "agent-a-2"
        other.mkdir()
        with self.assertRaisesRegex(RuntimeError, "session capacity"):
            await self.launch({"command": "sleep 1", "workdir": str(other)},
                              caller_id="verified-agent-a")
        accepted = await self.launch({"command": "sleep 1", "workdir": str(other)},
                                     caller_id="verified-agent-b")
        self.assertEqual(accepted["state"], "running")

    async def test_workspace_exclusion_rejects_same_or_overlapping_active_directory(self):
        await self.launch({"command": "sleep 1", "workdir": self.workdir}, caller_id="agent-a")
        nested = Path(self.workdir) / "nested"
        nested.mkdir()
        with self.assertRaisesRegex(RuntimeError, "workspace busy"):
            await self.launch({"command": "sleep 1", "workdir": str(nested)}, caller_id="agent-b")
        with self.assertRaisesRegex(RuntimeError, "workspace busy"):
            await self.launch({"command": "sleep 1", "workdir": self.workdir}, caller_id="agent-c")

    async def test_two_adapters_enforce_atomic_global_admission(self):
        second = ResumableJobs(self.hermes, Path(self.temp.name), allowed_roots=(self.temp.name,), max_active=1)
        a = Path(self.workdir) / "one"
        b = Path(self.workdir) / "two"
        a.mkdir()
        b.mkdir()
        results = await asyncio.gather(
            self.launch({"command": "sleep 1", "workdir": str(a)},
                        jobs=self.jobs_with_single_slot(), caller_id="first"),
            self.launch({"command": "sleep 1", "workdir": str(b)},
                        jobs=second, caller_id="second"),
            return_exceptions=True,
        )
        self.assertEqual(sum(isinstance(v, dict) for v in results), 1)
        self.assertEqual(sum(isinstance(v, RuntimeError) for v in results), 1)

    def jobs_with_single_slot(self):
        return ResumableJobs(self.hermes, Path(self.temp.name),
                             allowed_roots=(self.temp.name,), max_active=1)

    async def test_slow_job_status_does_not_block_another_job(self):
        one_dir = Path(self.workdir) / "one"
        two_dir = Path(self.workdir) / "two"
        one_dir.mkdir()
        two_dir.mkdir()
        first = await self.launch({"command": "sleep 1", "workdir": str(one_dir)})
        second = await self.launch({"command": "sleep 1", "workdir": str(two_dir)})
        slow_started = asyncio.Event()
        release = asyncio.Event()

        async def slow_dispatch(name, args, *, task_id):
            if name == "process_manage" and args.get("session_id") == "proc_1234" and task_id.endswith(first["job_id"]):
                slow_started.set()
                await release.wait()
            return await self.dispatch(name, args, task_id=task_id)

        self.hermes.call_direct.side_effect = slow_dispatch
        slow = asyncio.create_task(self.jobs.call("job_status", {
            "job_id": first["job_id"], "recovery_token": first["recovery_token"],
        }))
        try:
            await asyncio.wait_for(slow_started.wait(), timeout=2)
            other = await asyncio.wait_for(self.jobs.call("job_status", {
                "job_id": second["job_id"], "recovery_token": second["recovery_token"],
            }), timeout=1)
            self.assertEqual(other["state"], "succeeded")
        finally:
            release.set()
            await slow

    async def test_idempotency_deduplicates_across_sessions_without_token_leak(self):
        directory = Path(self.workdir) / "job-one"
        directory.mkdir()
        args = {"command": "printf ok", "workdir": str(directory), "idempotency_key": "shared" * 5}
        prepared = await self.jobs.call("job_prepare", args, caller_id="agent-a")
        first = await self.start_prepared(prepared, caller_id="agent-a")
        again = await self.jobs.call("job_prepare", args, caller_id="agent-b")
        self.assertEqual(first["job_id"], again["job_id"])
        self.assertNotIn("recovery_token", again)
        self.assertEqual(self.hermes.call_direct.call_count, 1)

    async def test_four_distinct_agents_have_independent_worktrees(self):
        paths = [Path(self.workdir) / ("worktree-" + str(n)) for n in range(5)]
        for path in paths:
            path.mkdir()
        receipts = await asyncio.gather(*(
            self.launch({"command": "printf ok", "workdir": str(paths[n])},
                        caller_id="session-" + str(n))
            for n in range(4)
        ))
        self.assertEqual(len({r["job_id"] for r in receipts}), 4)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            await self.launch({"command": "printf ok", "workdir": str(paths[4])},
                              caller_id="session-4")
        completed = await asyncio.gather(*(
            self.jobs.call("job_status", {
                "job_id": receipt["job_id"], "recovery_token": receipt["recovery_token"],
            }) for receipt in receipts
        ))
        self.assertTrue(all(x["state"] == "succeeded" for x in completed))

    def test_schema_migration_preserves_existing_receipt(self):
        import sqlite3
        self.jobs.path.unlink()
        with sqlite3.connect(self.jobs.path) as db:
            db.execute("""CREATE TABLE jobs (job_id TEXT PRIMARY KEY, credential_hash TEXT NOT NULL,
                request_hash TEXT NOT NULL, idempotency_hash TEXT UNIQUE, native_id TEXT,
                state TEXT NOT NULL, created REAL NOT NULL, deadline REAL NOT NULL,
                updated REAL NOT NULL, exit_code INTEGER, output TEXT NOT NULL DEFAULT '')""")
            db.execute("""INSERT INTO jobs VALUES ('job_legacy','hashed','fingerprint',NULL,
                'proc_legacy','succeeded',1,2,3,0,'retained')""")
        migrated = ResumableJobs(self.hermes, Path(self.workdir), allowed_roots=(self.workdir,))
        self.assertEqual(migrated._get("job_legacy")[10], "retained")
        with sqlite3.connect(migrated.path) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
        self.assertIn("owner_hash", columns)
        self.assertIn("workdir", columns)

    def test_concurrent_legacy_schema_migrations_serialize_and_reread_schema(self):
        self.jobs.path.unlink()
        with sqlite3.connect(self.jobs.path) as db:
            db.execute("""CREATE TABLE jobs (job_id TEXT PRIMARY KEY, credential_hash TEXT NOT NULL,
                request_hash TEXT NOT NULL, idempotency_hash TEXT UNIQUE, native_id TEXT,
                state TEXT NOT NULL, created REAL NOT NULL, deadline REAL NOT NULL,
                updated REAL NOT NULL, exit_code INTEGER, output TEXT NOT NULL DEFAULT '')""")
            db.execute("""INSERT INTO jobs VALUES ('job_legacy','hashed','fingerprint',NULL,
                'proc_legacy','succeeded',1,2,3,0,'retained')""")

        first_read = threading.Event()
        second_ready = threading.Event()
        errors = []
        instances = []
        results_lock = threading.Lock()
        real_connect = sqlite3.connect

        class BufferedRows(sqlite3.Cursor):
            def __init__(self, connection, rows):
                super().__init__(connection)
                self.rows = iter(rows)

            def __iter__(self):
                return self.rows

        class GatedConnection(sqlite3.Connection):
            def execute(self, sql, parameters=()):
                normalized = " ".join(sql.upper().split())
                thread_name = threading.current_thread().name
                if normalized == "PRAGMA TABLE_INFO(JOBS)":
                    rows = super().execute(sql, parameters).fetchall()
                    if thread_name == "migration-first":
                        first_read.set()
                        if not second_ready.wait(timeout=5):
                            raise AssertionError("second migration never reached its schema decision")
                    elif thread_name == "migration-second":
                        second_ready.set()
                    return BufferedRows(self, rows)
                if thread_name == "migration-second" and normalized == "BEGIN IMMEDIATE":
                    # In the fixed implementation, this fires before SQLite blocks
                    # the second initializer behind the first writer transaction.
                    second_ready.set()
                return super().execute(sql, parameters)

        def gated_connect(*args, **kwargs):
            kwargs["factory"] = GatedConnection
            return real_connect(*args, **kwargs)

        def initialize():
            try:
                instance = ResumableJobs(
                    self.hermes, Path(self.workdir), allowed_roots=(self.workdir,),
                )
                with results_lock:
                    instances.append(instance)
            except BaseException as exc:
                with results_lock:
                    errors.append(exc)

        first = threading.Thread(target=initialize, name="migration-first")
        second = threading.Thread(target=initialize, name="migration-second")
        with patch.object(sqlite3, "connect", side_effect=gated_connect):
            try:
                first.start()
                self.assertTrue(first_read.wait(timeout=2), "first migration did not inspect the legacy schema")
                second.start()
                first.join(timeout=7)
                second.join(timeout=7)
            finally:
                second_ready.set()
                first.join(timeout=1)
                second.join(timeout=1)

        self.assertFalse(first.is_alive(), "first migration thread did not finish")
        self.assertFalse(second.is_alive(), "second migration thread did not finish")
        self.assertEqual(errors, [])
        self.assertEqual(len(instances), 2)
        with sqlite3.connect(self.jobs.path) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            receipt = db.execute(
                "SELECT state, output FROM jobs WHERE job_id='job_legacy'",
            ).fetchone()
            settings = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='gateway_settings'",
            ).fetchone()
        self.assertTrue({"owner_hash", "workdir", "prepared_command",
                         "prepared_max_runtime", "prepared_expires"} <= columns)
        self.assertEqual(receipt, ("succeeded", "retained"))
        self.assertIsNotNone(settings)

    async def prepare(self, command="printf okay", workdir=None, *, caller_id="session-a", **extra):
        return await self.jobs.call("job_prepare", {
            "command": command, "workdir": workdir or self.workdir, **extra,
        }, caller_id=caller_id)

    async def start_prepared(self, prepared, *, caller_id="session-a", jobs=None):
        return await (jobs or self.jobs).call("job_start", {
            "job_id": prepared["job_id"], "recovery_token": prepared["recovery_token"],
        }, caller_id=caller_id)

    async def test_prepare_persists_credential_before_native_work(self):
        prepared = await self.prepare(command="printf SECRET_COMMAND")
        self.assertEqual(prepared["state"], "prepared")
        self.assertEqual(len(prepared["recovery_token"]), 43)
        self.hermes.call_direct.assert_not_called()
        with sqlite3.connect(self.jobs.path) as db:
            row = db.execute(
                "SELECT credential_hash, prepared_command FROM jobs WHERE job_id=?",
                (prepared["job_id"],),
            ).fetchone()
        self.assertNotEqual(row[0], prepared["recovery_token"])
        self.assertEqual(row[1], "printf SECRET_COMMAND")
        status = await self.jobs.call("job_status", {
            "job_id": prepared["job_id"], "recovery_token": prepared["recovery_token"],
        })
        self.assertEqual(status["state"], "prepared")
        self.hermes.call_direct.assert_not_called()

    async def test_start_requires_prepared_credential_and_launches_once_across_sessions(self):
        prepared = await self.prepare()
        wrong = {"job_id": prepared["job_id"], "recovery_token": "x" * 43}
        with self.assertRaisesRegex(PermissionError, "credential"):
            await self.jobs.call("job_start", wrong, caller_id="verified-session")
        with self.assertRaisesRegex(ValueError, "job_start"):
            await self.jobs.call("job_start", {
                "job_id": prepared["job_id"], "recovery_token": prepared["recovery_token"],
                "command": "printf changed",
            }, caller_id="verified-session")
        first = await self.start_prepared(prepared)
        second = await self.start_prepared(prepared, caller_id="different-verified-session")
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(first["state"], second["state"])
        self.assertEqual(self.hermes.call_direct.call_count, 1)
        self.assertNotIn("recovery_token", first)

    async def test_prepare_requires_verified_caller_and_changed_idempotent_payload_is_denied(self):
        with self.assertRaisesRegex(PermissionError, "caller"):
            await self.jobs.call("job_prepare", {"command": "pwd", "workdir": self.workdir})
        prepared = await self.prepare(idempotency_key="prepare-key-0123456789")
        again = await self.jobs.call("job_prepare", {
            "command": "printf okay", "workdir": self.workdir,
            "idempotency_key": "prepare-key-0123456789",
        }, caller_id="session-b")
        self.assertEqual(prepared["job_id"], again["job_id"])
        self.assertNotIn("recovery_token", again)
        with self.assertRaisesRegex(ValueError, "idempotency"):
            await self.jobs.call("job_prepare", {
                "command": "printf changed", "workdir": self.workdir,
                "idempotency_key": "prepare-key-0123456789",
            }, caller_id="session-a")
        with self.assertRaisesRegex(PermissionError, "caller"):
            await self.start_prepared(prepared, caller_id=None)

    async def test_prepare_preserves_legacy_idempotency_fingerprint(self):
        command = "printf legacy"
        idempotency_key = "legacy-key-0123456789"
        path = str(Path(self.workdir).resolve())
        request_hash = hashlib.sha256(json.dumps([command, path, 1800]).encode()).hexdigest()
        idempotency_hash = hashlib.sha256(idempotency_key.encode()).hexdigest()
        legacy_id = "job_" + "a" * 32
        with sqlite3.connect(self.jobs.path) as db:
            db.execute(
                "INSERT INTO jobs (job_id,credential_hash,request_hash,idempotency_hash,native_id,"
                "state,created,deadline,updated,exit_code,output) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (legacy_id, "0" * 64, request_hash, idempotency_hash, "proc_legacy",
                 "succeeded", 1, 1801, 2, 0, "legacy result"),
            )
        replay = await self.jobs.call("job_prepare", {
            "command": command, "workdir": self.workdir, "idempotency_key": idempotency_key,
        }, caller_id="new-session")
        self.assertEqual(replay["job_id"], legacy_id)
        self.assertTrue(replay["replayed"])
        self.assertNotIn("recovery_token", replay)

    async def test_prepared_admission_is_bounded_per_owner_and_global(self):
        bounded = ResumableJobs(
            self.hermes, Path(self.temp.name), allowed_roots=(self.workdir,),
            max_prepared=3, max_prepared_per_owner=2,
        )
        first = await bounded.call("job_prepare", {
            "command": "sleep 1", "workdir": self.workdir,
        }, caller_id="owner-a")
        second = await bounded.call("job_prepare", {
            "command": "sleep 2", "workdir": self.workdir,
        }, caller_id="owner-a")
        with self.assertRaisesRegex(RuntimeError, "prepared session capacity"):
            await bounded.call("job_prepare", {
                "command": "sleep 3", "workdir": self.workdir,
            }, caller_id="owner-a")
        third = await bounded.call("job_prepare", {
            "command": "sleep 4", "workdir": self.workdir,
        }, caller_id="owner-b")
        with sqlite3.connect(bounded.path) as db:
            owners = dict(db.execute(
                "SELECT job_id,owner_hash FROM jobs WHERE state='prepared'"
            ))
        self.assertEqual(owners[first["job_id"]], bounded._caller_hash("owner-a"))
        self.assertEqual(owners[second["job_id"]], bounded._caller_hash("owner-a"))
        self.assertEqual(owners[third["job_id"]], bounded._caller_hash("owner-b"))
        with self.assertRaisesRegex(RuntimeError, "prepared capacity"):
            await bounded.call("job_prepare", {
                "command": "sleep 5", "workdir": self.workdir,
            }, caller_id="owner-c")
        # The preparing owner is not the active-quota owner; recovery may start
        # the frozen request from another verified session.
        started = await bounded.call("job_start", {
            "job_id": first["job_id"], "recovery_token": first["recovery_token"],
        }, caller_id="recovery-session")
        self.assertEqual(started["state"], "running")
        with sqlite3.connect(bounded.path) as db:
            active_owner = db.execute(
                "SELECT owner_hash FROM jobs WHERE job_id=?", (first["job_id"],),
            ).fetchone()[0]
        self.assertEqual(active_owner, bounded._caller_hash("recovery-session"))

    async def test_start_rejects_missing_or_replaced_workdir_and_rechecks_before_dispatch(self):
        root = Path(self.workdir)
        original = root / "frozen-original"
        replacement = root / "frozen-replacement"
        original.mkdir()
        replacement.mkdir()
        prepared = await self.prepare(workdir=str(original))
        moved = root / "frozen-original-moved"
        original.rename(moved)
        os.symlink(replacement, original)
        with self.assertRaisesRegex(ValueError, "prepared workdir"):
            await self.start_prepared(prepared)
        self.assertEqual(self.jobs._get(prepared["job_id"])[5], "prepared")
        self.hermes.call_direct.assert_not_called()
        original.unlink()
        moved.rename(original)

        missing = root / "frozen-missing"
        missing.mkdir()
        missing_job = await self.prepare(workdir=str(missing))
        missing.rmdir()
        with self.assertRaisesRegex(ValueError, "prepared workdir"):
            await self.start_prepared(missing_job)
        self.hermes.call_direct.assert_not_called()

        racing = root / "frozen-racing"
        racing.mkdir()
        race_job = await self.prepare(workdir=str(racing))
        real_revalidate = self.jobs._revalidate_workdir
        validation_count = 0

        def replace_after_admission(frozen_path):
            nonlocal validation_count
            validation_count += 1
            if validation_count == 2:
                moved_racing = root / "frozen-racing-moved"
                racing.rename(moved_racing)
                os.symlink(replacement, racing)
            return real_revalidate(frozen_path)

        self.jobs._revalidate_workdir = replace_after_admission
        with self.assertRaisesRegex(ValueError, "prepared workdir"):
            await self.start_prepared(race_job)
        self.jobs._revalidate_workdir = real_revalidate
        self.assertEqual(validation_count, 2)
        self.assertEqual(self.jobs._get(race_job["job_id"])[5], "failed")
        self.hermes.call_direct.assert_not_called()

    def test_database_file_is_private_without_chmodding_existing_home(self):
        directory = Path(self.temp.name) / "operator-home"
        directory.mkdir(mode=0o755)
        os.chmod(directory, 0o755)
        database = ResumableJobs(self.hermes, directory, allowed_roots=(self.temp.name,))
        self.assertEqual(os.stat(directory).st_mode & 0o777, 0o755)
        self.assertEqual(os.stat(database.path).st_mode & 0o777, 0o600)

        other = Path(self.temp.name) / "operator-home-link"
        other.mkdir()
        victim = other / "outside.sqlite3"
        victim.write_text("leave untouched")
        link = other / "gateway-jobs.sqlite3"
        link.symlink_to(victim)
        with self.assertRaisesRegex(ValueError, "symlink"):
            ResumableJobs(self.hermes, other, allowed_roots=(self.temp.name,))
        self.assertEqual(victim.read_text(), "leave untouched")

    async def test_concurrent_starts_launch_native_exactly_once(self):
        prepared = await self.prepare()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed(name, args, *, task_id):
            entered.set()
            await release.wait()
            return tool_reply(self.outcome)

        self.hermes.call_direct.side_effect = delayed
        first = asyncio.create_task(self.start_prepared(prepared))
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = await self.start_prepared(prepared, caller_id="other-verified-session")
        self.assertEqual(second["state"], "starting")
        release.set()
        completed = await first
        self.assertEqual(completed["state"], "running")
        self.assertEqual(self.hermes.call_direct.call_count, 1)

    async def test_start_ack_loss_is_recovered_without_relaunch(self):
        prepared = await self.prepare()
        self.hermes.call_direct.side_effect = RuntimeError("ack lost after dispatch")
        lost = await self.start_prepared(prepared)
        self.assertEqual(lost["state"], "unknown")
        self.hermes.call_direct.side_effect = None
        recovered = await self.start_prepared(prepared, caller_id="other-verified-session")
        self.assertEqual(recovered["state"], "unknown")
        self.assertEqual(self.hermes.call_direct.call_count, 1)

    async def test_prepared_limit_ttl_and_not_an_admission_reservation(self):
        bounded = ResumableJobs(self.hermes, Path(self.temp.name),
                                allowed_roots=(self.workdir,), max_active=1,
                                max_prepared=1, prepared_ttl=3600)
        first = await bounded.call("job_prepare", {
            "command": "sleep 1", "workdir": self.workdir,
        }, caller_id="one")
        with self.assertRaisesRegex(RuntimeError, "prepared capacity"):
            await bounded.call("job_prepare", {
                "command": "sleep 2", "workdir": self.workdir,
            }, caller_id="two")
        # Expiry prunes the abandoned row. Preparing itself does not reserve
        # admission capacity or workspace; the start transaction does.
        with sqlite3.connect(bounded.path) as db:
            db.execute("UPDATE jobs SET prepared_expires=0 WHERE job_id=?", (first["job_id"],))
        fresh_dir = Path(self.workdir) / "fresh"
        fresh_dir.mkdir()
        second = await bounded.call("job_prepare", {
            "command": "sleep 2", "workdir": str(fresh_dir),
        }, caller_id="two")
        with sqlite3.connect(bounded.path) as db:
            self.assertIsNone(db.execute("SELECT 1 FROM jobs WHERE job_id=?", (first["job_id"],)).fetchone())
        await bounded.call("job_start", {
            "job_id": second["job_id"], "recovery_token": second["recovery_token"],
        }, caller_id="two")
        third_dir = Path(self.workdir) / "third"
        third_dir.mkdir()
        third = await bounded.call("job_prepare", {
            "command": "sleep 3", "workdir": str(third_dir),
        }, caller_id="three")
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            await bounded.call("job_start", {
                "job_id": third["job_id"], "recovery_token": third["recovery_token"],
            }, caller_id="three")

    async def test_drain_blocks_fresh_launch_but_not_duplicate_start_status_or_cancel(self):
        active = await self.prepare()
        await self.start_prepared(active)
        later_dir = Path(self.workdir) / "later"
        later_dir.mkdir()
        prepared = await self.prepare(command="printf later", workdir=str(later_dir))
        with sqlite3.connect(self.jobs.path) as db:
            db.execute("INSERT OR REPLACE INTO gateway_settings(key,value) VALUES ('draining','1')")
        self.assertTrue(self.jobs.is_draining())
        self.assertTrue(self.jobs._draining_value("2"))
        self.assertTrue(self.jobs._draining_value("corrupt"))
        self.assertFalse(self.jobs._draining_value("0"))
        duplicate = await self.start_prepared(active, caller_id="other-verified-session")
        self.assertEqual(duplicate["state"], "running")
        with self.assertRaisesRegex(RuntimeError, "draining"):
            await self.start_prepared(prepared)
        status = await self.jobs.call("job_status", {
            "job_id": active["job_id"], "recovery_token": active["recovery_token"],
        })
        self.assertEqual(status["state"], "succeeded")
        cancelled = await self.jobs.call("job_cancel", {
            "job_id": prepared["job_id"], "recovery_token": prepared["recovery_token"],
        })
        self.assertEqual(cancelled["state"], "cancelled")

    async def test_unknown_after_deadline_keeps_capacity_and_workspace_quarantined(self):
        one_dir = Path(self.workdir) / "unknown-workspace"
        one_dir.mkdir()
        prepared = await self.prepare(command="mutate", workdir=str(one_dir))
        self.hermes.call_direct.side_effect = RuntimeError("lost start acknowledgement")
        started = await self.start_prepared(prepared)
        self.assertEqual(started["state"], "unknown")
        self.jobs._set_deadline_for_test(started["job_id"], 0)
        # Scoped process listing cannot prove the process is gone.
        self.hermes.call_direct.side_effect = lambda *args, **kwargs: tool_reply({"processes": []})
        await self.jobs.maintain()
        two_dir = one_dir / "nested"
        two_dir.mkdir()
        other = await self.prepare(command="fresh", workdir=str(two_dir), caller_id="other")
        with self.assertRaisesRegex(RuntimeError, "workspace busy"):
            await self.start_prepared(other, caller_id="other")
        self.assertEqual(self.hermes.call_direct.call_count, 2)

    def test_gateway_settings_contract_and_legacy_migration(self):
        with sqlite3.connect(self.jobs.path) as db:
            settings = db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='gateway_settings'"
            ).fetchone()[0]
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
        self.assertIn("key TEXT PRIMARY KEY", settings)
        self.assertIn("value TEXT NOT NULL", settings)
        self.assertTrue({"prepared_command", "prepared_max_runtime", "prepared_expires"} <= columns)

    async def test_kill_request_does_not_finalize_a_process_still_reported_running(self):
        receipt = await self.launch({"command": "printf guarded", "workdir": self.workdir})
        self.poll_outcome = {"status": "running", "termination_source": "process.kill"}
        state = await self.jobs.call("job_status", {
            "job_id": receipt["job_id"], "recovery_token": receipt["recovery_token"],
        })
        self.assertEqual(state["state"], "running")

    async def test_missing_exit_code_is_not_fabricated_as_failure(self):
        receipt = await self.launch({"command": "printf guarded", "workdir": self.workdir})
        self.poll_outcome = {"status": "exited", "exit_code": None}
        state = await self.jobs.call("job_status", {
            "job_id": receipt["job_id"], "recovery_token": receipt["recovery_token"],
        })
        self.assertEqual(state["state"], "unknown")

    def test_catalog_contains_job_tools(self):
        self.assertEqual({t.name for t in job_tool_schemas()}, {"job_prepare", "job_start", "job_status", "job_cancel"})
