import asyncio
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from mcp import types

from hermes_mcp_gateway.config import SAMCHON_GRAPH_TOOL, GatewayConfig
from hermes_mcp_gateway.upstreams import SamchonGraphClient


class TaskBoundContext:
    def __init__(self, value, *, on_exit=None):
        self.value = value
        self.owner = None
        self.entered = asyncio.Event()
        self.exited = asyncio.Event()
        self.exit_owner = None
        self.on_exit = on_exit

    async def __aenter__(self):
        self.owner = asyncio.current_task()
        self.entered.set()
        return self.value

    async def __aexit__(self, *_exc):
        if asyncio.current_task() is not self.owner:
            raise RuntimeError("MCP context exited from a different task")
        self.exit_owner = asyncio.current_task()
        if self.on_exit is not None:
            await self.on_exit()
        self.exited.set()


class FakeGraphSession:
    def __init__(self, root, *, block_call=False):
        self.root = root
        self.block_call = block_call
        self.call_started = asyncio.Event()
        self.call_finished = asyncio.Event()
        self.release_call = asyncio.Event()
        self.call_count = 0

    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                types.Tool(
                    name=SAMCHON_GRAPH_TOOL,
                    description="graph",
                    input_schema={"type": "object", "properties": {}},
                )
            ]
        )

    async def call_tool(self, _name, _arguments, **_kwargs):
        self.call_count += 1
        self.call_started.set()
        try:
            if self.block_call:
                await self.release_call.wait()
            return types.CallToolResult(content=[types.TextContent(text=self.root)])
        finally:
            self.call_finished.set()


class GraphHarness:
    def __init__(self, roots, *, block_root=None, slow_close_root=None):
        self.sessions = {
            root: FakeGraphSession(root, block_call=root == block_root) for root in roots
        }
        self.contexts = {root: [] for root in roots}
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.slow_close_root = slow_close_root

    def stdio_client(self, params):
        root = params.cwd
        session = self.sessions[root]

        async def on_exit():
            if root == self.slow_close_root:
                self.close_started.set()
                await self.release_close.wait()

        context = TaskBoundContext((session, object()), on_exit=on_exit)
        self.contexts[root].append(context)
        return context

    def client_session(self, read_stream, _write_stream):
        context = TaskBoundContext(read_stream)
        self.contexts[read_stream.root].append(context)
        return context


class SamchonGraphLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_logs_never_echo_upstream_error_payloads(self):
        client = SamchonGraphClient(GatewayConfig())
        async def failed_cleanup():
            raise RuntimeError("private-graph-error-fixture")
        with self.assertLogs("hermes_mcp_gateway.upstreams", level="ERROR") as logs:
            task = client._track_cleanup_task(asyncio.create_task(failed_cleanup()))
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)
        self.assertNotIn("private-graph-error-fixture", "\n".join(logs.output))
        self.assertIn("RuntimeError", "\n".join(logs.output))

    def _client(self, root, *, max_sessions=1):
        return SamchonGraphClient(
            GatewayConfig(
                samchon_graph_allowed_roots=(root,),
                samchon_graph_schema_cwd=root,
                samchon_graph_max_sessions=max_sessions,
            )
        )

    def _patch_mcp(self, harness):
        return (
            patch(
                "hermes_mcp_gateway.upstreams.stdio_client",
                side_effect=harness.stdio_client,
            ),
            patch(
                "hermes_mcp_gateway.upstreams.ClientSession",
                side_effect=harness.client_session,
            ),
        )

    async def test_cancelled_discovery_aborts_owner_and_exits_transports(self):
        with TemporaryDirectory() as root:
            client = self._client(root)
            harness = GraphHarness([root])
            started = asyncio.Event()

            async def wait_for_cancel():
                started.set()
                await asyncio.Future()

            harness.sessions[root].initialize = wait_for_cancel
            stdio_patch, session_patch = self._patch_mcp(harness)
            with stdio_patch, session_patch:
                discovery = asyncio.create_task(client.describe())
                await asyncio.wait_for(started.wait(), timeout=1)
                discovery.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(discovery, timeout=1)

                for context in harness.contexts[root]:
                    await asyncio.wait_for(context.exited.wait(), timeout=1)
                    self.assertIs(context.exit_owner, context.owner)
                self.assertEqual(client._sessions, {})
                self.assertFalse(
                    any(
                        task.get_name() == f"samchon-graph:{root}" and not task.done()
                        for task in asyncio.all_tasks()
                    )
                )
                await asyncio.wait_for(client.close(), timeout=1)

    async def test_cancelled_inflight_call_stays_pinned_during_lru_pressure(self):
        with TemporaryDirectory() as root_a, TemporaryDirectory() as root_b:
            client = SamchonGraphClient(
                GatewayConfig(
                    samchon_graph_allowed_roots=(root_a, root_b),
                    samchon_graph_schema_cwd=root_a,
                    samchon_graph_max_sessions=1,
                )
            )
            harness = GraphHarness([root_a, root_b], block_root=root_a)
            stdio_patch, session_patch = self._patch_mcp(harness)
            with stdio_patch, session_patch:
                cancelled = asyncio.create_task(
                    client.call({"cwd": root_a, "question": "once"})
                )
                await asyncio.wait_for(
                    harness.sessions[root_a].call_started.wait(), timeout=1
                )
                cancelled.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(cancelled, timeout=1)

                session_a = client._sessions[client.resolve_cwd(root_a)]
                self.assertTrue(session_a.has_active_work)
                result_b = await asyncio.wait_for(
                    client.call({"cwd": root_b, "question": "independent"}),
                    timeout=1,
                )
                self.assertEqual(self._result_text(result_b), root_b)
                self.assertFalse(harness.sessions[root_a].release_call.is_set())
                self.assertEqual(harness.sessions[root_a].call_count, 1)
                self.assertEqual(harness.sessions[root_b].call_count, 1)

                harness.sessions[root_a].release_call.set()
                await asyncio.wait_for(
                    harness.sessions[root_a].call_finished.wait(), timeout=1
                )
                await asyncio.wait_for(client.close(), timeout=1)

    async def test_slow_lru_close_does_not_hold_client_lock(self):
        with (
            TemporaryDirectory() as root_a,
            TemporaryDirectory() as root_b,
            TemporaryDirectory() as root_c,
        ):
            client = SamchonGraphClient(
                GatewayConfig(
                    samchon_graph_allowed_roots=(root_a, root_b, root_c),
                    samchon_graph_schema_cwd=root_a,
                    samchon_graph_max_sessions=1,
                )
            )
            harness = GraphHarness(
                [root_a, root_b, root_c], slow_close_root=root_a
            )
            stdio_patch, session_patch = self._patch_mcp(harness)
            with stdio_patch, session_patch:
                await client.call({"cwd": root_a, "question": "first"})
                replacing = asyncio.create_task(
                    client.call({"cwd": root_b, "question": "second"})
                )
                await asyncio.wait_for(harness.close_started.wait(), timeout=1)

                independent = await asyncio.wait_for(
                    client.call({"cwd": root_c, "question": "third"}),
                    timeout=1,
                )
                self.assertEqual(self._result_text(independent), root_c)
                self.assertFalse(replacing.done())

                harness.release_close.set()
                result_b = await asyncio.wait_for(replacing, timeout=1)
                self.assertEqual(self._result_text(result_b), root_b)
                await asyncio.wait_for(client.close(), timeout=1)

    async def test_cancelled_lru_eviction_is_owned_until_client_close(self):
        with TemporaryDirectory() as root_a, TemporaryDirectory() as root_b:
            client = SamchonGraphClient(
                GatewayConfig(
                    samchon_graph_allowed_roots=(root_a, root_b),
                    samchon_graph_schema_cwd=root_a,
                    samchon_graph_max_sessions=1,
                )
            )
            harness = GraphHarness([root_a, root_b], slow_close_root=root_a)
            stdio_patch, session_patch = self._patch_mcp(harness)
            with stdio_patch, session_patch:
                await client.call({"cwd": root_a, "question": "first"})
                replacing = asyncio.create_task(
                    client.call({"cwd": root_b, "question": "second"})
                )
                await asyncio.wait_for(harness.close_started.wait(), timeout=1)
                replacing.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(replacing, timeout=1)

                shutdown = asyncio.create_task(client.close())
                try:
                    await asyncio.wait({shutdown}, timeout=0.05)
                    self.assertFalse(shutdown.done())
                finally:
                    harness.release_close.set()
                await asyncio.wait_for(shutdown, timeout=1)
                for root in (root_a, root_b):
                    for context in harness.contexts[root]:
                        self.assertTrue(context.exited.is_set())

    async def test_startup_abort_timeout_remains_owned_by_client_close(self):
        with TemporaryDirectory() as root:
            client = self._client(root)
            harness = GraphHarness([root], slow_close_root=root)
            started = asyncio.Event()

            async def wait_for_cancel():
                started.set()
                await asyncio.Future()

            harness.sessions[root].initialize = wait_for_cancel
            stdio_patch, session_patch = self._patch_mcp(harness)
            with stdio_patch, session_patch:
                discovery = asyncio.create_task(client.describe())
                await asyncio.wait_for(started.wait(), timeout=1)
                discovery.cancel()
                await asyncio.wait_for(harness.close_started.wait(), timeout=1)
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(discovery, timeout=2)

                shutdown = asyncio.create_task(client.close())
                try:
                    await asyncio.wait({shutdown}, timeout=0.05)
                    self.assertFalse(shutdown.done())
                finally:
                    harness.release_close.set()
                await asyncio.wait_for(shutdown, timeout=1)
                for context in harness.contexts[root]:
                    self.assertTrue(context.exited.is_set())

    @staticmethod
    def _result_text(result):
        content = result.content[0]
        assert isinstance(content, types.TextContent)
        return content.text


if __name__ == "__main__":
    unittest.main()
