import unittest
from unittest.mock import ANY, patch

from hermes_mcp_gateway.config import DIRECT_HERMES_TOOLS, DIRECT_HERMES_TOOLSETS, GatewayConfig
from hermes_mcp_gateway.upstreams import HermesToolsClient


class DirectHermesToolTests(unittest.IsolatedAsyncioTestCase):
    def test_direct_tool_contract_excludes_image_generation_and_computer_use(self):
        self.assertEqual(
            DIRECT_HERMES_TOOLS,
            {
                "read_file",
                "write_file",
                "patch",
                "search_files",
                "terminal",
                "process_manage",
                "video_analyze",
            },
        )
        self.assertEqual(DIRECT_HERMES_TOOLSETS, ("file", "terminal", "video"))

    async def test_direct_dispatch_is_request_scoped(self):
        client = HermesToolsClient(GatewayConfig())
        with patch(
            "model_tools.handle_function_call", return_value='{"success":true}'
        ) as dispatch:
            result = await client.call_direct(
                "terminal", {"command": "printf ok"}, task_id="chatgpt:session-a"
            )

        self.assertFalse(result.is_error)
        self.assertEqual(result.content[0].text, '{"success":true}')
        dispatch.assert_called_once_with(
            "terminal",
            {"command": "printf ok"},
            task_id="chatgpt:session-a",
            session_id="chatgpt:session-a",
            tool_call_id=ANY,
            enabled_toolsets=list(DIRECT_HERMES_TOOLSETS),
        )

    async def test_concurrent_direct_calls_have_distinct_hook_call_ids(self):
        import asyncio

        client = HermesToolsClient(GatewayConfig())
        with patch(
            "model_tools.handle_function_call", return_value='{"success":true}'
        ) as dispatch:
            await asyncio.gather(
                client.call_direct(
                    "terminal", {"command": "printf a"}, task_id="chatgpt:session-a"
                ),
                client.call_direct(
                    "terminal", {"command": "printf b"}, task_id="chatgpt:session-a"
                ),
            )
        call_ids = [call.kwargs.get("tool_call_id") for call in dispatch.call_args_list]
        self.assertEqual(len(call_ids), 2)
        self.assertTrue(
            all(call_ids), "each native dispatch must identify its own hook invocation"
        )
        self.assertTrue(call_ids[0] != call_ids[1])
        self.assertTrue(
            all(
                call.kwargs["task_id"] == "chatgpt:session-a"
                for call in dispatch.call_args_list
            )
        )

    async def test_direct_dispatch_binds_native_session_key_without_leaking_context(
        self,
    ):
        from gateway.session_context import get_session_env

        client = HermesToolsClient(GatewayConfig())
        before = get_session_env("HERMES_SESSION_KEY", "")
        seen = []

        def observe(*_args, **_kwargs):
            seen.append(get_session_env("HERMES_SESSION_KEY", ""))
            return '{"success":true}'

        with patch("model_tools.handle_function_call", side_effect=observe):
            await client.call_direct(
                "terminal", {"command": "printf a"}, task_id="chatgpt:session-a"
            )
        self.assertTrue(
            seen == ["chatgpt:session-a"], "native session key binding missing"
        )
        self.assertTrue(get_session_env("HERMES_SESSION_KEY", "") == before)

    async def test_concurrent_direct_calls_bind_distinct_native_session_keys(self):
        import asyncio
        import threading

        from gateway.session_context import get_session_env

        client = HermesToolsClient(GatewayConfig())
        barrier = threading.Barrier(2)
        seen = []

        def observe(*_args, **_kwargs):
            barrier.wait(timeout=5)
            seen.append(get_session_env("HERMES_SESSION_KEY", ""))
            return '{"success":true}'

        with patch("model_tools.handle_function_call", side_effect=observe):
            await asyncio.gather(
                client.call_direct(
                    "terminal", {"command": "printf a"}, task_id="chatgpt:session-a"
                ),
                client.call_direct(
                    "terminal", {"command": "printf b"}, task_id="chatgpt:session-b"
                ),
            )
        self.assertTrue(
            set(seen) == {"chatgpt:session-a", "chatgpt:session-b"},
            "concurrent native session keys must remain isolated",
        )

    async def test_process_operations_cannot_cross_mcp_sessions(self):
        client = HermesToolsClient(GatewayConfig())
        foreign = type(
            "Session",
            (),
            {"task_id": "default", "session_key": "chatgpt:session-b"},
        )()
        with patch("tools.process_registry.process_registry.get", return_value=foreign):
            with self.assertRaisesRegex(PermissionError, "another MCP session"):
                await client.call_direct(
                    "process_manage",
                    {"action": "kill", "session_id": "proc_foreign"},
                    task_id="chatgpt:session-a",
                )

    async def test_process_operation_accepts_same_mcp_session_key(self):
        client = HermesToolsClient(GatewayConfig())
        owned = type(
            "Session",
            (),
            {"task_id": "default", "session_key": "chatgpt:session-a"},
        )()
        with patch("tools.process_registry.process_registry.get", return_value=owned):
            with patch(
                "model_tools.handle_function_call", return_value='{"status":"running"}'
            ):
                result = await client.call_direct(
                    "process_manage",
                    {"action": "poll", "session_id": "proc_owned"},
                    task_id="chatgpt:session-a",
                )
        self.assertEqual(result.content[0].text, '{"status":"running"}')

    async def test_process_manage_operations_cannot_cross_mcp_sessions(self):
        client = HermesToolsClient(GatewayConfig())
        foreign = type(
            "Session",
            (),
            {"task_id": "default", "session_key": "chatgpt:session-b"},
        )()
        with patch("tools.process_registry.process_registry.get", return_value=foreign):
            with self.assertRaisesRegex(PermissionError, "another MCP session"):
                await client.call_direct(
                    "process_manage",
                    {"action": "kill", "session_id": "proc_foreign"},
                    task_id="chatgpt:session-a",
                )

    def test_discover_direct_tools_preserves_process_manage_name(self):
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "process_manage",
                    "description": "manage processes",
                },
            },
            {
                "type": "function",
                "function": {"name": "terminal", "description": "run shell"},
            },
            {
                "type": "function",
                "function": {"name": "unexpected", "description": "ignore"},
            },
        ]
        with patch("model_tools.get_tool_definitions", return_value=definitions):
            tools = HermesToolsClient._discover_direct_tools()
        names = [t.name for t in tools]
        self.assertIn("process_manage", names)
        self.assertNotIn("process", names)
        self.assertIn("terminal", names)
        self.assertNotIn("unexpected", names)


if __name__ == "__main__":
    unittest.main()
