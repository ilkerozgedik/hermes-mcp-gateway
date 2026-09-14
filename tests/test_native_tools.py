import unittest
from unittest.mock import AsyncMock, patch

from mcp import types

from hermes_mcp_gateway.config import DIRECT_HERMES_TOOLS, DIRECT_HERMES_TOOLSETS
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
                "process",
                "video_analyze",
            },
        )
        self.assertEqual(DIRECT_HERMES_TOOLSETS, ("file", "terminal", "video"))

    async def test_direct_dispatch_is_request_scoped(self):
        client = object.__new__(HermesToolsClient)
        with patch("model_tools.handle_function_call", return_value='{"success":true}') as dispatch:
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
            enabled_toolsets=list(DIRECT_HERMES_TOOLSETS),
        )


    async def test_process_operations_cannot_cross_mcp_sessions(self):
        client = object.__new__(HermesToolsClient)
        foreign = type(
            "Session",
            (),
            {"task_id": "default", "session_key": "chatgpt:session-b"},
        )()
        with patch("tools.process_registry.process_registry.get", return_value=foreign):
            with self.assertRaisesRegex(PermissionError, "another MCP session"):
                await client.call_direct(
                    "process",
                    {"action": "kill", "session_id": "proc_foreign"},
                    task_id="chatgpt:session-a",
                )


    async def test_process_operation_accepts_same_mcp_session_key(self):
        client = object.__new__(HermesToolsClient)
        owned = type(
            "Session",
            (),
            {"task_id": "default", "session_key": "chatgpt:session-a"},
        )()
        with patch("tools.process_registry.process_registry.get", return_value=owned):
            with patch("model_tools.handle_function_call", return_value='{"status":"running"}'):
                result = await client.call_direct(
                    "process",
                    {"action": "poll", "session_id": "proc_owned"},
                    task_id="chatgpt:session-a",
                )
        self.assertEqual(result.content[0].text, '{"status":"running"}')


if __name__ == "__main__":
    unittest.main()
