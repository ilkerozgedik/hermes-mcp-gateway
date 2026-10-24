from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from mcp import types

from hermes_mcp_gateway.lsp import LSP_TOOLS
from hermes_mcp_gateway.lsp_stdio import create_handlers


class LSPStdioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.adapter = Mock()
        self.adapter.call.return_value = {"diagnostics": []}
        self.list_tools, self.call_tool = create_handlers(self.adapter)

    async def test_only_existing_read_only_lsp_tools_are_exposed(self) -> None:
        result = await self.list_tools(None, None)
        self.assertEqual({tool.name for tool in result.tools}, LSP_TOOLS)
        self.assertFalse(any("rename" in tool.name or "format" in tool.name for tool in result.tools))

    async def test_delegates_to_existing_adapter(self) -> None:
        arguments = {"file": "/home/hermes/worktrees/project/example.py"}
        result = await self.call_tool(
            None, types.CallToolRequestParams(name="lsp_diagnostics", arguments=arguments)
        )
        self.assertFalse(result.is_error)
        self.assertEqual(json.loads(result.content[0].text), {"diagnostics": []})
        self.adapter.call.assert_called_once_with("lsp_diagnostics", arguments)

    async def test_denies_unknown_tool(self) -> None:
        result = await self.call_tool(
            None, types.CallToolRequestParams(name="write_file", arguments={})
        )
        self.assertTrue(result.is_error)
        self.adapter.call.assert_not_called()

    async def test_surfaces_adapter_permission_denial(self) -> None:
        self.adapter.call.side_effect = PermissionError("outside allowed roots")
        result = await self.call_tool(
            None, types.CallToolRequestParams(
                name="lsp_diagnostics", arguments={"file": "/etc/passwd"}
            )
        )
        self.assertTrue(result.is_error)
        self.assertIn("outside allowed roots", result.content[0].text)

    async def test_redacts_sensitive_values_in_result_and_error(self) -> None:
        secret = "sk-proj-FAKE-DEMO-123456789"
        self.adapter.call.return_value = {"result": {"contents": "OPENAI_API_KEY=" + secret}}
        result = await self.call_tool(
            None, types.CallToolRequestParams(name="lsp_hover", arguments={})
        )
        item = result.content[0]
        assert isinstance(item, types.TextContent)
        self.assertNotIn(secret, item.text)
        self.assertIn("contents", json.loads(item.text)["result"])
        self.adapter.call.side_effect = RuntimeError("OPENAI_API_KEY=" + secret)
        result = await self.call_tool(
            None, types.CallToolRequestParams(name="lsp_hover", arguments={})
        )
        self.assertTrue(result.is_error)
        item = result.content[0]
        assert isinstance(item, types.TextContent)
        self.assertNotIn(secret, item.text)

    async def test_large_result_stays_bounded_valid_json(self) -> None:
        self.adapter.call.return_value = {"contents": ('quote"\\\n' * 20_000)}
        result = await self.call_tool(
            None, types.CallToolRequestParams(name="lsp_hover", arguments={})
        )
        item = result.content[0]
        assert isinstance(item, types.TextContent)
        self.assertLessEqual(len(item.text), 65_536)
        self.assertTrue(json.loads(item.text)["truncated"])
