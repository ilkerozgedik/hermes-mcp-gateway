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
