import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp import types

from hermes_mcp_gateway.server import Gateway, build_catalog, create_app
from hermes_mcp_gateway.tool_discovery import ToolDiscovery, public_schemas


def tool(name):
    return types.Tool(
        name=name, input_schema={"type": "object", "properties": {}}, description=name
    )


class DiscoveryTests(unittest.TestCase):
    def test_public_surface_and_exact_search(self):
        catalog = build_catalog(
            [tool("ctx_execute")],
            [tool("skills_list"), tool("browser_click")],
            [tool("memory_context")],
            [tool("startup_context")],
        )
        self.assertEqual(len(public_schemas(catalog)), 6)
        discovery = ToolDiscovery(catalog)
        self.assertEqual(
            discovery.search({"queries": ["ctx_execute"]})["results"][0]["matches"],
            ["ctx_execute"],
        )
        self.assertNotIn("skills_list", discovery.catalog)

    def test_catalog_search_all_names(self):
        catalog = build_catalog(
            [tool("ctx_execute")],
            [tool("skills_list"), tool("browser_click")],
            [tool("memory_context")],
            [tool("startup_context")],
            lsp_tools=[tool("lsp_diagnostics")],
        )
        discovery = ToolDiscovery(catalog)
        for name in discovery.catalog:
            with self.subTest(name=name):
                self.assertEqual(
                    discovery.search({"queries": [name], "limit": 1})["results"][0][
                        "matches"
                    ],
                    [name],
                )
        self.assertIn(
            "browser_click", discovery.describe({"names": ["browser_click"]})["tools"]
        )
        self.assertEqual(
            discovery.describe({"names": ["unknown"]})["not_found"], ["unknown"]
        )

    def test_invalid_inputs_rejected(self):
        discovery = ToolDiscovery(
            build_catalog(
                [tool("ctx_execute")],
                [tool("skills_list")],
                [tool("memory_context")],
                [tool("startup_context")],
            )
        )
        for args in (
            {"queries": []},
            {"queries": [""]},
            {"queries": ["x"] * 8},
            {"queries": ["x"], "limit": True},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                discovery.search(args)
        for args in (
            {"calls": []},
            {"calls": [{"name": "ctx_execute", "arguments": {}}] * 2},
            {"calls": [{"name": "unknown", "arguments": {}}]},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                discovery.validate_call(args)


class BridgeDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gateway = Gateway()
        self.gateway.catalog = build_catalog(
            [tool("ctx_execute")],
            [tool("skills_list"), tool("browser_click")],
            [tool("memory_context")],
            [tool("startup_context")],
        )
        self.gateway.discovery = ToolDiscovery(self.gateway.catalog)

    async def test_direct_hidden_denied_and_bridge_works(self):
        self.gateway.context.call = AsyncMock(
            return_value=types.CallToolResult(content=[types.TextContent(text="ok")])
        )
        result = await self.gateway.call(
            "tool_call", {"calls": [{"name": "ctx_execute", "arguments": {}}]}
        )
        assert isinstance(result.content[0], types.TextContent)
        self.assertEqual(result.content[0].text, "ok")
        self.gateway.context.call.assert_awaited_once_with("ctx_execute", {})

    async def test_mcp_endpoint_rejects_direct_hidden_tools(self):
        with patch("hermes_mcp_gateway.server.Server") as server:
            create_app()
        callback = server.call_args.kwargs["on_call_tool"]
        result = await callback(None, SimpleNamespace(name="ctx_execute", arguments={}))
        self.assertTrue(result.is_error)

    async def test_browser_session_forwarded(self):
        self.gateway.hermes.call_browser = AsyncMock(
            return_value=types.CallToolResult(content=[types.TextContent(text="ok")])
        )
        result = await self.gateway.call(
            "tool_call",
            {"calls": [{"name": "browser_click", "arguments": {}}]},
            task_id="chatgpt:session-a",
        )
        self.assertFalse(result.is_error)
        self.gateway.hermes.call_browser.assert_awaited_once_with(
            "browser_click", {}, task_id="chatgpt:session-a"
        )

    async def test_schema_and_error_preserved(self):
        self.gateway.context.call = AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(text="failed")],
                structured_content={"error": "blocked"},
                is_error=True,
            )
        )
        result = await self.gateway.call(
            "tool_call", {"calls": [{"name": "ctx_execute", "arguments": {}}]}
        )
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content, {"error": "blocked"})
        search = await self.gateway.call("tool_search", {"queries": ["browser click"]})
        assert isinstance(search.content[0], types.TextContent)
        self.assertIn("browser_click", json.loads(search.content[0].text)["tools"])
