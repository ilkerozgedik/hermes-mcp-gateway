import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from mcp import types

from hermes_mcp_gateway.capabilities import CAPABILITY_TOOLS
from hermes_mcp_gateway.config import (
    BROWSER_TOOLS,
    HERMES_ALLOWLIST,
    HERMES_REQUIRED,
    MEMORY_TOOLS,
    SERENA_TOOLS,
    WEB_TOOLS,
)
from hermes_mcp_gateway.server import build_catalog, normalize_result
from hermes_mcp_gateway.upstreams import missing_expected


def tool(name: str) -> types.Tool:
    return types.Tool(
        name=name, description=name, input_schema={"type": "object", "properties": {}}
    )


class GatewayPolicyTests(unittest.TestCase):
    def test_default_deny_filters_hermes_tools(self):
        available = [
            tool("vision_analyze"),
            tool("skills_list"),
            tool("terminal"),
            tool("image_generate"),
            tool("computer_use"),
        ]
        catalog = build_catalog([tool("ctx_search")], available, [])
        self.assertEqual(
            set(catalog),
            {"ctx_search", "vision_analyze", "skills_list", "terminal", "image_generate"},
        )
        self.assertNotIn("computer_use", catalog)

    def test_collision_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            build_catalog([tool("vision_analyze")], [tool("vision_analyze")], [])

    def test_missing_expected_is_explicit(self):
        self.assertEqual(
            missing_expected({"vision_analyze"}, {"vision_analyze", "skills_list"}),
            {"skills_list"},
        )

    def test_output_is_redacted_and_bounded(self):
        raw = types.CallToolResult(
            content=[
                types.TextContent(
                    text="OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789\n"
                    + "x" * 100
                )
            ]
        )
        out = normalize_result(raw, limit=64)
        text = out.content[0].text
        self.assertLessEqual(len(text), 64)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz0123456789", text)
        self.assertIn("truncated", text)

    def test_requested_native_tools_are_explicitly_allowlisted(self):
        expected = {
            "terminal",
            "process",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "image_generate",
            "video_analyze",
        }
        self.assertTrue(expected <= HERMES_ALLOWLIST)
        self.assertFalse(
            {"computer_use", "browser_exec", "browser_cdp", "skill_manage", "memory"}
            & HERMES_ALLOWLIST
        )

    def test_web_is_a_final_gate_not_a_fabricated_startup_tool(self):
        self.assertEqual(WEB_TOOLS, {"web_search", "web_extract"})
        self.assertFalse(WEB_TOOLS & HERMES_REQUIRED)
        self.assertEqual(HERMES_REQUIRED | WEB_TOOLS, HERMES_ALLOWLIST)

    def test_capabilities_are_not_stateless_hermes_allowlist_tools(self):
        self.assertFalse(CAPABILITY_TOOLS & HERMES_ALLOWLIST)

    def test_serena_surface_is_read_only_semantic_subset(self):
        self.assertEqual(
            SERENA_TOOLS,
            {
                "activate_project",
                "find_declaration",
                "find_implementations",
                "find_referencing_symbols",
                "find_symbol",
                "get_diagnostics_for_file",
                "get_symbols_overview",
            },
        )
        self.assertFalse(
            {"execute_shell_command", "replace_symbol_body", "rename_symbol", "create_text_file"}
            & SERENA_TOOLS
        )

    def test_catalog_exposes_only_allowlisted_serena_tools(self):
        catalog = build_catalog(
            [],
            [],
            [],
            serena_tools=[tool("find_symbol"), tool("execute_shell_command")],
        )
        self.assertIn("find_symbol", catalog)
        self.assertEqual(catalog["find_symbol"].source, "serena")
        self.assertNotIn("execute_shell_command", catalog)

    def test_camofox_browser_tools_are_explicitly_exposed(self):
        expected = {
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_press",
            "browser_snapshot",
            "browser_scroll",
            "browser_back",
            "browser_get_images",
            "browser_console",
            "browser_vision",
        }
        self.assertEqual(BROWSER_TOOLS, expected)
        self.assertEqual(
            {name for name in HERMES_ALLOWLIST if name.startswith("browser_")},
            expected,
        )
        self.assertNotIn("browser_exec", HERMES_ALLOWLIST)
        self.assertNotIn("browser_cdp", HERMES_ALLOWLIST)

    def test_vision_analyze_drops_broken_upstream_output_schema(self):
        vision = types.Tool(
            name="vision_analyze",
            description="vision",
            input_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
            },
        )
        catalog = build_catalog([], [vision], [])
        self.assertIsNone(catalog["vision_analyze"].tool.output_schema)
        self.assertIsNotNone(vision.output_schema)

    def test_browser_tools_drop_upstream_output_schema_for_text_dispatch(self):
        browser = types.Tool(
            name="browser_navigate",
            description="browser",
            input_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
            },
        )
        catalog = build_catalog([], [browser], [])
        self.assertIsNone(catalog["browser_navigate"].tool.output_schema)
        self.assertIsNotNone(browser.output_schema)

    def test_public_surface_count_is_50_with_serena(self):
        from hermes_mcp_gateway.config import CONTEXT_REQUIRED

        self.assertEqual(
            len(CONTEXT_REQUIRED)
            + len(HERMES_ALLOWLIST)
            + len(CAPABILITY_TOOLS)
            + len(MEMORY_TOOLS)
            + len(SERENA_TOOLS)
            + 1,
            50,
        )


if __name__ == "__main__":
    unittest.main()


class BrowserGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_navigate_recovers_once_from_stale_camofox_tab(self):
        from unittest.mock import patch

        from hermes_mcp_gateway.config import GatewayConfig
        from hermes_mcp_gateway.upstreams import HermesToolsClient

        stale = '{"success": false, "error": "410 Client Error: Gone"}'
        fresh = '{"success": true, "url": "https://example.com"}'
        client = HermesToolsClient(GatewayConfig())
        with patch(
            "model_tools.handle_function_call", side_effect=[stale, fresh]
        ) as dispatch, patch(
            "hermes_mcp_gateway.upstreams._reset_camofox_session"
        ) as reset:
            result = await client.call_browser(
                "browser_navigate",
                {"url": "https://example.com"},
                task_id="chatgpt:session-a",
            )

        self.assertEqual(result.content[0].text, fresh)
        self.assertEqual(dispatch.call_count, 2)
        reset.assert_called_once_with("chatgpt:session-a")

    async def test_non_navigate_stale_camofox_result_is_actionable_not_410(self):
        from unittest.mock import patch

        from hermes_mcp_gateway.config import GatewayConfig
        from hermes_mcp_gateway.upstreams import HermesToolsClient

        stale = '{"success": false, "error": "410 Client Error: Gone"}'
        client = HermesToolsClient(GatewayConfig())
        with patch("model_tools.handle_function_call", return_value=stale), patch(
            "hermes_mcp_gateway.upstreams._reset_camofox_session"
        ) as reset:
            result = await client.call_browser(
                "browser_snapshot", {}, task_id="chatgpt:session-a"
            )

        text = result.content[0].text
        self.assertNotIn("410", text)
        self.assertIn('"code": "stale_tab"', text)
        reset.assert_called_once_with("chatgpt:session-a")

    async def test_browser_dispatch_uses_request_scoped_task_id(self):
        from hermes_mcp_gateway.server import Gateway

        gateway = Gateway()
        gateway.catalog = build_catalog([], [tool("browser_navigate")], [])
        gateway.hermes.call_browser = AsyncMock(
            return_value=types.CallToolResult(content=[types.TextContent(text='{"success":true}')])
        )
        gateway.hermes.call = AsyncMock()

        result = await gateway.call(
            "browser_navigate", {"url": "https://example.com"}, task_id="chatgpt:session-a"
        )

        self.assertFalse(result.is_error)
        gateway.hermes.call_browser.assert_awaited_once_with(
            "browser_navigate", {"url": "https://example.com"}, task_id="chatgpt:session-a"
        )
        gateway.hermes.call.assert_not_awaited()

    def test_browser_task_id_prefers_mcp_session_header(self):
        from hermes_mcp_gateway.server import browser_task_id

        ctx = SimpleNamespace(
            request=SimpleNamespace(headers={"mcp-session-id": "session-a"}),
            session=SimpleNamespace(),
        )
        self.assertEqual(browser_task_id(ctx), "chatgpt:session-a")

    def test_browser_task_id_has_safe_fallback(self):
        from hermes_mcp_gateway.server import browser_task_id

        ctx = SimpleNamespace(request=None, session=SimpleNamespace())
        self.assertEqual(browser_task_id(ctx), "chatgpt:gateway")


class VisionGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_vision_analyze_uses_native_adapter_and_preserves_image_content(self):
        from unittest.mock import AsyncMock

        from hermes_mcp_gateway.server import Gateway

        gateway = Gateway()
        gateway.catalog = build_catalog([], [tool("vision_analyze")], [])
        gateway.hermes.call_vision = AsyncMock(
            return_value=types.CallToolResult(
                content=[
                    types.TextContent(text="image ready"),
                    types.ImageContent(data="QUJD", mime_type="image/png"),
                ]
            )
        )
        gateway.hermes.call = AsyncMock()

        result = await gateway.call(
            "vision_analyze",
            {"image_url": "https://example.com/a.png", "question": "what?"},
        )

        self.assertFalse(result.is_error)
        self.assertEqual([type(item).__name__ for item in result.content], ["TextContent", "ImageContent"])
        gateway.hermes.call_vision.assert_awaited_once()
        gateway.hermes.call.assert_not_awaited()


class GatewayOwnedToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_context_is_catalogued_as_gateway_tool(self):
        from hermes_mcp_gateway.server import CatalogEntry, Gateway
        from hermes_mcp_gateway.startup import startup_tool_schema

        catalog = build_catalog([], [], [], [startup_tool_schema()])
        self.assertEqual(catalog["startup_context"].source, "gateway")
        self.assertEqual(catalog["startup_context"].upstream_name, "startup_context")
        self.assertIsInstance(catalog["startup_context"], CatalogEntry)

        gateway = Gateway()
        gateway.catalog = catalog
        gateway.startup.read = MagicMock(
            return_value={"files": [{"path": "fixed", "content": "rules"}]}
        )
        result = await gateway.call("startup_context", {})
        self.assertFalse(result.is_error)
        self.assertIn('"path": "fixed"', result.content[0].text)
        gateway.startup.read.assert_called_once_with()

    async def test_startup_context_rejects_runtime_arguments(self):
        from hermes_mcp_gateway.server import Gateway
        from hermes_mcp_gateway.startup import startup_tool_schema

        gateway = Gateway()
        gateway.catalog = build_catalog([], [], [], [startup_tool_schema()])
        gateway.startup.read = MagicMock(return_value={"files": []})
        result = await gateway.call("startup_context", {"path": "/etc/passwd"})
        self.assertTrue(result.is_error)
        self.assertIn("does not accept arguments", result.content[0].text)
        gateway.startup.read.assert_not_called()


class GatewayHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_context_is_a_readiness_component(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from hermes_mcp_gateway.config import HERMES_REQUIRED, WEB_TOOLS
        from hermes_mcp_gateway.server import Gateway

        class Response:
            status_code = 200

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def get(self, _url):
                return Response()

        gateway = Gateway()
        gateway.hermes.discover = AsyncMock(
            return_value=[tool(name) for name in sorted(HERMES_REQUIRED | WEB_TOOLS)]
        )
        gateway.startup.check = MagicMock(return_value=False)
        with patch(
            "hermes_mcp_gateway.server.httpx.AsyncClient", return_value=Client()
        ):
            payload = await gateway.health()
        self.assertIn("startup_context", payload["components"])
        self.assertFalse(payload["components"]["startup_context"])
        self.assertEqual(payload["status"], "degraded")


class BrowserIndependenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_readiness_has_no_cloakbrowser_component(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from hermes_mcp_gateway.config import HERMES_REQUIRED, WEB_TOOLS
        from hermes_mcp_gateway.server import Gateway

        class Response:
            status_code = 200

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def get(self, _url):
                return Response()

        gateway = Gateway()
        gateway.hermes.discover = AsyncMock(
            return_value=[tool(name) for name in sorted(HERMES_REQUIRED | WEB_TOOLS)]
        )
        gateway.web_tools_ready = True
        gateway.serena.healthy = AsyncMock(return_value=True)
        gateway.startup.check = MagicMock(return_value=True)
        gateway.capabilities.check = MagicMock(return_value=True)
        with patch(
            "hermes_mcp_gateway.server.httpx.AsyncClient", return_value=Client()
        ):
            payload = await gateway.health()
        self.assertNotIn("cloakbrowser_cdp", payload["components"])
        self.assertEqual(payload["status"], "ok")


class ContextReadinessContractTests(unittest.TestCase):
    def test_gateway_readiness_uses_context_readyz_not_healthz(self):
        from hermes_mcp_gateway.config import GatewayConfig

        cfg = GatewayConfig()
        self.assertEqual(cfg.context_ready_url, "http://127.0.0.1:3050/readyz")


class OutputSanitizationTests(unittest.TestCase):
    def test_structured_content_is_redacted_even_when_under_limit(self):
        raw = types.CallToolResult(
            content=[types.TextContent(text="ok")],
            structured_content={
                "credential": "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789"
            },
        )
        out = normalize_result(raw, limit=4096)
        rendered = str(out.structured_content)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz0123456789", rendered)

    def test_non_text_content_cannot_bypass_output_budget(self):
        image = types.ImageContent(type="image", data="A" * 4096, mimeType="image/png")
        raw = types.CallToolResult(content=[image])
        out = normalize_result(raw, limit=128)
        self.assertEqual(len(out.content), 1)
        self.assertIsInstance(out.content[0], types.TextContent)
        self.assertLessEqual(len(out.content[0].text), 128)
        self.assertIn("omitted", out.content[0].text.lower())


class WebReadinessProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_probe_rejects_error_payload_even_when_mcp_is_error_false(self):
        from unittest.mock import AsyncMock

        from hermes_mcp_gateway.server import Gateway

        gateway = Gateway()
        gateway.hermes.call = AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(text='{"error":"Log in to Nous Portal"}')],
                is_error=False,
            )
        )
        self.assertFalse(await gateway.probe_web_tools())

    async def test_web_probe_accepts_search_and_extract_without_nested_errors(self):
        from unittest.mock import AsyncMock

        from hermes_mcp_gateway.server import Gateway

        gateway = Gateway()
        gateway.hermes.call = AsyncMock(
            side_effect=[
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            text='{"results":[{"url":"https://example.com"}]}'
                        )
                    ]
                ),
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            text='{"results":[{"url":"https://example.com","content":"ok"}]}'
                        )
                    ]
                ),
            ]
        )
        self.assertTrue(await gateway.probe_web_tools())

class HermesCapabilityGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_capability_tools_are_catalogued_separately_and_dispatched(self):
        from unittest.mock import AsyncMock

        from hermes_mcp_gateway.capabilities import capability_tool_schemas
        from hermes_mcp_gateway.server import Gateway

        gateway = Gateway()
        gateway.catalog = build_catalog(
            [], [], [], [], capability_tools=capability_tool_schemas()
        )
        gateway.capabilities.call = AsyncMock(
            return_value=types.CallToolResult(content=[types.TextContent(text='{"success":true}')])
        )
        result = await gateway.call("session_search", {"query": "auth"})
        self.assertFalse(result.is_error)
        gateway.capabilities.call.assert_awaited_once_with("session_search", {"query": "auth"})
        self.assertEqual(gateway.catalog["session_search"].source, "capability")

    def test_final_surface_contains_exactly_three_capability_tools(self):
        from hermes_mcp_gateway.capabilities import (
            CAPABILITY_TOOLS,
            capability_tool_schemas,
        )

        names = {tool.name for tool in capability_tool_schemas()}
        self.assertEqual(names, {"session_search", "delegate_task", "cronjob"})
        self.assertEqual(names, CAPABILITY_TOOLS)


class HermesCapabilityHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_capability_readiness_degrades_gateway(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from hermes_mcp_gateway.config import HERMES_REQUIRED, WEB_TOOLS
        from hermes_mcp_gateway.server import Gateway

        class Response:
            status_code = 200

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def get(self, _url):
                return Response()

        gateway = Gateway()
        gateway.hermes.discover = AsyncMock(
            return_value=[tool(name) for name in sorted(HERMES_REQUIRED | WEB_TOOLS)]
        )
        gateway.web_tools_ready = True
        gateway.startup.check = MagicMock(return_value=True)
        gateway.capabilities.check = MagicMock(return_value=False)
        with patch("hermes_mcp_gateway.server.httpx.AsyncClient", return_value=Client()):
            payload = await gateway.health()
        self.assertFalse(payload["components"]["hermes_capabilities"])
        self.assertEqual(payload["status"], "degraded")
