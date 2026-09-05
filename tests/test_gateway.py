import unittest
from unittest.mock import MagicMock

from mcp import types

from hermes_mcp_gateway.config import HERMES_ALLOWLIST, HERMES_REQUIRED, WEB_TOOLS
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
        ]
        catalog = build_catalog([tool("ctx_search")], available, [])
        self.assertEqual(set(catalog), {"ctx_search", "vision_analyze", "skills_list"})
        self.assertNotIn("terminal", catalog)
        self.assertNotIn("image_generate", catalog)

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

    def test_requested_allowlist_has_no_shell_or_file_tools(self):
        forbidden = {
            "terminal",
            "read_file",
            "write_file",
            "patch",
            "search_files",
            "process",
            "image_generate",
            "text_to_speech",
        }
        self.assertFalse(forbidden & HERMES_ALLOWLIST)

    def test_web_is_a_final_gate_not_a_fabricated_startup_tool(self):
        self.assertEqual(WEB_TOOLS, {"web_search", "web_extract"})
        self.assertFalse(WEB_TOOLS & HERMES_REQUIRED)
        self.assertEqual(HERMES_REQUIRED | WEB_TOOLS, HERMES_ALLOWLIST)


if __name__ == "__main__":
    unittest.main()


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
