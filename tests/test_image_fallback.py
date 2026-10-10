"""Image discovery must not depend on an OAuth credential's temporary cooldown."""

import json
import unittest
from unittest.mock import AsyncMock, patch

from mcp import types

from hermes_mcp_gateway.config import (
    CONTEXT_REQUIRED,
    HERMES_REQUIRED,
    SAMCHON_GRAPH_TOOL,
    WEB_TOOLS,
)
from hermes_mcp_gateway.server import (
    CatalogEntry,
    Gateway,
    call_deferred_image,
    deferred_image_tool,
)


def tool(name):
    return types.Tool(
        name=name, description=name, input_schema={"type": "object", "properties": {}}
    )


class DeferredImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_cooldown_does_not_break_startup_or_hide_image_tool(self):
        gateway = Gateway()
        gateway.hermes.start = AsyncMock()
        gateway.hermes.call = AsyncMock()
        gateway.context.discover = AsyncMock(
            return_value=[tool(name) for name in CONTEXT_REQUIRED]
        )
        gateway.hermes.discover = AsyncMock(
            return_value=[
                tool(name)
                for name in (HERMES_REQUIRED | WEB_TOOLS) - {"image_generate"}
            ]
        )
        gateway.graph.describe = AsyncMock(return_value=tool(SAMCHON_GRAPH_TOOL))
        gateway.memory.start = AsyncMock()
        gateway.probe_web_tools = AsyncMock(return_value=True)

        await gateway.start()

        self.assertTrue(gateway._deferred_image)
        self.assertIn("image_generate", gateway.catalog)
        assert gateway.discovery is not None
        spec = gateway.discovery.describe({"names": ["image_generate"]})["tools"][
            "image_generate"
        ]
        self.assertIn("prompt", spec["parameters"]["properties"])
        with patch(
            "hermes_mcp_gateway.server.call_deferred_image",
            return_value=types.CallToolResult(
                content=[
                    types.TextContent(
                        text='{"success":false,"error":"provider rate limited"}'
                    )
                ],
                is_error=True,
            ),
        ) as call:
            result = await gateway.call(
                "tool_call",
                {
                    "calls": [
                        {"name": "image_generate", "arguments": {"prompt": "mountain"}}
                    ]
                },
            )
        self.assertTrue(result.is_error)
        call.assert_called_once_with({"prompt": "mountain"})
        self.assertEqual(gateway.hermes.call.await_count, 0)  # type: ignore[attr-defined]

    async def test_normal_registered_tool_uses_existing_hermes_route(self):
        gateway = Gateway()
        gateway.catalog["image_generate"] = CatalogEntry(
            tool("image_generate"), "hermes"
        )
        gateway.hermes.call = AsyncMock(
            return_value=types.CallToolResult(content=[types.TextContent(text="ok")])
        )
        result = await gateway.call("image_generate", {"prompt": "mountain"})
        self.assertFalse(result.is_error)
        gateway.hermes.call.assert_awaited_once_with(
            "image_generate", {"prompt": "mountain"}
        )
        self.assertFalse(gateway._deferred_image)

    def test_schema_reads_hermes_capabilities_not_oauth_availability(self):
        with patch(
            "tools.image_generation_tool._build_dynamic_image_schema",
            return_value={
                "description": "generate images",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
            },
        ) as build:
            spec = deferred_image_tool()
        self.assertEqual(spec.name, "image_generate")
        self.assertEqual(spec.input_schema["required"], ["prompt"])
        build.assert_called_once_with()

    def test_fallback_dispatch_respects_hermes_error_contract(self):
        for response, expected_error in (
            ('{"success":false,"error":"rate limited"}', True),
            ('{"success":true,"image":"url"}', False),
        ):
            with (
                self.subTest(response=response),
                patch(
                    "model_tools.handle_function_call", return_value=response
                ) as dispatch,
            ):
                result = call_deferred_image({"prompt": "mountain"})
                self.assertEqual(result.is_error, expected_error)
                assert isinstance(result.content[0], types.TextContent)
                self.assertEqual(
                    json.loads(result.content[0].text), json.loads(response)
                )
                dispatch.assert_called_once_with(
                    "image_generate", {"prompt": "mountain"}
                )


if __name__ == "__main__":
    unittest.main()
