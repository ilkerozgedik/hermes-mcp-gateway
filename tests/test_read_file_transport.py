"""Read-file gateway responses must be usable by structured MCP clients."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp import types

from hermes_mcp_gateway.config import GatewayConfig
from hermes_mcp_gateway.server import Gateway, build_catalog


class ReadFileTransportTests(unittest.IsolatedAsyncioTestCase):
    def gateway(self, value: str, *, failed: bool = False) -> Gateway:
        gateway = object.__new__(Gateway)
        gateway.config = GatewayConfig()
        gateway.catalog = build_catalog(
            [types.Tool(name="read_file", input_schema={"type": "object"}, description="Read")],
            [],
        )
        gateway.hermes = SimpleNamespace(
            call_direct=AsyncMock(
                return_value=types.CallToolResult(
                    content=[types.TextContent(text=value)], is_error=failed
                )
            )
        )
        return gateway

    async def test_read_file_exposes_structured_file_text_without_losing_legacy_text(self):
        payload = {
            "content": "1|hello\\n2|world",
            "total_lines": 2,
            "file_size": 11,
            "truncated": False,
            "not_found": False,
        }
        gateway = self.gateway(json.dumps(payload))
        result = await gateway.call("read_file", {"path": "/tmp/public-test.txt"})
        self.assertFalse(result.is_error)
        self.assertEqual(result.content[0].text, json.dumps(payload))
        self.assertEqual(result.structured_content["file_text"], payload["content"])
        self.assertEqual(result.structured_content["total_lines"], 2)
        self.assertNotIn("content", result.structured_content)

    async def test_oversized_file_response_preserves_pagination(self):
        content = "\n".join(f"{i}|line-{i:03d}-" + "x" * 55 for i in range(1, 24))
        payload = {
            "content": content,
            "total_lines": 23,
            "file_size": len(content),
            "truncated": False,
            "not_found": False,
        }
        gateway = self.gateway(json.dumps(payload))
        gateway.config = GatewayConfig(max_text_chars=1024)
        result = await gateway.call("read_file", {"path": "/tmp/large.txt"})
        self.assertFalse(result.is_error)
        structured = result.structured_content
        self.assertIsNotNone(structured)
        self.assertTrue(structured["truncated"])
        self.assertTrue(structured["file_text"].startswith("1|line-001"))
        self.assertNotIn("23|line-023", structured["file_text"])
        self.assertIsInstance(structured["next_offset"], int)
        self.assertGreater(structured["next_offset"], 1)
        self.assertIn(f"offset={structured['next_offset']}", structured["hint"])
        self.assertEqual(json.loads(result.content[0].text)["content"], structured["file_text"])

    async def test_oversized_single_line_fails_without_silent_content_loss(self):
        payload = {
            "content": "1|" + "x" * 3000,
            "total_lines": 1,
            "file_size": 3002,
            "truncated": False,
        }
        gateway = self.gateway(json.dumps(payload))
        gateway.config = GatewayConfig(max_text_chars=1024)
        result = await gateway.call("read_file", {"path": "/tmp/minified.txt"})
        self.assertTrue(result.is_error)
        self.assertIn("single line", result.content[0].text)
        self.assertIsNone(result.structured_content)

    async def test_read_file_does_not_convert_errors_to_success(self):
        gateway = self.gateway(json.dumps({"error": "file not found"}), failed=True)
        result = await gateway.call("read_file", {"path": "/tmp/missing"})
        self.assertTrue(result.is_error)
        self.assertIsNone(result.structured_content)

    async def test_read_file_unexpected_payload_is_not_fabricated(self):
        gateway = self.gateway("plain unexpected upstream result")
        result = await gateway.call("read_file", {"path": "/tmp/public-test.txt"})
        self.assertIsNone(result.structured_content)
        self.assertEqual(result.content[0].text, "plain unexpected upstream result")


if __name__ == "__main__":
    unittest.main()
