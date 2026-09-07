import unittest
from unittest.mock import patch

from mcp import types

from hermes_mcp_gateway.config import GatewayConfig
from hermes_mcp_gateway.upstreams import HermesToolsClient, _vision_value_to_result


class VisionResultConversionTests(unittest.TestCase):
    def test_multimodal_envelope_becomes_mcp_text_and_image_content(self):
        raw = {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "Inspect this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
            ],
            "text_summary": "Image attached.",
        }
        result = _vision_value_to_result(raw)
        self.assertFalse(result.is_error)
        self.assertEqual([type(item).__name__ for item in result.content], ["TextContent", "ImageContent"])
        self.assertEqual(result.content[1].mime_type, "image/png")
        self.assertEqual(result.content[1].data, "QUJD")

    def test_invalid_multimodal_image_data_fails_closed(self):
        raw = {
            "_multimodal": True,
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%%"}},
            ],
        }
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            _vision_value_to_result(raw)

    def test_plain_text_result_remains_text(self):
        result = _vision_value_to_result("legacy vision answer")
        self.assertFalse(result.is_error)
        self.assertEqual(result.content[0].text, "legacy vision answer")


class VisionClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_vision_bypasses_stdio_mcp_output_validation(self):
        client = HermesToolsClient(GatewayConfig())
        envelope = {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "Inspect this image."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            ],
        }
        with patch("model_tools.handle_function_call", return_value=envelope) as dispatch:
            result = await client.call_vision({"image_url": "x", "question": "y"})
        dispatch.assert_called_once_with("vision_analyze", {"image_url": "x", "question": "y"})
        self.assertIsInstance(result.content[1], types.ImageContent)


if __name__ == "__main__":
    unittest.main()
