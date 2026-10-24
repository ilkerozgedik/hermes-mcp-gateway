"""Shared MCP output policy must redact without breaking JSON envelopes."""
import json
import unittest

from mcp import types

from hermes_mcp_gateway.output import json_result


def result_text(result: types.CallToolResult) -> str:
    item = result.content[0]
    assert isinstance(item, types.TextContent)
    return item.text


class JsonOutputTests(unittest.TestCase):
    def test_lone_surrogates_remain_utf8_serializable(self):
        for limit in (100, 1024):
            with self.subTest(limit=limit):
                result = json_result({"value": "\ud800x\udfff" * 30}, limit=limit)
                wire = result.model_dump_json()
                wire.encode("utf-8")
                self.assertLessEqual(len(result_text(result)), limit)
                json.loads(result_text(result))

    def test_nested_sensitive_values_are_redacted_before_serialization(self):
        secret = "sk-proj-FAKE-DEMO-123456789"
        result = json_result({"items": [{"value": "OPENAI_API_KEY=" + secret}]}, limit=1024)
        text = result_text(result)
        self.assertNotIn(secret, text)
        self.assertEqual(len(json.loads(text)["items"]), 1)

    def test_escaping_is_included_in_truncation_budget(self):
        for value in ('"' * 1000, "\\" * 1000, "\n" * 1000, "ş" * 1000):
            with self.subTest(value=value[:1]):
                result = json_result({"value": value}, limit=100)
                text = result_text(result)
                self.assertLessEqual(len(text), 100)
                self.assertTrue(json.loads(text)["truncated"])

    def test_small_results_preserve_shape(self):
        payload = {"diagnostics": [], "result": None}
        result = json_result(payload, limit=100)
        self.assertEqual(json.loads(result_text(result)), payload)
