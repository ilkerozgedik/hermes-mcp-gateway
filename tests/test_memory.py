import unittest

from hermes_mcp_gateway.memory import memory_tool_schemas, validate_memory_write


class MemoryPolicyTests(unittest.TestCase):
    def test_memory_tool_names_are_renamed_honcho_schemas(self):
        names = {tool.name for tool in memory_tool_schemas()}
        self.assertEqual(
            names,
            {
                "memory_profile",
                "memory_search",
                "memory_context",
                "memory_reasoning",
                "memory_conclude",
            },
        )

    def test_profile_is_read_only(self):
        profile = next(t for t in memory_tool_schemas() if t.name == "memory_profile")
        self.assertNotIn("card", profile.input_schema.get("properties", {}))

    def test_conclusion_write_requires_durable_kind(self):
        ok, _ = validate_memory_write(
            {
                "conclusion": "User prefers concise technical answers.",
                "kind": "preference",
            }
        )
        self.assertTrue(ok)
        ok, error = validate_memory_write(
            {"conclusion": "User prefers concise technical answers."}
        )
        self.assertFalse(ok)
        self.assertIn("kind", error)

    def test_conclusion_rejects_secrets(self):
        ok, error = validate_memory_write(
            {
                "conclusion": "API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789",
                "kind": "project_state",
            }
        )
        self.assertFalse(ok)
        self.assertIn("sensitive", error.lower())

    def test_conclusion_rejects_temporary_chat_detail(self):
        ok, error = validate_memory_write(
            {
                "conclusion": "For this chat, use the temporary branch foo.",
                "kind": "project_state",
            }
        )
        self.assertFalse(ok)
        self.assertIn("temporary", error.lower())

    def test_list_delete_do_not_require_kind(self):
        self.assertTrue(validate_memory_write({"list": True})[0])
        self.assertTrue(validate_memory_write({"delete_id": "abc123"})[0])


if __name__ == "__main__":
    unittest.main()
