import unittest

from plugins.memory.honcho.client import HonchoClientConfig

from hermes_mcp_gateway.memory import resolve_memory_config


class IdentityTests(unittest.TestCase):
    def test_chatgpt_ai_peer_preserves_resolved_user_workspace_and_strategy(self):
        base = HonchoClientConfig(
            host="hermes",
            workspace_id="hermes",
            peer_name="ilker",
            ai_peer="hermes",
            enabled=True,
            base_url="http://127.0.0.1:8000",
            session_strategy="per-directory",
        )
        resolved = resolve_memory_config(base, ai_peer="chatgpt")
        self.assertEqual(resolved.workspace_id, "hermes")
        self.assertEqual(resolved.peer_name, "ilker")
        self.assertEqual(resolved.session_strategy, "per-directory")
        self.assertEqual(resolved.ai_peer, "chatgpt")
        self.assertEqual(base.ai_peer, "hermes")


if __name__ == "__main__":
    unittest.main()
