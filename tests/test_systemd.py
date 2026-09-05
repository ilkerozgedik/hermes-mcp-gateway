import unittest
from pathlib import Path


class SystemdTemplateTests(unittest.TestCase):
    def test_service_is_bootstrap_relocatable_and_non_cascading(self):
        unit = Path("systemd/hermes-mcp-gateway.service").read_text()
        self.assertIn("User=user", unit)
        self.assertIn("Group=user", unit)
        self.assertIn("WorkingDirectory=/srv/agents/src/hermes-mcp-gateway", unit)
        self.assertIn("Environment=HOME=/home/user", unit)
        self.assertIn("Environment=HERMES_HOME=/home/user/.hermes", unit)
        self.assertIn(
            "Environment=PYTHONPATH=/srv/agents/src/hermes-mcp-gateway/src:/srv/agents/src/hermes-agent",
            unit,
        )
        self.assertIn(
            "Wants=network-online.target context-mode.service agent-runtime.service",
            unit,
        )
        self.assertNotIn("Requires=", unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("NoNewPrivileges=true", unit)


if __name__ == "__main__":
    unittest.main()
