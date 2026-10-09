import unittest
from pathlib import Path


class SystemdTemplateTests(unittest.TestCase):
    def test_service_is_bootstrap_relocatable_and_non_cascading(self):
        repo_root = Path(__file__).resolve().parent.parent
        unit = (repo_root / "systemd/hermes-mcp-gateway.service").read_text()
        self.assertIn("User=user", unit)
        self.assertIn("Group=user", unit)
        self.assertIn("WorkingDirectory=/srv/agents/src/hermes-mcp-gateway", unit)
        self.assertIn("Environment=HOME=/home/user", unit)
        self.assertIn("Environment=HERMES_HOME=/home/user/.hermes", unit)
        self.assertIn("EnvironmentFile=-/etc/agents/hermes.env", unit)
        self.assertNotIn("EnvironmentFile=-/etc/agents/hermes-headroom.env", unit)
        self.assertNotIn("EnvironmentFile=-/etc/agents/ilkerce.env", unit)
        self.assertNotIn("HERMES_HOME=/home/user/.hermes/profiles/", unit)
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

    def test_coordinator_service_is_loopback_and_bootstrap_relocatable(self):
        repo_root = Path(__file__).resolve().parent.parent
        unit = (repo_root / "systemd/multi-agent-coordinator.service").read_text()
        self.assertIn("User=user", unit)
        self.assertIn("Group=user", unit)
        self.assertIn("/home/user/worktrees", unit)
        self.assertIn("/srv/agents/runtime/multi-agent/coordinator.sqlite", unit)
        self.assertIn("AGENT_COORDINATOR_REPOS=peacify", unit)
        self.assertIn("hermes_mcp_gateway.coordinator.server", unit)
        self.assertIn("ProtectSystem=full", unit)
        self.assertIn("UMask=0077", unit)
        self.assertNotIn("Requires=", unit)


if __name__ == "__main__":
    unittest.main()
