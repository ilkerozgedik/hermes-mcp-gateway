import json
import unittest
from unittest.mock import MagicMock, patch

from hermes_mcp_gateway.capabilities import HermesCapabilities, capability_tool_schemas


class SessionSearchCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_search_calls_hermes_read_only_search_without_profile_override(self):
        capabilities = HermesCapabilities()
        with patch(
            "hermes_mcp_gateway.capabilities.session_search",
            return_value=json.dumps({"success": True, "sessions": [{"session_id": "s1"}]}),
        ) as search:
            result = await capabilities.call("session_search", {"query": "auth", "limit": 2})

        self.assertFalse(result.is_error)
        search.assert_called_once_with(
            query="auth",
            role_filter=None,
            limit=2,
            session_id=None,
            around_message_id=None,
            window=5,
            sort=None,
            detail="adaptive",
        )

    async def test_session_search_rejects_profile_override(self):
        capabilities = HermesCapabilities()
        result = await capabilities.call("session_search", {"query": "auth", "profile": "other"})
        self.assertTrue(result.is_error)
        self.assertIn("profile", result.content[0].text)

    def test_session_search_schema_does_not_expose_profile(self):
        schemas = {tool.name: tool for tool in capability_tool_schemas()}
        self.assertIn("session_search", schemas)
        self.assertNotIn("profile", schemas["session_search"].input_schema["properties"])


class CapabilityErrorPropagationTests(unittest.IsolatedAsyncioTestCase):
    async def test_underlying_hermes_error_payload_sets_mcp_error(self):
        capabilities = HermesCapabilities()
        with patch(
            "hermes_mcp_gateway.capabilities.session_search",
            return_value=json.dumps({"success": False, "error": "session db failed"}),
        ):
            result = await capabilities.call("session_search", {"query": "auth"})
        self.assertTrue(result.is_error)
        self.assertIn("session db failed", result.content[0].text)


class CronCapabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_cron_schema_hides_unsafe_execution_and_provider_fields(self):
        schemas = {tool.name: tool for tool in capability_tool_schemas()}
        self.assertIn("cronjob", schemas)
        props = schemas["cronjob"].input_schema["properties"]
        for forbidden in {
            "model", "provider", "base_url", "script", "no_agent",
            "monitor_script", "monitor_url", "enabled_toolsets",
        }:
            self.assertNotIn(forbidden, props)
        self.assertIn("confirmed", props)

    async def test_cron_mutation_requires_explicit_confirmation(self):
        capabilities = HermesCapabilities()
        with patch("hermes_mcp_gateway.capabilities.cronjob") as cron:
            result = await capabilities.call(
                "cronjob",
                {"action": "create", "schedule": "0 9 * * *", "prompt": "daily summary"},
            )
        self.assertTrue(result.is_error)
        self.assertIn("confirmed", result.content[0].text)
        cron.assert_not_called()

    async def test_cron_create_defaults_to_local_delivery_and_forwards_safe_fields(self):
        capabilities = HermesCapabilities()
        with patch(
            "hermes_mcp_gateway.capabilities.cronjob",
            return_value=json.dumps({"success": True, "job": {"id": "j1"}}),
        ) as cron:
            result = await capabilities.call(
                "cronjob",
                {
                    "action": "create",
                    "schedule": "0 9 * * *",
                    "prompt": "daily summary",
                    "name": "summary",
                    "confirmed": True,
                },
            )
        self.assertFalse(result.is_error)
        cron.assert_called_once_with(
            action="create",
            job_id=None,
            prompt="daily summary",
            schedule="0 9 * * *",
            name="summary",
            repeat=None,
            deliver="local",
            include_disabled=True,
            skill=None,
            skills=None,
            context_from=None,
            continuity=None,
            workdir=None,
            attach_to_session=None,
        )

    async def test_cron_rejects_hidden_unsafe_arguments_even_if_manually_sent(self):
        capabilities = HermesCapabilities()
        result = await capabilities.call(
            "cronjob",
            {
                "action": "create",
                "schedule": "1h",
                "prompt": "x",
                "script": "/tmp/x",
                "confirmed": True,
            },
        )
        self.assertTrue(result.is_error)
        self.assertIn("script", result.content[0].text)


class DelegationCapabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_delegate_schema_is_leaf_only_and_caps_batch_at_two(self):
        schemas = {tool.name: tool for tool in capability_tool_schemas()}
        self.assertIn("delegate_task", schemas)
        props = schemas["delegate_task"].input_schema["properties"]
        self.assertEqual(props["tasks"]["maxItems"], 2)
        for forbidden in {"background", "role", "max_iterations", "action", "subagent_id"}:
            self.assertNotIn(forbidden, props)
        self.assertIn("confirmed", props)

    async def test_delegate_requires_confirmation_before_spawning(self):
        capabilities = HermesCapabilities()
        with patch("hermes_mcp_gateway.capabilities.delegate_task") as delegate:
            result = await capabilities.call("delegate_task", {"goal": "Research auth design"})
        self.assertTrue(result.is_error)
        self.assertIn("confirmed", result.content[0].text)
        delegate.assert_not_called()

    async def test_delegate_runs_synchronously_as_leaf_and_closes_parent(self):
        capabilities = HermesCapabilities()
        parent = MagicMock()
        with patch(
            "hermes_mcp_gateway.capabilities._build_delegation_parent",
            return_value=parent,
        ), patch(
            "hermes_mcp_gateway.capabilities.delegate_task",
            return_value=json.dumps({"success": True, "results": [{"result": "done"}]}),
        ) as delegate:
            result = await capabilities.call(
                "delegate_task",
                {
                    "goal": "Research auth design",
                    "context": "Read-only research",
                    "confirmed": True,
                },
            )
        self.assertFalse(result.is_error)
        delegate.assert_called_once_with(
            goal="Research auth design",
            context="Read-only research",
            tasks=None,
            role="leaf",
            background=False,
            parent_agent=parent,
        )
        parent.close.assert_called_once_with()

    async def test_delegate_rejects_more_than_two_batch_tasks(self):
        capabilities = HermesCapabilities()
        tasks = [{"goal": f"Task {i} with enough detail"} for i in range(3)]
        result = await capabilities.call("delegate_task", {"tasks": tasks, "confirmed": True})
        self.assertTrue(result.is_error)
        self.assertIn("2", result.content[0].text)

    async def test_delegate_rejects_background_override(self):
        capabilities = HermesCapabilities()
        result = await capabilities.call(
            "delegate_task", {"goal": "Research auth design", "background": True, "confirmed": True}
        )
        self.assertTrue(result.is_error)
        self.assertIn("background", result.content[0].text)


class DelegationParentPolicyTests(unittest.TestCase):
    def test_parent_enables_only_safe_native_toolsets_plus_context_mode_mcp(self):
        from hermes_mcp_gateway.capabilities import _build_delegation_parent

        parent = MagicMock()
        runtime = {
            "provider": "test-provider",
            "requested_provider": "test-provider",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
            "api_key": "not-printed",
            "credential_pool": None,
        }
        with patch(
            "hermes_cli.config.load_config",
            return_value={"model": {"default": "test-model"}},
        ), patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime
        ), patch(
            "tools.mcp_tool.discover_mcp_tools", return_value=["mcp__context_mode__ctx_execute"]
        ) as discover, patch(
            "hermes_cli.fallback_config.get_fallback_chain", return_value=[]
        ), patch("run_agent.AIAgent", return_value=parent) as agent_cls:
            built = _build_delegation_parent()
        self.assertIs(built, parent)
        discover.assert_called_once()
        kwargs = agent_cls.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["provider"], "test-provider")
        self.assertEqual(
            kwargs["enabled_toolsets"],
            ["web", "vision", "skills", "mcp-context-mode"],
        )
        self.assertTrue(kwargs["skip_memory"])
        self.assertTrue(kwargs["skip_context_files"])


if __name__ == "__main__":
    unittest.main()
