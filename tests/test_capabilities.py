import json
import unittest
from unittest.mock import AsyncMock, patch

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
    async def test_cron_list_forwards_to_hermes_including_disabled(self):
        capabilities = HermesCapabilities()
        native = json.dumps({"success": True, "count": 2, "jobs": [
            {"job_id": "one", "enabled": True},
            {"job_id": "two", "enabled": False},
        ]})
        with patch("hermes_mcp_gateway.capabilities.cronjob", return_value=native) as cron:
            result = await capabilities.call(
                "cronjob", {"action": "list", "include_disabled": True}
            )
        self.assertFalse(result.is_error)
        self.assertEqual(result.content[0].text, native)
        self.assertEqual(cron.call_args.kwargs["action"], "list")
        self.assertTrue(cron.call_args.kwargs["include_disabled"])

    def test_cron_schema_hides_unsafe_execution_and_provider_fields(self):
        schemas = {tool.name: tool for tool in capability_tool_schemas()}
        self.assertIn("cronjob", schemas)
        props = schemas["cronjob"].input_schema["properties"]
        for forbidden in {
            "model", "provider", "base_url", "script", "no_agent",
            "monitor_script", "monitor_url", "enabled_toolsets",
        }:
            self.assertNotIn(forbidden, props)
        self.assertNotIn("confirmed", props)

    async def test_cron_mutation_without_redundant_confirmation(self):
        capabilities = HermesCapabilities()
        with patch("hermes_mcp_gateway.capabilities.cronjob", return_value=json.dumps({"success": True})) as cron:
            result = await capabilities.call(
                "cronjob",
                {"action": "create", "schedule": "0 9 * * *", "prompt": "daily summary"},
            )
        self.assertFalse(result.is_error)
        cron.assert_called_once()

    async def test_cron_legacy_confirmation_is_not_a_supported_argument(self):
        capabilities = HermesCapabilities()
        with patch('hermes_mcp_gateway.capabilities.cronjob') as cron:
            result = await capabilities.call('cronjob', {'action': 'create', 'confirmed': True})
        self.assertTrue(result.is_error)
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
        self.assertNotIn("confirmed", props)

    async def test_delegate_fails_fast_without_starting_model_inference(self):
        capabilities = HermesCapabilities()
        with patch(
            "hermes_mcp_gateway.capabilities.NativeCalls.run",
            new_callable=AsyncMock,
        ) as to_thread:
            result = await capabilities.call("delegate_task", {"goal": "Research auth design"})
        self.assertTrue(result.is_error)
        self.assertIn("no enforceable deadline", result.content[0].text)
        self.assertIn("native Hermes delegate_task", result.content[0].text)
        to_thread.assert_not_awaited()

    async def test_delegate_legacy_confirmation_is_not_a_supported_argument(self):
        capabilities = HermesCapabilities()
        with patch(
            "hermes_mcp_gateway.capabilities.NativeCalls.run",
            new_callable=AsyncMock,
        ) as to_thread:
            result = await capabilities.call('delegate_task', {'goal': 'Research auth design', 'confirmed': True})
        self.assertTrue(result.is_error)
        to_thread.assert_not_awaited()

    async def test_delegate_batch_fails_fast_without_starting_model_inference(self):
        capabilities = HermesCapabilities()
        with patch(
            "hermes_mcp_gateway.capabilities.NativeCalls.run",
            new_callable=AsyncMock,
        ) as to_thread:
            result = await capabilities.call(
                "delegate_task",
                {
                    "tasks": [
                        {"goal": "Research auth design", "context": "Read-only research"},
                        {"goal": "Review API constraints"},
                    ],
                },
            )
        self.assertTrue(result.is_error)
        self.assertIn("no enforceable deadline", result.content[0].text)
        to_thread.assert_not_awaited()

    async def test_delegate_rejects_more_than_two_batch_tasks(self):
        capabilities = HermesCapabilities()
        tasks = [{"goal": f"Task {i} with enough detail"} for i in range(3)]
        result = await capabilities.call("delegate_task", {"tasks": tasks})
        self.assertTrue(result.is_error)
        self.assertIn("2", result.content[0].text)

    async def test_delegate_rejects_background_override(self):
        capabilities = HermesCapabilities()
        result = await capabilities.call(
            "delegate_task", {"goal": "Research auth design", "background": True}
        )
        self.assertTrue(result.is_error)
        self.assertIn("background", result.content[0].text)


if __name__ == "__main__":
    unittest.main()
