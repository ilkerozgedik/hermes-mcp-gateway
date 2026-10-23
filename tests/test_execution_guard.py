"""Fail-closed Gateway guard for ad hoc command execution."""
import json
import unittest
from unittest.mock import AsyncMock

from mcp import types

from hermes_mcp_gateway.config import GatewayConfig
from hermes_mcp_gateway.server import Gateway, build_catalog


def tool(name):
    properties = {"timeout": {"type": "integer", "minimum": 1, "default": 180}} if name == "terminal" else {}
    return types.Tool(name=name, description=name, input_schema={"type":"object","properties":properties})


class ExecutionGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gateway = Gateway()
        self.gateway.catalog = build_catalog([tool("terminal"), tool("execute_code")], [])
        self.gateway.hermes.call_direct = AsyncMock(return_value=types.CallToolResult(content=[types.TextContent(text='{"output":"ok","exit_code":0}')]))
        self.gateway.execution.call = AsyncMock(return_value=types.CallToolResult(content=[types.TextContent(text='{"status":"ok"}')]))

    async def asyncTearDown(self):
        await self.gateway.execution.close()

    def test_terminal_schema_discloses_guard(self):
        schema = self.gateway.catalog["terminal"].tool
        self.assertIn("job_start", schema.description or "")
        self.assertEqual(schema.input_schema["properties"]["timeout"]["maximum"], 30)
        self.assertEqual(schema.input_schema["properties"]["timeout"]["default"], 30)

    async def test_short_terminal_uses_native_timeout_even_if_omitted(self):
        result = await self.gateway.call("terminal", {"command":"printf short"}, task_id="session-1")
        self.assertFalse(result.is_error)
        args = self.gateway.hermes.call_direct.call_args.args
        self.assertEqual(args[0], "terminal")
        self.assertEqual(args[1]["timeout"], 30)

    async def test_short_explicit_timeout_preserved(self):
        await self.gateway.call("terminal", {"command":"pwd","timeout":10}, task_id="session-1")
        self.assertEqual(self.gateway.hermes.call_direct.call_args.args[1]["timeout"], 10)

    async def test_long_foreground_is_rejected_before_dispatch(self):
        result = await self.gateway.call("terminal", {"command":"pnpm test","timeout":600}, task_id="session-1")
        self.assertTrue(result.is_error)
        self.assertIn("LONG_RUNNING_OPERATION_REQUIRES_JOB", result.content[0].text)
        self.gateway.hermes.call_direct.assert_not_called()

    async def test_direct_background_cannot_bypass_job_authorization(self):
        for options in ({"background":True},{"background":True,"persist_on_release":True},{"pty":True}):
            with self.subTest(options=options):
                result = await self.gateway.call("terminal", {"command":"pnpm test",**options}, task_id="session-1")
                self.assertTrue(result.is_error)
                self.assertIn("LONG_RUNNING_OPERATION_REQUIRES_JOB", result.content[0].text)
        self.gateway.hermes.call_direct.assert_not_called()

    async def test_invalid_timeout_fails_closed(self):
        for timeout in (0,-1,31,True,"30"):
            with self.subTest(timeout=timeout):
                result = await self.gateway.call("terminal", {"command":"pwd","timeout":timeout}, task_id="session-1")
                self.assertTrue(result.is_error)
        self.gateway.hermes.call_direct.assert_not_called()

    async def test_reserved_background_only_flags_rejected(self):
        for options in ({"notify":True},{"persist_on_release":True},{"heartbeat":60}):
            result = await self.gateway.call("terminal", {"command":"pwd",**options}, task_id="session-1")
            self.assertTrue(result.is_error)
        self.gateway.hermes.call_direct.assert_not_called()

    async def test_execute_code_timeout_is_short_and_worker_cleanup_kept(self):
        self.assertEqual(GatewayConfig().execution_timeout_seconds, 30)
        result = await self.gateway.call("execute_code", {"code":"print(1)"})
        self.assertFalse(result.is_error)
        self.gateway.execution.call.assert_awaited_once()

    async def test_job_start_uses_separate_gateway_capability_and_is_not_affected(self):
        from hermes_mcp_gateway.jobs import job_tool_schemas
        self.gateway.catalog = build_catalog([], [], job_tool_schemas())
        self.gateway.jobs = AsyncMock()
        self.gateway.jobs.call.return_value = {"job_id":"job_a", "state":"running", "recovery_token":"x" * 43}
        result = await self.gateway.call("job_start", {"command":"pnpm test","workdir":"/home/hermes/work"}, task_id="session-1")
        self.assertFalse(result.is_error)
        self.gateway.jobs.call.assert_awaited_once()
