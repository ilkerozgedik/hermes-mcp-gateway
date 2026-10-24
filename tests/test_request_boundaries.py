from __future__ import annotations

import asyncio
import json
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp import types

from hermes_mcp_gateway.server import Gateway, build_catalog, create_app
from hermes_mcp_gateway.sessions import SessionCapabilityStore
from hermes_mcp_gateway.tool_discovery import ToolDiscovery


RECOVERY_TOKEN = "A" * 43
JOB_ID = "job_0123456789abcdef0123456789abcdef"


def tool(name: str) -> types.Tool:
    return types.Tool(
        name=name,
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
    )


def gateway_tool(name: str) -> types.Tool:
    return types.Tool(
        name=name,
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )


def job_tool(name: str) -> types.Tool:
    credential = {"type": "string", "minLength": 43, "maxLength": 43}
    if name == "job_prepare":
        properties = {
            "command": {"type": "string", "minLength": 1, "maxLength": 8192},
            "workdir": {"type": "string", "minLength": 1, "maxLength": 1024},
            "max_runtime": {
                "type": "integer",
                "minimum": 1,
                "maximum": 7200,
                "default": 1800,
            },
            "idempotency_key": {"type": "string", "minLength": 16, "maxLength": 128},
        }
        required = ["command", "workdir"]
    else:
        properties = {
            "job_id": {"type": "string", "pattern": r"^job_[a-f0-9]{32}$"},
            "recovery_token": credential,
        }
        required = ["job_id", "recovery_token"]
    return types.Tool(
        name=name,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


class CapturedServer:
    def __init__(self, *_args, **kwargs):
        self.handler = kwargs["on_call_tool"]
        self.routes = []

    def streamable_http_app(self, **kwargs):
        self.routes = kwargs["custom_starlette_routes"]
        return self


def captured_app(gateway: Gateway):
    with (
        patch("hermes_mcp_gateway.server.Gateway", return_value=gateway),
        patch("hermes_mcp_gateway.server.Server", CapturedServer),
    ):
        app = create_app()
    return app


def call_params(name: str, arguments: dict, *, session_token: str | None = None):
    payload = {"calls": [{"name": name, "arguments": arguments}]}
    if session_token is not None:
        payload["session_token"] = session_token
    return SimpleNamespace(name="tool_call", arguments=payload)


def request_context(headers: dict | None = None):
    return SimpleNamespace(
        request=SimpleNamespace(headers=headers or {}),
        session=SimpleNamespace(),
    )


class JobRequestBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def make_gateway(self, *names: str) -> Gateway:
        gateway = Gateway()
        gateway.catalog = build_catalog(
            [],
            [],
            [job_tool(name) if name.startswith("job_") else gateway_tool(name) for name in names],
        )
        gateway.discovery = ToolDiscovery(gateway.catalog)
        return gateway

    async def test_job_prepare_requires_verified_session_and_issues_credential_unredacted(self):
        gateway = self.make_gateway("job_prepare", "job_start")
        gateway.jobs = AsyncMock()
        gateway.jobs.call.return_value = {
            "job_id": JOB_ID,
            "state": "prepared",
            "recovery_token": RECOVERY_TOKEN,
        }

        denied = await gateway.call(
            "job_prepare",
            {"command": "compile", "workdir": "/allowed"},
        )
        self.assertTrue(denied.is_error)
        gateway.jobs.call.assert_not_awaited()

        token, _record = gateway.session_capabilities.issue()
        app = captured_app(gateway)
        with patch(
            "hermes_mcp_gateway.server.normalize_result",
            side_effect=AssertionError("credential response must bypass output redaction"),
        ):
            result = await app.handler(
                request_context(),
                call_params(
                    "job_prepare",
                    {"command": "compile", "workdir": "/allowed"},
                    session_token=token,
                ),
            )

        self.assertFalse(result.is_error)
        self.assertIn(RECOVERY_TOKEN, result.content[0].text)
        gateway.jobs.call.assert_awaited_once()
        name, arguments = gateway.jobs.call.await_args.args
        self.assertEqual(name, "job_prepare")
        self.assertEqual(arguments, {"command": "compile", "workdir": "/allowed"})
        self.assertEqual(
            gateway.jobs.call.await_args.kwargs["caller_id"],
            next(iter(gateway.session_capabilities._by_owner)),
        )

    async def test_invalid_session_identity_never_authorizes_job_start(self):
        gateway = self.make_gateway("job_prepare", "job_start")
        gateway.jobs = AsyncMock()
        app = captured_app(gateway)

        for name in ("job_prepare", "job_start"):
            with self.subTest(name=name):
                arguments = (
                    {"command": "compile", "workdir": "/allowed"}
                    if name == "job_prepare"
                    else {"job_id": JOB_ID, "recovery_token": RECOVERY_TOKEN}
                )
                result = await app.handler(
                    request_context(),
                    call_params(name, arguments, session_token="invalid"),
                )
                self.assertTrue(result.is_error)
        gateway.jobs.call.assert_not_awaited()

    async def test_expired_session_token_is_ignored_only_for_credential_recovery(self):
        gateway = self.make_gateway("job_prepare", "job_start", "job_status", "job_cancel")
        now = [10.0]
        gateway.session_capabilities = SessionCapabilityStore(
            ttl_seconds=5, clock=lambda: now[0]
        )
        old_token, old_record = gateway.session_capabilities.issue()
        now[0] = 15.0
        gateway._cleanup_session_records = AsyncMock()
        authorized = []

        async def job_call(name, arguments, *, caller_id=None):
            if arguments.get("recovery_token") != RECOVERY_TOKEN:
                raise PermissionError("invalid job credential")
            authorized.append((name, caller_id))
            return {"job_id": arguments["job_id"], "state": "running"}

        gateway.jobs = SimpleNamespace(call=AsyncMock(side_effect=job_call))
        app = captured_app(gateway)

        for name in ("job_status", "job_cancel"):
            result = await app.handler(
                request_context(),
                call_params(
                    name,
                    {"job_id": JOB_ID, "recovery_token": RECOVERY_TOKEN},
                    session_token=old_token,
                ),
            )
            self.assertFalse(result.is_error, result.content[0].text)

        denied = await app.handler(
            request_context(),
            call_params(
                "job_start",
                {"job_id": JOB_ID, "recovery_token": RECOVERY_TOKEN},
                session_token=old_token,
            ),
        )
        self.assertTrue(denied.is_error)
        self.assertEqual(authorized, [("job_status", None), ("job_cancel", None)])
        self.assertEqual(gateway.jobs.call.await_count, 2)
        gateway._cleanup_session_records.assert_awaited_once_with([old_record])

    async def test_wrong_job_credential_does_not_authorize_status_or_cancel(self):
        gateway = self.make_gateway("job_status", "job_cancel")

        async def job_call(_name, _arguments, *, caller_id=None):
            raise PermissionError("invalid job credential")

        gateway.jobs = SimpleNamespace(call=AsyncMock(side_effect=job_call))
        app = captured_app(gateway)
        for name in ("job_status", "job_cancel"):
            result = await app.handler(
                request_context(),
                call_params(
                    name,
                    {"job_id": JOB_ID, "recovery_token": "B" * 43},
                    session_token="invalid-session-token",
                ),
            )
            self.assertTrue(result.is_error)
        self.assertEqual(gateway.jobs.call.await_count, 2)


class RequestTelemetryTests(unittest.IsolatedAsyncioTestCase):
    def make_gateway(self) -> Gateway:
        gateway = Gateway()
        gateway.catalog = build_catalog([tool("web_search")], [], [])
        gateway.discovery = ToolDiscovery(gateway.catalog)
        return gateway

    def test_info_telemetry_has_an_output_handler(self):
        logger = logging.getLogger("hermes_mcp_gateway")
        self.assertEqual(logger.level, logging.INFO)
        self.assertTrue(logger.handlers or logging.getLogger().handlers)

    async def test_inflight_count_is_visible_on_cheap_healthz_and_returns_to_zero(self):
        gateway = self.make_gateway()
        started = asyncio.Event()
        release = asyncio.Event()

        async def pending_call(*_args, **_kwargs):
            started.set()
            await release.wait()
            return types.CallToolResult(content=[types.TextContent(text="done")])

        gateway.call = AsyncMock(side_effect=pending_call)
        gateway.health = AsyncMock(side_effect=AssertionError("healthz must not probe upstreams"))
        app = captured_app(gateway)
        request = asyncio.create_task(
            app.handler(
                request_context(),
                call_params("web_search", {"query": "private query"}),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        self.assertEqual(gateway.inflight_requests, 1)

        healthz = next(route.endpoint for route in app.routes if route.path == "/healthz")
        response = await healthz(None)
        self.assertEqual(
            json.loads(response.body),
            {"status": "ok", "inflight_requests": 1},
        )
        gateway.health.assert_not_awaited()

        release.set()
        result = await request
        self.assertFalse(result.is_error)
        self.assertEqual(gateway.inflight_requests, 0)

    async def test_request_log_uses_safe_correlation_and_never_logs_payload_or_error(self):
        gateway = self.make_gateway()
        gateway.hermes.call = AsyncMock(
            side_effect=RuntimeError("OPENAI_API_KEY=secret-marker command=private-command")
        )
        app = captured_app(gateway)

        token, _record = gateway.session_capabilities.issue()
        with self.assertLogs("hermes_mcp_gateway", level=logging.INFO) as captured:
            result = await app.handler(
                request_context({"x-request-id": "bad\r\nAuthorization: Bearer ***"}),
                call_params(
                    "web_search", {"query": "private-query-value"}, session_token=token
                ),
            )
        self.assertTrue(result.is_error)
        output = "\n".join(captured.output)
        self.assertIn("tool=web_search", output)
        self.assertIn("result=tool_error", output)
        self.assertIn("correlation_id=none", output)
        for secret in (
            "secret-marker",
            "private-command",
            "private-query-value",
            token,
            "Authorization",
            "Bearer",
            "***",
        ):
            self.assertNotIn(secret, output)

        gateway.hermes.call = AsyncMock(
            return_value=types.CallToolResult(content=[types.TextContent(text="ok")])
        )
        with self.assertLogs("hermes_mcp_gateway", level=logging.INFO) as safe_log:
            await app.handler(
                request_context({"x-request-id": "safe-id:42"}),
                call_params("web_search", {"query": "another private query"}),
            )
        safe_output = "\n".join(safe_log.output)
        self.assertIn("correlation_id=safe-id:42", safe_output)
        self.assertNotIn("another private query", safe_output)

    async def test_timeout_is_logged_as_timeout_not_tool_error(self):
        gateway = self.make_gateway()
        gateway.hermes.call = AsyncMock(side_effect=TimeoutError("timeout-private-detail"))
        app = captured_app(gateway)

        with self.assertLogs("hermes_mcp_gateway", level=logging.INFO) as captured:
            result = await app.handler(
                request_context(), call_params("web_search", {"query": "x"})
            )
        self.assertTrue(result.is_error)
        self.assertEqual(
            result.structured_content,
            {"_gateway_error_category": "timeout"},
        )
        output = "\n".join(captured.output)
        self.assertIn("result=timeout", output)
        self.assertNotIn("timeout-private-detail", output)
        self.assertEqual(gateway.inflight_requests, 0)

    async def test_cancelled_request_is_logged_cancelled_and_decrements_counter(self):
        gateway = self.make_gateway()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending_call(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        gateway.call = AsyncMock(side_effect=pending_call)
        app = captured_app(gateway)
        with self.assertLogs("hermes_mcp_gateway", level=logging.INFO) as captured:
            request = asyncio.create_task(
                app.handler(request_context(), call_params("web_search", {"query": "x"}))
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            self.assertEqual(gateway.inflight_requests, 1)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            await asyncio.wait_for(cancelled.wait(), timeout=1)

        output = "\n".join(captured.output)
        self.assertIn("result=cancelled", output)
        self.assertNotIn("result=success", output)
        self.assertEqual(gateway.inflight_requests, 0)


if __name__ == "__main__":
    unittest.main()
