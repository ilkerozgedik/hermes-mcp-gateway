"""Modern HTTP is sessionless; only persistent resources need transport ownership."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import uvicorn
from mcp import types

from hermes_mcp_gateway.server import (
    Gateway,
    browser_task_id,
    build_catalog,
    create_app,
)
from hermes_mcp_gateway.tool_discovery import ToolDiscovery

PROTOCOL = "2026-07-28"


def tool(name):
    return types.Tool(
        name=name, input_schema={"type": "object", "additionalProperties": True}
    )


def success():
    return types.CallToolResult(content=[types.TextContent(text="executed")])


def rpc_payload(response):
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        return json.loads(
            next(
                line[6:]
                for line in response.text.splitlines()
                if line.startswith("data: ")
            )
        )
    return response.json()


class StatelessHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.observations = []
        self.browser_owners = []

        async def fake_start(gateway):
            self.gateway = gateway
            gateway.catalog = build_catalog(
                [tool(name) for name in ("ctx_doctor", "ctx_search", "ctx_execute")],
                [
                    tool(name)
                    for name in (
                        "skills_list",
                        "skill_view",
                        "browser_click",
                        "terminal",
                        "process_manage",
                        "read_file",
                    )
                ],
                [tool("memory_context")],
                [tool("startup_context")],
            )
            gateway.discovery = ToolDiscovery(gateway.catalog)
            gateway.context.call = AsyncMock(return_value=success())
            gateway.hermes.call = AsyncMock(return_value=success())
            gateway.hermes.call_direct = AsyncMock(return_value=success())

            async def browser(_name, _arguments, *, task_id):
                self.browser_owners.append(task_id)
                return success()

            gateway.hermes.call_browser = AsyncMock(side_effect=browser)

        async def fake_close(gateway):
            await gateway.context.close()

        # Temporary observation at the real SDK handler boundary: only safe metadata.
        from hermes_mcp_gateway.server import Server

        def observed_server(*args, **kwargs):
            handler = kwargs["on_call_tool"]

            async def observe(ctx, params):
                self.observations.append(
                    {
                        "protocol": ctx.protocol_version,
                        "header_present": bool(
                            getattr(ctx.request, "headers", {}).get("mcp-session-id")
                        ),
                        "sdk_session_present": bool(ctx.session._connection.session_id),
                    }
                )
                return await handler(ctx, params)

            kwargs["on_call_tool"] = observe
            return Server(*args, **kwargs)

        self.patches = [
            patch.object(Gateway, "start", fake_start),
            patch.object(Gateway, "close", fake_close),
            patch("hermes_mcp_gateway.server.Server", observed_server),
            # Each test server models a fresh process; sse-starlette keeps its
            # shutdown flag process-global across Uvicorn instances/tests.
            patch("sse_starlette.sse.AppStatus.should_exit", False),
        ]
        for item in self.patches:
            item.start()
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(),
                host="127.0.0.1",
                port=0,
                log_level="error",
                access_log=False,
                lifespan="on",
            )
        )
        self.runner = asyncio.create_task(self.server.serve())
        for _ in range(200):
            if self.server.started and self.server.servers:
                break
            await asyncio.sleep(0.01)
        self.assertTrue(self.server.started)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=5
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.server.should_exit = True
        await asyncio.wait_for(self.runner, timeout=8)
        for item in reversed(self.patches):
            item.stop()

    async def modern_call(self, name, arguments=None, *, extra_headers=None):
        response = await self.client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": PROTOCOL,
                "Mcp-Method": "tools/call",
                "Mcp-Name": "tool_call",
                **(extra_headers or {}),
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "tool_call",
                    "arguments": {
                        "calls": [{"name": name, "arguments": arguments or {}}]
                    },
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": PROTOCOL,
                        "io.modelcontextprotocol/clientInfo": {
                            "name": "chatgpt-like-test",
                            "version": "1",
                        },
                        "io.modelcontextprotocol/clientCapabilities": {},
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("mcp-session-id", response.headers)
        return types.CallToolResult.model_validate(rpc_payload(response)["result"])

    async def assert_stateless_executes(self, name, arguments=None):
        result = await self.modern_call(name, arguments)
        self.assertEqual(
            self.observations[-1],
            {
                "protocol": PROTOCOL,
                "header_present": False,
                "sdk_session_present": False,
            },
        )
        self.assertFalse(result.is_error, result.content[0].text)
        self.assertEqual(result.content[0].text, "executed")

    async def test_sessionless_skill_view(self):
        await self.assert_stateless_executes("skill_view", {"name": "hermes-agent"})
        self.gateway.hermes.call.assert_awaited_once()

    async def test_sessionless_ctx_doctor(self):
        await self.assert_stateless_executes("ctx_doctor")
        self.gateway.context.call.assert_awaited_once()

    async def test_sessionless_ctx_search(self):
        await self.assert_stateless_executes("ctx_search", {"queries": ["needle"]})
        self.gateway.context.call.assert_awaited_once()

    async def test_sessionless_independent_ctx_execute(self):
        await self.assert_stateless_executes(
            "ctx_execute", {"language": "python", "code": "print('ok')"}
        )
        self.gateway.context.call.assert_awaited_once()

    async def test_sessionless_native_file_tool(self):
        await self.assert_stateless_executes("read_file", {"path": "README.md"})
        self.gateway.hermes.call_direct.assert_awaited_once()

    async def test_sessionless_stateful_tools_fail_closed(self):
        for name, arguments in [
            ("browser_click", {"ref": "@e1"}),
            ("terminal", {"command": "printf ok"}),
            ("process_manage", {"action": "list"}),
        ]:
            with self.subTest(name=name):
                result = await self.modern_call(name, arguments)
                self.assertTrue(result.is_error)
        self.gateway.hermes.call_browser.assert_not_awaited()
        self.gateway.hermes.call_direct.assert_not_awaited()

    async def test_modern_session_header_cannot_claim_resource_ownership(self):
        result = await self.modern_call(
            "browser_click",
            {"ref": "@e1"},
            extra_headers={"Mcp-Session-Id": "client-forged-owner"},
        )
        self.assertTrue(result.is_error)
        self.assertFalse(self.observations[-1]["sdk_session_present"])
        self.gateway.hermes.call_browser.assert_not_awaited()

    async def test_sessionless_concurrent_tools_execute_without_shared_owner(self):
        results = await asyncio.gather(
            self.modern_call("ctx_doctor"),
            self.modern_call(
                "ctx_execute", {"language": "python", "code": "print('ok')"}
            ),
        )
        self.assertTrue(all(not result.is_error for result in results))
        self.assertEqual(self.gateway.context.call.await_count, 2)
        self.assertFalse(
            any(observation["sdk_session_present"] for observation in self.observations)
        )

    async def test_unknown_and_invalid_deferred_calls_still_fail_closed(self):
        result = await self.modern_call("not_allowed")
        self.assertTrue(result.is_error)
        self.gateway.context.call.assert_not_awaited()
        self.gateway.hermes.call.assert_not_awaited()

    async def legacy_initialize(self):
        response = await self.client.post(
            "/mcp",
            headers={"Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "legacy-test", "version": "1"},
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        headers = {
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": rpc_payload(response)["result"]["protocolVersion"],
            "Mcp-Session-Id": response.headers["mcp-session-id"],
        }
        initialized = await self.client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        self.assertEqual(initialized.status_code, 202)
        return headers

    async def legacy_call(self, headers, name, arguments=None):
        response = await self.client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "tool_call",
                    "arguments": {
                        "calls": [{"name": name, "arguments": arguments or {}}]
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        return types.CallToolResult.model_validate(rpc_payload(response)["result"])

    async def test_legacy_clients_keep_distinct_stable_browser_ownership(self):
        first, second = await asyncio.gather(
            self.legacy_initialize(), self.legacy_initialize()
        )
        for _ in range(2):
            results = await asyncio.gather(
                self.legacy_call(first, "browser_click"),
                self.legacy_call(second, "browser_click"),
            )
            self.assertTrue(all(not result.is_error for result in results))
        self.assertEqual(len(self.browser_owners), 4)
        self.assertTrue(self.browser_owners[0] != self.browser_owners[1])
        self.assertTrue(self.browser_owners[:2] == self.browser_owners[2:])
        self.assertTrue(
            all(observation["sdk_session_present"] for observation in self.observations)
        )

    async def test_modern_client_cannot_reuse_a_real_legacy_owner_header(self):
        headers = await self.legacy_initialize()
        result = await self.modern_call(
            "browser_click", extra_headers={"Mcp-Session-Id": headers["Mcp-Session-Id"]}
        )
        self.assertTrue(result.is_error)
        self.assertFalse(self.observations[-1]["sdk_session_present"])
        self.gateway.hermes.call_browser.assert_not_awaited()

    async def test_unknown_legacy_session_is_rejected_before_dispatch(self):
        response = await self.client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-06-18",
                "Mcp-Session-Id": "unknown-client-supplied",
            },
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "tool_call",
                    "arguments": {
                        "calls": [{"name": "browser_click", "arguments": {}}]
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.observations, [])
        self.gateway.hermes.call_browser.assert_not_awaited()

    async def test_cross_client_process_handle_denied_at_real_http_boundary(self):
        from hermes_mcp_gateway.upstreams import HermesToolsClient

        first, second = await asyncio.gather(
            self.legacy_initialize(), self.legacy_initialize()
        )
        await self.legacy_call(first, "browser_click")
        owned = SimpleNamespace(task_id="default", session_key=self.browser_owners[0])
        self.gateway.hermes.call_direct = HermesToolsClient.call_direct.__get__(
            self.gateway.hermes
        )
        with (
            patch("tools.process_registry.process_registry.get", return_value=owned),
            patch(
                "model_tools.handle_function_call", return_value='{"status":"running"}'
            ) as dispatch,
        ):
            denied = await self.legacy_call(
                second,
                "process_manage",
                {"action": "kill", "session_id": "proc_test_owned"},
            )
            self.assertTrue(denied.is_error)
            dispatch.assert_not_called()
            accepted = await self.legacy_call(
                first,
                "process_manage",
                {"action": "poll", "session_id": "proc_test_owned"},
            )
            self.assertFalse(accepted.is_error)
            dispatch.assert_called_once()

    async def test_modern_disconnect_cancels_downstream_call(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def blocking_call(_name, _arguments):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.gateway.context.call = AsyncMock(side_effect=blocking_call)
        request = asyncio.create_task(
            self.modern_call(
                "ctx_execute", {"language": "python", "code": "print('ok')"}
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            await asyncio.wait_for(cancelled.wait(), timeout=3)
        finally:
            request.cancel()


class OwnershipPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_identity_uses_validated_sdk_connection_not_raw_header(self):
        ctx = SimpleNamespace(
            request=SimpleNamespace(headers={"mcp-session-id": "client-claimed"}),
            session=SimpleNamespace(
                _connection=SimpleNamespace(session_id="sdk-owned")
            ),
        )
        self.assertEqual(browser_task_id(ctx), "chatgpt:sdk-owned")

    def test_raw_header_alone_is_not_a_verified_identity(self):
        ctx = SimpleNamespace(
            request=SimpleNamespace(headers={"mcp-session-id": "client-claimed"}),
            session=SimpleNamespace(),
        )
        self.assertIsNone(browser_task_id(ctx))

    def test_no_transport_identity_is_valid_for_stateless_calls(self):
        ctx = SimpleNamespace(
            request=None,
            session=SimpleNamespace(_connection=SimpleNamespace(session_id=None)),
        )
        self.assertIsNone(browser_task_id(ctx))

    async def test_gateway_dispatch_cannot_fall_back_to_global_owner(self):
        gateway = Gateway()
        gateway.catalog = build_catalog(
            [],
            [tool(name) for name in ("browser_click", "terminal", "process_manage")],
            [],
        )
        gateway.hermes.call_browser = AsyncMock(return_value=success())
        gateway.hermes.call_direct = AsyncMock(return_value=success())
        try:
            for name in gateway.catalog:
                with self.subTest(name=name):
                    self.assertTrue((await gateway.call(name, {})).is_error)
            gateway.hermes.call_browser.assert_not_awaited()
            gateway.hermes.call_direct.assert_not_awaited()
        finally:
            await gateway.context.close()
