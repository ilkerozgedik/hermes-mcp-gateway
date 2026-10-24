"""Regressions for native work surviving request cancellation and timeout."""
from __future__ import annotations

import asyncio
import json
import threading
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from test_request_boundaries import call_params, captured_app, request_context, tool

from hermes_mcp_gateway import memory as memory_module
from hermes_mcp_gateway.memory import MemoryAdapter
from hermes_mcp_gateway.native_activity import NativeCalls
from hermes_mcp_gateway.server import Gateway, build_catalog
from hermes_mcp_gateway.sessions import session_close_tool_schema
from hermes_mcp_gateway.tool_discovery import ToolDiscovery


async def until(predicate):
    deadline = asyncio.get_running_loop().time() + 3
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Lifecycle state did not reach the expected value")
        await asyncio.sleep(0.01)


class MemoryNativeActivityTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_timeout_keeps_shared_native_work_counted(self):
        calls = NativeCalls()
        adapter = MemoryAdapter(timeout=0.03, native_calls=calls)
        started, release = threading.Event(), threading.Event()

        class Config:
            enabled = True
            api_key = "test-key"
            base_url = ""
            context_tokens = 256

            @staticmethod
            def resolve_session_name(**_kwargs):
                return "test-session"

        class Manager:
            def __init__(self, **_kwargs):
                pass

            def get_or_create(self, _session_key):
                started.set()
                release.wait(3)

        try:
            with (
                patch.object(
                    memory_module.HonchoClientConfig,
                    "from_global_config",
                    return_value=Config(),
                ),
                patch.object(memory_module, "resolve_memory_config", side_effect=lambda cfg, **_kw: cfg),
                patch.object(memory_module, "get_honcho_client", return_value=object()),
                patch.object(memory_module, "HonchoSessionManager", Manager),
            ):
                with self.assertRaises(TimeoutError):
                    await adapter.start()
                self.assertTrue(started.is_set())
                self.assertEqual(calls.active_count, 1)
                release.set()
                await until(lambda: calls.active_count == 0)
        finally:
            release.set()

    async def test_memory_write_timeout_reports_unknown_and_native_drain_state(self):
        gateway = Gateway()
        gateway.catalog = build_catalog([], [tool("memory_conclude")], [])
        gateway.discovery = ToolDiscovery(gateway.catalog)
        gateway.memory.timeout = 0.03
        started, release = threading.Event(), threading.Event()
        conclusion = "Private durable preference fixture"

        def create_conclusion(*_args, **_kwargs):
            started.set()
            release.wait(3)
            return True

        gateway.memory.manager = SimpleNamespace(create_conclusion=create_conclusion)
        gateway.memory.session_key = "test-session"
        app = cast(Any, captured_app(gateway))
        try:
            with self.assertLogs("hermes_mcp_gateway", level="INFO") as captured:
                result = await app.handler(
                    request_context(),
                    call_params(
                        "memory_conclude",
                        {"conclusion": conclusion, "kind": "preference"},
                    ),
                )

            self.assertTrue(started.is_set())
            self.assertEqual(gateway.inflight_requests, 0)
            self.assertEqual(gateway.native_calls.active_count, 1)
            payload = json.loads(result.content[0].text)
            self.assertEqual(payload["state"], "unknown")
            self.assertTrue(payload["readback_required"])
            self.assertIn("absence on an early readback", payload["message"])
            self.assertIn("original work has finished", payload["message"])
            self.assertIn("readback", payload["message"])
            log_output = "\n".join(captured.output)
            self.assertNotIn(conclusion, log_output)

            release.set()
            await until(lambda: gateway.native_calls.active_count == 0)
        finally:
            release.set()
            await gateway.execution.close()


class NativeSessionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_waits_for_cancelled_native_work_and_owner_cleanup(self):
        gateway = Gateway()
        token, record = gateway.session_capabilities.issue()
        request_lease = gateway.session_capabilities.acquire(token)
        assert request_lease is not None
        self.assertIs(request_lease, record)
        worker_started, release_worker = threading.Event(), threading.Event()
        cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event()
        close_order = []

        def native_work():
            worker_started.set()
            release_worker.wait(3)

        async def cleanup_owner(_task_id):
            cleanup_started.set()
            await release_cleanup.wait()
            close_order.append("owner_cleanup")

        async def close_execution():
            close_order.append("execution")

        async def close_hermes():
            close_order.append("hermes")

        async def close_graph():
            close_order.append("graph")

        def close_lsp():
            close_order.append("lsp")

        request = None
        closing = None
        try:
            with (
                patch.object(gateway, "_cleanup_session_owner", side_effect=cleanup_owner),
                patch.object(gateway.execution, "close", side_effect=close_execution),
                patch.object(gateway.hermes, "close", side_effect=close_hermes),
                patch.object(gateway.graph, "close", side_effect=close_graph),
                patch.object(gateway.lsp, "close", side_effect=close_lsp),
            ):
                try:
                    with gateway.native_calls.owner_lease(
                        retain=lambda: gateway.session_capabilities.retain(request_lease),
                        release=lambda: gateway._release_native_session_token(request_lease),
                    ):
                        request = asyncio.create_task(gateway.native_calls.run(native_work))
                    await until(worker_started.is_set)
                    self.assertEqual(gateway.native_calls.active_count, 1)
                    request.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await request
                    self.assertEqual(record.active_calls, 2)

                    await gateway.release_session_token(request_lease)
                    self.assertEqual(record.active_calls, 1)
                    closing = asyncio.create_task(gateway.close())
                    await until(lambda: record.revoked)
                    self.assertFalse(closing.done())

                    release_worker.set()
                    await asyncio.wait_for(cleanup_started.wait(), timeout=3)
                    self.assertFalse(closing.done())
                    self.assertEqual(close_order, [])

                    release_cleanup.set()
                    await closing
                    self.assertEqual(
                        close_order,
                        ["owner_cleanup", "execution", "hermes", "graph", "lsp"],
                    )
                    self.assertEqual(record.active_calls, 0)
                    self.assertNotIn(record.task_id, gateway.session_capabilities._by_owner)
                finally:
                    release_worker.set()
                    release_cleanup.set()
                    if request is not None and not request.done():
                        request.cancel()
                        try:
                            await request
                        except asyncio.CancelledError:
                            pass
                    if closing is not None and not closing.done():
                        await closing
        finally:
            release_worker.set()
            release_cleanup.set()

    async def test_cancelled_camofox_worker_holds_owner_until_late_binding_is_cleaned(self):
        from tools import browser_camofox

        gateway = Gateway()
        gateway.catalog = build_catalog(
            [tool("browser_navigate")], [], [session_close_tool_schema()]
        )
        gateway.discovery = ToolDiscovery(gateway.catalog)
        token, record = gateway.session_capabilities.issue()
        owner_id = record.task_id
        browser_camofox._drop_session(owner_id)
        app = cast(Any, captured_app(gateway))
        worker_started, release_worker = threading.Event(), threading.Event()
        binding_created = threading.Event()
        observed_binding = {}

        def fake_native_dispatch(_name, arguments, *, task_id, **_kwargs):
            worker_started.set()
            release_worker.wait(3)
            return browser_camofox.camofox_navigate(arguments["url"], task_id=task_id)

        def fake_post(_path, body, timeout=None):
            observed_binding["session"] = browser_camofox._sessions[owner_id]
            binding_created.set()
            return {"tabId": "late-tab", "url": body["url"]}

        request = asyncio.create_task(
            app.handler(
                request_context(),
                call_params(
                    "browser_navigate",
                    {"url": "https://example.com"},
                    session_token=token,
                ),
            )
        )
        try:
            with (
                patch("model_tools.handle_function_call", side_effect=fake_native_dispatch),
                patch.object(browser_camofox, "_get_camofox_config", return_value={}),
                patch.object(browser_camofox, "get_secret", side_effect=lambda _name, default="": default),
                patch.object(browser_camofox, "_post", side_effect=fake_post),
                patch.object(browser_camofox, "_fetch_snapshot", return_value=("", 0)),
                patch.object(browser_camofox, "get_vnc_url", return_value=None),
                patch.object(browser_camofox, "_drop_session", wraps=browser_camofox._drop_session) as drop_session,
                patch("tools.process_registry.process_registry.kill_all"),
                patch("tools.terminal_tool_lifecycle.cleanup_vm"),
            ):
                await until(worker_started.is_set)
                self.assertEqual(gateway.native_calls.active_count, 1)

                closed = await app.handler(
                    request_context(), call_params("session_close", {}, session_token=token)
                )
                self.assertFalse(closed.is_error)
                self.assertTrue(record.revoked)

                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request
                self.assertEqual(gateway.inflight_requests, 0)
                self.assertEqual(record.active_calls, 1)
                self.assertIn(record.task_id, gateway.session_capabilities._by_owner)
                self.assertNotIn(owner_id, browser_camofox._sessions)
                drop_session.assert_not_called()

                release_worker.set()
                await until(binding_created.is_set)
                await until(
                    lambda: gateway.native_calls.active_count == 0
                    and not gateway._session_cleanup_tasks
                )
                self.assertEqual(observed_binding["session"].get("tab_id"), "late-tab")
                self.assertNotIn(owner_id, browser_camofox._sessions)
                self.assertNotIn(record.task_id, gateway.session_capabilities._by_owner)
                drop_session.assert_called_once_with(owner_id)
        finally:
            release_worker.set()
            if not request.done():
                request.cancel()
                try:
                    await request
                except asyncio.CancelledError:
                    pass
            await until(lambda: gateway.native_calls.active_count == 0)
            if gateway._session_cleanup_tasks:
                await asyncio.gather(*tuple(gateway._session_cleanup_tasks))
            browser_camofox._drop_session(owner_id)
            await gateway.execution.close()


if __name__ == "__main__":
    unittest.main()
