"""Cancellation must not hide native threads from deployment quiescence."""
import asyncio
from contextvars import ContextVar
import threading
import unittest

from hermes_mcp_gateway.native_activity import NativeCalls


async def until(predicate):
    deadline = asyncio.get_running_loop().time() + 3
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Native activity did not reach the expected state")
        await asyncio.sleep(0.01)


class NativeActivityTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_waiter_keeps_actual_thread_counted_until_it_finishes(self):
        calls = NativeCalls()
        started, release = threading.Event(), threading.Event()
        def blocking():
            started.set()
            release.wait(3)
            return "done"
        request = asyncio.create_task(calls.run(blocking))
        try:
            await until(started.is_set)
            self.assertEqual(calls.active_count, 1)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            self.assertEqual(calls.active_count, 1)
        finally:
            release.set()
            await until(lambda: calls.active_count == 0)

    async def test_contextvars_and_result_are_preserved_without_another_executor(self):
        calls = NativeCalls()
        owner = ContextVar("native_activity_owner", default="none")
        token = owner.set("verified-owner")
        try:
            self.assertEqual(await calls.run(owner.get), "verified-owner")
        finally:
            owner.reset(token)
        self.assertEqual(calls.active_count, 0)

    async def test_gateway_health_counts_a_cancelled_native_handler_until_thread_finishes(self):
        from unittest.mock import patch
        from hermes_mcp_gateway.server import Gateway, build_catalog
        from hermes_mcp_gateway.tool_discovery import ToolDiscovery
        from test_request_boundaries import captured_app, request_context, call_params, tool
        gateway = Gateway()
        gateway.catalog = build_catalog([tool("read_file")], [], [])
        gateway.discovery = ToolDiscovery(gateway.catalog)
        app = captured_app(gateway)
        handler = getattr(app, "handler")
        started, release = threading.Event(), threading.Event()
        def native(*_args, **_kwargs):
            started.set()
            release.wait(3)
            return '{"content":"1|fixture"}'
        with patch("model_tools.handle_function_call", side_effect=native):
            request = asyncio.create_task(handler(request_context(), call_params("read_file", {})))
            try:
                await until(started.is_set)
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request
                self.assertEqual(gateway.inflight_requests, 0)
                self.assertEqual(gateway.native_calls.active_count, 1)
                endpoint = next(getattr(route, "endpoint") for route in app.routes if getattr(route, "path", None) == "/healthz")
                import json
                data = json.loads((await endpoint(None)).body)
                self.assertEqual(data["inflight_requests"], 0)
                self.assertEqual(data["inflight_native_calls"], 1)
            finally:
                release.set()
                await until(lambda: gateway.native_calls.active_count == 0)

    async def test_native_error_releases_activity(self):
        calls = NativeCalls()
        def fails():
            raise ValueError("private diagnostic")
        with self.assertRaises(ValueError):
            await calls.run(fails)
        self.assertEqual(calls.active_count, 0)
