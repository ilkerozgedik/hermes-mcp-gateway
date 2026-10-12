import unittest
from unittest.mock import AsyncMock

from hermes_mcp_gateway.config import GatewayConfig
from hermes_mcp_gateway.server import Gateway
from hermes_mcp_gateway.upstreams import ContextModeClient


class ContextTimeoutConfigTests(unittest.TestCase):
    def test_gateway_context_timeout_exceeds_context_mode_foreground_limit(self):
        config = GatewayConfig(timeout_seconds=30.0, context_timeout_seconds=90.0)
        gateway = Gateway(config)
        self.assertEqual(gateway.context.timeout, 90.0)
        self.assertEqual(gateway.config.timeout_seconds, 30.0)
        self.assertGreaterEqual(GatewayConfig().context_timeout_seconds, 90.0)
        self.assertGreater(GatewayConfig().context_timeout_seconds - 45.0, 30.0)


class ContextModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_preserves_schema_and_annotations(self):
        client = ContextModeClient("http://127.0.0.1:3050/mcp")
        client._rpc = AsyncMock(
            return_value={
                "tools": [
                    {
                        "name": "ctx_execute",
                        "description": "execute",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"code": {"type": "string"}},
                            "required": ["code"],
                        },
                        "annotations": {"readOnlyHint": False},
                    }
                ]
            }
        )
        tools = await client.discover()
        self.assertEqual(tools[0].name, "ctx_execute")
        self.assertEqual(tools[0].input_schema["required"], ["code"])
        self.assertFalse(tools[0].annotations.read_only_hint)
        await client.close()

    async def test_non_ctx_tool_from_context_mode_fails_closed(self):
        client = ContextModeClient("http://127.0.0.1:3050/mcp")
        client._rpc = AsyncMock(
            return_value={
                "tools": [{"name": "terminal", "inputSchema": {"type": "object"}}]
            }
        )
        with self.assertRaisesRegex(ValueError, "unexpected Context Mode tool"):
            await client.discover()
        await client.close()


if __name__ == "__main__":
    unittest.main()


class ContextTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_rpc_uses_cancelable_async_transport(self):
        import asyncio
        from unittest.mock import patch

        client = ContextModeClient("http://127.0.0.1:3050/mcp")
        waiting = asyncio.Event()
        try:
            async def blocked(*_args, **_kwargs):
                waiting.set()
                await asyncio.Event().wait()

            with patch.object(client._client, "post", side_effect=blocked):
                task = asyncio.create_task(client._rpc("tools/list"))
                await waiting.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
        finally:
            await client.close()
