"""Native MCP session identity must survive repeated calls and isolate clients."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import uvicorn
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from hermes_mcp_gateway.server import Gateway, build_catalog, create_app
from hermes_mcp_gateway.tool_discovery import ToolDiscovery


def tool(name):
    return types.Tool(name=name, input_schema={"type": "object", "properties": {}})


class MCPClientIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_real_mcp_sessions_have_distinct_stable_tool_ownership(self):
        async def fake_start(gateway):
            gateway.catalog = build_catalog(
                [tool("skills_list"), tool("browser_click")],
                [tool("memory_context")],
                [tool("startup_context")],
            )
            gateway.discovery = ToolDiscovery(gateway.catalog)

        async def fake_browser(_self, _name, _args, *, task_id):
            return types.CallToolResult(content=[types.TextContent(text=task_id)])

        with (
            patch.object(Gateway, "start", fake_start),
            patch.object(Gateway, "close", new_callable=AsyncMock),
            patch(
                "hermes_mcp_gateway.upstreams.HermesToolsClient.call_browser",
                fake_browser,
            ),
        ):
            server = uvicorn.Server(
                uvicorn.Config(
                    create_app(),
                    host="127.0.0.1",
                    port=0,
                    log_level="error",
                    lifespan="on",
                )
            )
            runner = asyncio.create_task(server.serve())
            try:
                for _ in range(200):
                    if server.started and server.servers:
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(server.started)
                port = server.servers[0].sockets[0].getsockname()[1]

                async def client_identity():
                    async with (
                        streamable_http_client(
                            f"http://127.0.0.1:{port}/mcp"
                        ) as streams,
                        ClientSession(streams[0], streams[1]) as session,
                    ):
                        await session.initialize()
                        owners = []
                        for _ in range(2):
                            result = await session.call_tool(
                                "tool_call",
                                {"calls": [{"name": "browser_click", "arguments": {}}]},
                            )
                            self.assertFalse(result.is_error)
                            owners.append(result.content[0].text)
                        self.assertEqual(owners[0], owners[1])
                        return owners[0]

                first, second = await asyncio.gather(
                    client_identity(), client_identity()
                )
                self.assertTrue(first.startswith("chatgpt:"))
                self.assertTrue(second.startswith("chatgpt:"))
                self.assertNotEqual(first, second)
            finally:
                server.should_exit = True
                await asyncio.wait_for(runner, timeout=8)


if __name__ == "__main__":
    unittest.main()
