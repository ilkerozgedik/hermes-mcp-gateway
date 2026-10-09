"""Gateway-side client for the standalone coordinator MCP endpoint."""

from __future__ import annotations

from mcp import types

from ..upstreams import ContextModeClient
from .server import TOOLS


class CoordinatorClient(ContextModeClient):
    async def discover(self) -> list[types.Tool]:
        result = await self._rpc("tools/list")
        tools = [types.Tool.model_validate(item) for item in result.get("tools", [])]
        names = {tool.name for tool in tools}
        if names != set(TOOLS):
            raise ValueError("coordinator MCP tool contract mismatch")
        return tools
