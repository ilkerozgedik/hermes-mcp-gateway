"""Narrow native Hermes MCP endpoint for the existing read-only Gateway LSP adapter.

Only stdio, no listening socket. No language-server installation and no file writes.
"""
from __future__ import annotations

import asyncio
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import GatewayConfig
from .lsp import LSP_TOOLS, LSPAdapter, lsp_tool_schemas
from .output import error_result, json_result


def create_handlers(adapter: LSPAdapter, *, max_text_chars: int = 65_536):
    async def list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=lsp_tool_schemas())

    async def call_tool(
        _ctx: Any, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        if params.name not in LSP_TOOLS:
            return error_result("Unsupported LSP tool", limit=max_text_chars)
        try:
            result = await asyncio.to_thread(adapter.call, params.name, params.arguments or {})
        except (ValueError, PermissionError, RuntimeError, OSError, TimeoutError) as exc:
            return error_result(str(exc), limit=max_text_chars)
        return json_result(result, limit=max_text_chars)

    return list_tools, call_tool


async def serve() -> None:
    config = GatewayConfig()
    adapter = LSPAdapter(
        allowed_roots=config.samchon_graph_allowed_roots,
        timeout_seconds=config.timeout_seconds,
    )
    list_tools, call_tool = create_handlers(adapter, max_text_chars=config.max_text_chars)
    server = Server(
        "hermes-readonly-lsp",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )
    try:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        await asyncio.to_thread(adapter.close)


def main() -> None:
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
