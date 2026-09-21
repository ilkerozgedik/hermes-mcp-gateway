from __future__ import annotations

import asyncio
import base64
import http.client
import json
import os
from collections import OrderedDict
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from .config import (
    DIRECT_HERMES_TOOLS,
    DIRECT_HERMES_TOOLSETS,
    SAMCHON_GRAPH_TOOL,
    GatewayConfig,
)

_PROTOCOL = "2026-07-28"


def missing_expected(actual: set[str], expected: set[str] | frozenset[str]) -> set[str]:
    return set(expected) - actual


def _vision_value_to_result(value: Any) -> types.CallToolResult:
    if isinstance(value, str):
        return types.CallToolResult(content=[types.TextContent(text=value)])
    if not (
        isinstance(value, dict)
        and value.get("_multimodal") is True
        and isinstance(value.get("content"), list)
    ):
        raise TypeError("Hermes vision returned an unsupported result shape")

    content: list[types.TextContent | types.ImageContent] = []
    for part in value["content"]:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            content.append(types.TextContent(text=part["text"]))
            continue
        if part.get("type") != "image_url":
            continue
        image_url = part.get("image_url")
        data_url = image_url.get("url") if isinstance(image_url, dict) else None
        if not isinstance(data_url, str):
            raise ValueError("Hermes vision image payload is missing a data URL")
        header, separator, data = data_url.partition(",")
        if (
            not separator
            or not header.startswith("data:image/")
            or not header.endswith(";base64")
        ):
            raise ValueError("Hermes vision returned an unsupported image URL")
        mime_type = header[5:-7]
        try:
            base64.b64decode(data, validate=True)
        except ValueError as exc:
            raise ValueError("Hermes vision returned invalid base64 image data") from exc
        content.append(types.ImageContent(data=data, mime_type=mime_type))

    if not any(isinstance(item, types.ImageContent) for item in content):
        raise ValueError("Hermes vision multimodal result did not contain an image")
    return types.CallToolResult(content=content)


_BROWSER_STALE_MARKERS = (
    "410 client error: gone",
    "tab no longer exists",
    "page crashed",
    "tab was destroyed",
    "browser was restarted",
)


def _browser_result_is_stale(value: str) -> bool:
    """Recognize Camofox stale-tab responses without trusting status text alone."""
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        payload = None

    fragments = [value]
    if isinstance(payload, dict):
        fragments.extend(
            str(payload.get(key, "")) for key in ("error", "code", "recovery")
        )
    haystack = " ".join(fragments).lower()
    return any(marker in haystack for marker in _BROWSER_STALE_MARKERS)


def _reset_camofox_session(task_id: str) -> None:
    """Drop only the local stale-tab binding; preserve managed profile state."""
    try:
        from tools.browser_camofox import _drop_session

        _drop_session(task_id)
    except Exception:
        # The gateway must remain usable with a non-Camofox Hermes installation.
        return


def _stale_browser_result() -> str:
    return json.dumps(
        {
            "success": False,
            "error": (
                "The Camofox browser tab expired or was destroyed before this "
                "operation completed. Call browser_navigate to recover it."
            ),
            "code": "stale_tab",
            "retryable": True,
            "recovery": "browser_navigate",
        }
    )


class ContextModeClient:
    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url
        self.timeout = timeout
        parsed = urlsplit(url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("Context Mode URL must be plain HTTP")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._path = parsed.path or "/"

    def _rpc_sync(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        meta = {
            "io.modelcontextprotocol/protocolVersion": _PROTOCOL,
            "io.modelcontextprotocol/clientInfo": {
                "name": "hermes-mcp-gateway",
                "version": "0.1.0",
            },
            "io.modelcontextprotocol/clientCapabilities": {},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": _PROTOCOL,
            "Mcp-Method": method,
            "Connection": "close",
        }
        if method == "tools/call" and params and params.get("name"):
            headers["Mcp-Name"] = str(params["name"])
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": {**(params or {}), "_meta": meta},
        }
        connection = http.client.HTTPConnection(
            self._host, self._port, timeout=self.timeout
        )
        try:
            connection.request("POST", self._path, json.dumps(body), headers)
            response = connection.getresponse()
            raw = response.read()
        finally:
            connection.close()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Context Mode returned invalid JSON") from exc
        if response.status != 200:
            message = (
                payload.get("error", {}).get("message")
                if isinstance(payload, dict)
                else None
            )
            raise RuntimeError(message or f"Context Mode HTTP {response.status}")
        if payload.get("error"):
            error = payload["error"]
            raise RuntimeError(str(error.get("message") or "Context Mode MCP error"))
        result = payload.get("result")
        if not isinstance(result, dict):
            raise TypeError("Context Mode returned an invalid MCP result")
        return result

    async def _rpc(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._rpc_sync, method, params)

    async def discover(self) -> list[types.Tool]:
        result = await self._rpc("tools/list")
        tools = [types.Tool.model_validate(item) for item in result.get("tools", [])]
        unexpected = [tool.name for tool in tools if not tool.name.startswith("ctx_")]
        if unexpected:
            raise ValueError(
                f"unexpected Context Mode tool(s): {', '.join(sorted(unexpected))}"
            )
        return tools

    async def call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        result = await self._rpc("tools/call", {"name": name, "arguments": arguments})
        return types.CallToolResult.model_validate(result)

    async def close(self) -> None:
        return None


class _SamchonGraphSession:
    def __init__(self, config: GatewayConfig, cwd: str):
        self.config = config
        self.cwd = cwd
        self.active_calls = 0
        self._call_lock = asyncio.Lock()
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        if self._session is not None:
            return
        stack = AsyncExitStack()
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"LANG", "LC_ALL", "TERM", "TMPDIR"} or key.startswith("XDG_")
        }
        env.update(
            {
                "HOME": "/home/hermes",
                "USER": "hermes",
                "LOGNAME": "hermes",
                "PATH": (
                    "/home/hermes/.local/bin:/home/hermes/.local/npm/bin:"
                    "/home/hermes/.cargo/bin:/home/hermes/.nix-profile/bin:"
                    "/opt/node/bin:/usr/local/bin:/usr/bin:/bin"
                ),
            }
        )
        params = StdioServerParameters(
            command=self.config.samchon_graph_command,
            args=[],
            cwd=self.cwd,
            env=env,
        )
        read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        self._stack = stack
        self._session = session

    async def discover(self) -> list[types.Tool]:
        if self._session is None:
            await self.start()
        assert self._session is not None
        return list((await self._session.list_tools()).tools)

    async def call(self, arguments: dict[str, Any]) -> types.CallToolResult:
        if self._session is None:
            await self.start()
        assert self._session is not None
        async with self._call_lock:
            result = await self._session.call_tool(
                SAMCHON_GRAPH_TOOL,
                arguments,
                read_timeout_seconds=self.config.samchon_graph_timeout_seconds,
            )
        if not isinstance(result, types.CallToolResult):
            raise TypeError("Samchon Graph returned unsupported MCP result type")
        return result

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None


class SamchonGraphClient:
    """Route one public graph tool to a bounded resident session per repository."""

    def __init__(self, config: GatewayConfig):
        self.config = config
        self._sessions: OrderedDict[str, _SamchonGraphSession] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def public_tool(upstream: types.Tool) -> types.Tool:
        schema = dict(upstream.input_schema or {})
        properties = dict(schema.get("properties") or {})
        properties["cwd"] = {
            "type": "string",
            "description": (
                "Absolute project root. Must be under an allowed work/source root; "
                "the gateway keeps a resident graph session per resolved root."
            ),
        }
        required = ["cwd", *[item for item in schema.get("required", []) if item != "cwd"]]
        schema["type"] = "object"
        schema["properties"] = properties
        schema["required"] = required
        description = (upstream.description or "").rstrip()
        if description:
            description += "\n\n"
        description += "Gateway routing: pass absolute project root in cwd."
        return upstream.model_copy(
            update={"input_schema": schema, "description": description}
        )

    def resolve_cwd(self, cwd: str) -> str:
        if not isinstance(cwd, str) or not cwd.strip():
            raise ValueError("cwd must be a non-empty absolute project path")
        candidate = Path(cwd)
        if not candidate.is_absolute():
            raise ValueError("cwd must be an absolute project path")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"cwd does not exist: {cwd}") from exc
        if not resolved.is_dir():
            raise ValueError(f"cwd is not a directory: {resolved}")
        roots = [Path(root).resolve() for root in self.config.samchon_graph_allowed_roots]
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise ValueError(
                "cwd is outside allowed roots: "
                + ", ".join(str(root) for root in roots)
            )
        return str(resolved)

    @staticmethod
    def _graph_tool(tools: list[types.Tool]) -> types.Tool:
        names = {tool.name for tool in tools}
        if names != {SAMCHON_GRAPH_TOOL}:
            raise RuntimeError(
                "Samchon Graph MCP surface mismatch: "
                f"expected only {SAMCHON_GRAPH_TOOL}, got {', '.join(sorted(names)) or 'none'}"
            )
        return next(tool for tool in tools if tool.name == SAMCHON_GRAPH_TOOL)

    async def _trim_sessions_unlocked(self, *, protected: str | None = None) -> None:
        while len(self._sessions) > self.config.samchon_graph_max_sessions:
            victim = next(
                (
                    (key, session)
                    for key, session in self._sessions.items()
                    if key != protected and session.active_calls == 0
                ),
                None,
            )
            if victim is None:
                return
            key, session = victim
            del self._sessions[key]
            await session.close()

    async def _get_session_unlocked(self, cwd: str) -> _SamchonGraphSession:
        existing = self._sessions.get(cwd)
        if existing is not None:
            self._sessions.move_to_end(cwd)
            return existing

        session = _SamchonGraphSession(self.config, cwd)
        try:
            self._graph_tool(await session.discover())
        except Exception:
            await session.close()
            raise
        self._sessions[cwd] = session
        self._sessions.move_to_end(cwd)
        await self._trim_sessions_unlocked(protected=cwd)
        return session

    async def describe(self) -> types.Tool:
        cwd = self.resolve_cwd(self.config.samchon_graph_schema_cwd)
        async with self._lock:
            session = await self._get_session_unlocked(cwd)
            return self.public_tool(self._graph_tool(await session.discover()))

    async def healthy(self) -> bool:
        cwd = self.resolve_cwd(self.config.samchon_graph_schema_cwd)
        session = self._sessions.get(cwd)
        if session is None:
            return False
        if session.active_calls:
            return True
        try:
            self._graph_tool(await session.discover())
        except Exception:  # noqa: BLE001 - readiness probe must degrade, not crash
            return False
        return True

    async def call(self, arguments: dict[str, Any]) -> types.CallToolResult:
        payload = dict(arguments)
        cwd = self.resolve_cwd(payload.pop("cwd", ""))
        async with self._lock:
            session = await self._get_session_unlocked(cwd)
            session.active_calls += 1
        try:
            return await session.call(payload)
        finally:
            async with self._lock:
                session.active_calls -= 1
                await self._trim_sessions_unlocked()

    async def close(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                await session.close()


class HermesToolsClient:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        if self._session is not None:
            return
        stack = AsyncExitStack()
        env = dict(os.environ)
        env.update(
            {
                "HOME": "/home/hermes",
                "USER": "hermes",
                "LOGNAME": "hermes",
                "HERMES_HOME": self.config.hermes_home,
                "HERMES_QUIET": "1",
                "HERMES_REDACT_SECRETS": "true",
            }
        )
        params = StdioServerParameters(
            command=self.config.hermes_python,
            args=["-m", "agent.transports.hermes_tools_mcp_server"],
            cwd=self.config.hermes_cwd,
            env=env,
        )
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._stack = stack
        self._session = session

    @staticmethod
    def _discover_direct_tools() -> list[types.Tool]:
        from model_tools import get_tool_definitions

        definitions = get_tool_definitions(
            enabled_toolsets=list(DIRECT_HERMES_TOOLSETS),
            quiet_mode=True,
            skip_tool_search_assembly=True,
        ) or []
        tools: list[types.Tool] = []
        for definition in definitions:
            function = definition.get("function") if isinstance(definition, dict) else None
            if not isinstance(function, dict) or function.get("name") not in DIRECT_HERMES_TOOLS:
                continue
            tools.append(
                types.Tool(
                    name=function["name"],
                    description=function.get("description"),
                    input_schema=function.get("parameters")
                    or {"type": "object", "properties": {}},
                )
            )
        return tools

    async def discover(self) -> list[types.Tool]:
        if self._session is None:
            await self.start()
        assert self._session is not None
        upstream, direct = await asyncio.gather(
            self._session.list_tools(),
            asyncio.to_thread(self._discover_direct_tools),
        )
        merged = {tool.name: tool for tool in upstream.tools}
        for tool in direct:
            merged.setdefault(tool.name, tool)
        return list(merged.values())

    async def call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        if self._session is None:
            raise RuntimeError("Hermes tools upstream is not started")
        result = await self._session.call_tool(
            name, arguments, read_timeout_seconds=self.config.timeout_seconds
        )
        if not isinstance(result, types.CallToolResult):
            raise TypeError(f"Hermes tool {name} returned unsupported MCP result type")
        return result

    async def call_direct(
        self, name: str, arguments: dict[str, Any], *, task_id: str
    ) -> types.CallToolResult:
        from model_tools import handle_function_call

        if name == "process" and arguments.get("action") != "list":
            session_id = arguments.get("session_id")
            if session_id:
                from tools.process_registry import process_registry

                session = process_registry.get(str(session_id))
                if session is not None and task_id not in {
                    session.task_id,
                    session.session_key,
                }:
                    raise PermissionError("process session belongs to another MCP session")

        value = await asyncio.to_thread(
            handle_function_call,
            name,
            arguments,
            task_id=task_id,
            session_id=task_id,
            enabled_toolsets=list(DIRECT_HERMES_TOOLSETS),
        )
        if not isinstance(value, str):
            raise TypeError(f"Hermes tool {name} returned unsupported result type")
        return types.CallToolResult(content=[types.TextContent(text=value)])

    async def call_browser(
        self, name: str, arguments: dict[str, Any], *, task_id: str
    ) -> types.CallToolResult:
        from model_tools import handle_function_call

        value = await asyncio.to_thread(
            handle_function_call, name, arguments, task_id=task_id
        )
        if not isinstance(value, str):
            raise TypeError(f"Hermes browser tool {name} returned unsupported result type")

        # Camofox deliberately returns HTTP 410 when a known tab was destroyed
        # (browser restart, crash, or tab reaper). The pinned Hermes client only
        # recovers 404 on navigate, so keep this compatibility boundary in the
        # gateway: reset the local binding, retry navigate once (safe after a
        # non-applied stale-tab response), and never replay other operations.
        if _browser_result_is_stale(value):
            _reset_camofox_session(task_id)
            if name == "browser_navigate":
                value = await asyncio.to_thread(
                    handle_function_call, name, arguments, task_id=task_id
                )
            if _browser_result_is_stale(value):
                value = _stale_browser_result()

        return types.CallToolResult(content=[types.TextContent(text=value)])

    async def call_vision(self, arguments: dict[str, Any]) -> types.CallToolResult:
        from model_tools import handle_function_call

        value = await asyncio.to_thread(
            handle_function_call, "vision_analyze", arguments
        )
        return _vision_value_to_result(value)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
