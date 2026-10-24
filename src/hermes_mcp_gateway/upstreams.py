from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections import OrderedDict
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from .config import (
    DIRECT_HERMES_TOOLS,
    DIRECT_HERMES_TOOLSETS,
    SAMCHON_GRAPH_TOOL,
    GatewayConfig,
)
from .native_activity import NativeCalls

_LOG = logging.getLogger(__name__)


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

    content: list[types.ContentBlock] = []
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
            raise TypeError("Hermes vision image payload is missing a data URL")
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

    if isinstance(payload, dict):
        if payload.get("success") is True:
            return False
        # Snapshot text is untrusted page content, not an error from Camofox.
        haystack = " ".join(
            str(payload.get(key, "")) for key in ("error", "code", "recovery")
        ).lower()
    else:
        haystack = value.lower()
    return any(marker in haystack for marker in _BROWSER_STALE_MARKERS)


def _reset_camofox_session(task_id: str) -> None:
    """Drop only the local stale-tab binding; preserve managed profile state."""
    try:
        from tools.browser_camofox import _drop_session

        _drop_session(task_id)
    except Exception:  # noqa: BLE001 - optional Camofox cleanup must be best effort
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


class _SamchonGraphSession:
    def __init__(self, config: GatewayConfig, cwd: str):
        self.config = config
        self.cwd = cwd
        self.active_calls = 0
        self._lifecycle_lock = asyncio.Lock()
        self._queue: (
            asyncio.Queue[tuple[str, dict[str, Any] | None, asyncio.Future[Any]]] | None
        ) = None
        self._task: asyncio.Task[None] | None = None
        self._failure: Exception | None = None
        self._pending_operations = 0
        self._on_activity_finished: Any = None

    @property
    def has_active_work(self) -> bool:
        return self.active_calls > 0 or self._pending_operations > 0

    def _finish_operation(self) -> None:
        self._pending_operations -= 1
        if self._on_activity_finished is not None:
            self._on_activity_finished()

    def _environment(self) -> dict[str, str]:
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
        return env

    async def _serve(
        self,
        queue: asyncio.Queue[tuple[str, dict[str, Any] | None, asyncio.Future[Any]]],
        ready: asyncio.Future[None],
    ) -> None:
        try:
            async with AsyncExitStack() as stack:
                params = StdioServerParameters(
                    command=self.config.samchon_graph_command,
                    args=[],
                    cwd=self.cwd,
                    env=self._environment(),
                )
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(params)
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                if not ready.done():
                    ready.set_result(None)

                while True:
                    operation, payload, response = await queue.get()
                    if operation == "close":
                        if not response.done():
                            response.set_result(None)
                        break
                    if response.cancelled():
                        self._finish_operation()
                        continue
                    try:
                        if operation == "discover":
                            value: Any = list((await session.list_tools()).tools)
                        elif operation == "call":
                            result = await session.call_tool(
                                SAMCHON_GRAPH_TOOL,
                                payload or {},
                                read_timeout_seconds=(
                                    self.config.samchon_graph_timeout_seconds
                                ),
                            )
                            if not isinstance(result, types.CallToolResult):
                                raise TypeError(
                                    "Samchon Graph returned unsupported MCP result type"
                                )
                            value = result
                        else:
                            raise RuntimeError(
                                f"unsupported Samchon Graph session operation: {operation}"
                            )
                    except Exception as exc:  # noqa: BLE001 - forward upstream failure
                        if not response.done():
                            response.set_exception(exc)
                    else:
                        if not response.done():
                            response.set_result(value)
                    finally:
                        self._finish_operation()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 - owner must wake all waiters
            self._failure = exc
            if not ready.done():
                ready.set_exception(exc)
        finally:
            failure = self._failure or RuntimeError("Samchon Graph session closed")
            while not queue.empty():
                operation, _, response = queue.get_nowait()
                if operation != "close":
                    self._finish_operation()
                if not response.done():
                    response.set_exception(failure)

    async def _start_unlocked(self) -> None:
        if self._task is not None and not self._task.done():
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any] | None, asyncio.Future[Any]]] = (
            asyncio.Queue()
        )
        ready: asyncio.Future[None] = loop.create_future()
        self._failure = None
        self._queue = queue
        self._task = asyncio.create_task(
            self._serve(queue, ready),
            name=f"samchon-graph:{self.cwd}",
        )
        await asyncio.shield(ready)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            await self._start_unlocked()

    async def _request(
        self, operation: str, payload: dict[str, Any] | None = None
    ) -> Any:
        loop = asyncio.get_running_loop()
        async with self._lifecycle_lock:
            await self._start_unlocked()
            assert self._queue is not None
            response: asyncio.Future[Any] = loop.create_future()
            self._pending_operations += 1
            self._queue.put_nowait((operation, payload, response))
        return await response

    async def discover(self) -> list[types.Tool]:
        result = await self._request("discover")
        if not isinstance(result, list) or not all(
            isinstance(tool, types.Tool) for tool in result
        ):
            raise TypeError("Samchon Graph returned unsupported tool discovery result")
        return result

    async def call(self, arguments: dict[str, Any]) -> types.CallToolResult:
        result = await self._request("call", arguments)
        if not isinstance(result, types.CallToolResult):
            raise TypeError("Samchon Graph returned unsupported MCP result type")
        return result

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lifecycle_lock:
            task = self._task
            queue = self._queue
            if task is None:
                return
            if not task.done() and queue is not None:
                response: asyncio.Future[Any] = loop.create_future()
                queue.put_nowait(("close", None, response))
                await response
            await task
            self._task = None
            self._queue = None
            self._failure = None

    async def abort(self) -> None:
        """Cancel a failed or abandoned startup and bound the owner-task wait."""
        async with self._lifecycle_lock:
            task = self._task
            if task is None:
                return
            if not task.done():
                task.cancel()
                done, _ = await asyncio.wait({task}, timeout=1.0)
                if not done:
                    return
            await asyncio.gather(task, return_exceptions=True)
            if self._task is task:
                self._task = None
                self._queue = None
                self._failure = None


class SamchonGraphClient:
    """Route one public graph tool to a bounded resident session per repository."""

    def __init__(self, config: GatewayConfig):
        self.config = config
        self._sessions: OrderedDict[str, _SamchonGraphSession] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()

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

    @staticmethod
    def _session_has_active_work(session: Any) -> bool:
        return bool(
            getattr(session, "has_active_work", getattr(session, "active_calls", 0) > 0)
        )

    def _trim_sessions_unlocked(
        self, *, protected: str | None = None
    ) -> list[_SamchonGraphSession]:
        evicted: list[_SamchonGraphSession] = []
        while len(self._sessions) > self.config.samchon_graph_max_sessions:
            victim = next(
                (
                    (key, session)
                    for key, session in self._sessions.items()
                    if key != protected and not self._session_has_active_work(session)
                ),
                None,
            )
            if victim is None:
                return evicted
            key, session = victim
            del self._sessions[key]
            evicted.append(session)
        return evicted

    @staticmethod
    async def _close_sessions(sessions: list[_SamchonGraphSession]) -> None:
        for session in sessions:
            await session.close()

    def _track_cleanup_task(self, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_task_finished)
        return task

    def _cleanup_task_finished(self, task: asyncio.Task[Any]) -> None:
        self._cleanup_tasks.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            _LOG.error(
                "Samchon Graph cleanup task failed error_type=%s",
                type(exception).__name__,
            )

    def _schedule_close_sessions(
        self, sessions: list[_SamchonGraphSession]
    ) -> asyncio.Task[Any] | None:
        if not sessions:
            return None
        task = asyncio.create_task(
            self._close_sessions(sessions), name="samchon-graph:retire-sessions"
        )
        return self._track_cleanup_task(task)

    async def _close_sessions_tracked(
        self, sessions: list[_SamchonGraphSession]
    ) -> None:
        task = self._schedule_close_sessions(sessions)
        if task is not None:
            await asyncio.shield(task)

    async def _wait_for_cleanup(self) -> None:
        while self._cleanup_tasks:
            tasks = tuple(self._cleanup_tasks)
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    def _schedule_trim_sessions(self) -> None:
        async def trim() -> None:
            async with self._lock:
                evicted = self._trim_sessions_unlocked()
            await self._close_sessions_tracked(evicted)

        task = asyncio.create_task(trim(), name="samchon-graph:lru-trim")
        self._track_cleanup_task(task)

    async def _get_session_unlocked(self, cwd: str) -> _SamchonGraphSession:
        existing = self._sessions.get(cwd)
        if existing is not None:
            self._sessions.move_to_end(cwd)
            return existing

        session = _SamchonGraphSession(self.config, cwd)
        try:
            self._graph_tool(await session.discover())
        except BaseException as startup_error:  # preserve cancellation
            try:
                await session.abort()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve original
                _LOG.warning(
                    "Samchon Graph startup cleanup failed startup_type=%s cleanup_type=%s",
                    type(startup_error).__name__,
                    type(cleanup_error).__name__,
                )
            finally:
                # abort() has a bounded wait; keep any late owner cleanup visible
                # to client.close() rather than orphaning the unregistered task.
                if session._task is not None:
                    self._track_cleanup_task(session._task)
            raise
        session._on_activity_finished = self._schedule_trim_sessions
        self._sessions[cwd] = session
        self._sessions.move_to_end(cwd)
        return session

    async def describe(self) -> types.Tool:
        cwd = self.resolve_cwd(self.config.samchon_graph_schema_cwd)
        async with self._lock:
            session = await self._get_session_unlocked(cwd)
            public = self.public_tool(self._graph_tool(await session.discover()))
            evicted = self._trim_sessions_unlocked(protected=cwd)
        await self._close_sessions_tracked(evicted)
        return public

    async def healthy(self) -> bool:
        cwd = self.resolve_cwd(self.config.samchon_graph_schema_cwd)
        session = self._sessions.get(cwd)
        if session is None:
            return False
        if self._session_has_active_work(session):
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
            evicted = self._trim_sessions_unlocked(protected=cwd)
        try:
            await self._close_sessions_tracked(evicted)
            return await session.call(payload)
        finally:
            async with self._lock:
                session.active_calls -= 1
                evicted = self._trim_sessions_unlocked()
            await self._close_sessions_tracked(evicted)

    async def close(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await self._close_sessions_tracked(sessions)
        await self._wait_for_cleanup()


class HermesToolsClient:
    def __init__(self, config: GatewayConfig, *, native_calls: NativeCalls | None = None):
        self.config = config
        self.native_calls = native_calls or NativeCalls()
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
        bootstrap_code = (
            "import sys, model_tools; "
            "_orig = model_tools.get_tool_definitions; "
            "model_tools.get_tool_definitions = lambda *a, **k: _orig(*a, **{**k, 'skip_tool_search_assembly': True}); "
            "from agent.transports.hermes_tools_mcp_server import main; "
            "sys.exit(main())"
        )
        hermes_agent_dir = str(Path(self.config.hermes_home) / "hermes-agent")
        cwd = hermes_agent_dir if os.path.isdir(hermes_agent_dir) else self.config.hermes_cwd
        if "PYTHONPATH" not in env and os.path.isdir(cwd):
            env["PYTHONPATH"] = cwd
        params = StdioServerParameters(
            command=self.config.hermes_python,
            args=["-c", bootstrap_code],
            cwd=cwd,
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
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if name not in DIRECT_HERMES_TOOLS:
                continue
            if any(t.name == name for t in tools):
                continue
            tools.append(
                types.Tool(
                    name=name,
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
            self.native_calls.run(self._discover_direct_tools),
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
        self, name: str, arguments: dict[str, Any], *, task_id: str | None
    ) -> types.CallToolResult:
        # These native synchronous handlers run in worker threads. Do not add an
        # asyncio timeout here: it would stop waiting without stopping the work.
        from gateway.session_context import clear_session_vars, set_session_vars
        from model_tools import handle_function_call

        if name == "process_manage" and arguments.get("action") != "list":
            session_id = arguments.get("session_id")
            if session_id:
                from tools.process_registry import process_registry

                session = process_registry.get(str(session_id))
                if session is not None and task_id not in {
                    session.task_id,
                    session.session_key,
                }:
                    raise PermissionError(
                        "process session belongs to another MCP session"
                    )

        def dispatch() -> str:
            # Native terminal backends key cwd/env by HERMES_SESSION_KEY, not
            # handle_function_call's task_id alone. Bind only in this worker's
            # copied ContextVars; never mutate process env or inherit a chat route.
            tokens = set_session_vars(
                session_key=task_id or "",
                session_id=task_id or "",
                async_delivery=False,
                session_history_delivery="",
            )
            try:
                return handle_function_call(
                    name,
                    arguments,
                    task_id=task_id,
                    session_id=task_id,
                    tool_call_id=uuid4().hex,
                    enabled_toolsets=list(DIRECT_HERMES_TOOLSETS),
                )
            finally:
                clear_session_vars(tokens)

        value = await self.native_calls.run(dispatch)
        if not isinstance(value, str):
            raise TypeError(f"Hermes tool {name} returned unsupported result type")
        return types.CallToolResult(content=[types.TextContent(text=value)])

    async def call_browser(
        self, name: str, arguments: dict[str, Any], *, task_id: str
    ) -> types.CallToolResult:
        # Browser dispatch is thread-backed; cancellation cannot terminate the
        # native handler, so this path has no claimed Gateway-wide deadline.
        from model_tools import handle_function_call

        value = await self.native_calls.run(
            handle_function_call,
            name,
            arguments,
            task_id=task_id,
            tool_call_id=uuid4().hex,
        )
        if not isinstance(value, str):
            raise TypeError(
                f"Hermes browser tool {name} returned unsupported result type"
            )

        # Camofox deliberately returns HTTP 410 when a known tab was destroyed
        # (browser restart, crash, or tab reaper). The pinned Hermes client only
        # recovers 404 on navigate, so keep this compatibility boundary in the
        # gateway: reset the local binding, retry navigate once (safe after a
        # non-applied stale-tab response), and never replay other operations.
        if _browser_result_is_stale(value):
            _reset_camofox_session(task_id)
            if name == "browser_navigate":
                value = await self.native_calls.run(
                    handle_function_call,
                    name,
                    arguments,
                    task_id=task_id,
                    tool_call_id=uuid4().hex,
                )
            if _browser_result_is_stale(value):
                value = _stale_browser_result()

        return types.CallToolResult(content=[types.TextContent(text=value)])

    async def call_vision(self, arguments: dict[str, Any]) -> types.CallToolResult:
        # Provider inference is synchronous in a worker thread and is not safely
        # cancellable from asyncio; keep its deadline owned by the native client.
        from model_tools import handle_function_call

        value = await self.native_calls.run(
            handle_function_call, "vision_analyze", arguments, tool_call_id=uuid4().hex
        )
        return _vision_value_to_result(value)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
