from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from contextvars import Context
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .capabilities import HermesCapabilities, capability_tool_schemas
from .config import (
    BROWSER_TOOLS,
    DIRECT_HERMES_TOOLS,
    HERMES_ALLOWLIST,
    HERMES_REQUIRED,
    IMAGE_TOOLS,
    SAMCHON_GRAPH_TOOL,
    STATEFUL_HERMES_TOOLS,
    WEB_TOOLS,
    GatewayConfig,
)
from .execution import NativeExecution, execution_schema
from .execution_guard import guard_terminal
from .jobs import ResumableJobs, job_tool_schemas
from .lsp import LSPAdapter, lsp_tool_schemas
from .memory import MemoryAdapter, memory_tool_schemas
from .native_activity import NativeCalls
from .output import _sanitize_structure, error_result, normalize_result
from .sessions import (
    SESSION_TOKEN_TTL_SECONDS,
    SessionCapability,
    SessionCapabilityStore,
    session_close_tool_schema,
    session_open_tool_schema,
)
from .startup import StartupContext, startup_tool_schema
from .tool_discovery import BRIDGE_NAMES, PUBLIC_NAMES, ToolDiscovery, public_schemas
from .upstreams import (
    HermesToolsClient,
    SamchonGraphClient,
    missing_expected,
)

logger = logging.getLogger("hermes_mcp_gateway")
if logger.level == logging.NOTSET:
    # Uvicorn's application log level is INFO; set this explicitly too so request
    # telemetry remains enabled when the Gateway is embedded by another runner.
    logger.setLevel(logging.INFO)
if not logger.handlers and not logging.getLogger().handlers:
    # Uvicorn configures its own loggers but leaves the root logger without a
    # handler. Give gateway request telemetry a real output path in that case.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False


def browser_task_id(ctx: Any) -> str | None:
    # MCP 2.0 lowlevel ServerRequestContext carries the validated connection
    # here. Modern (2026-07-28) HTTP has no protocol session, even when
    # stateless_http=False. Never trust a raw client-supplied session header.
    connection = getattr(getattr(ctx, "session", None), "_connection", None)
    session_id = getattr(connection, "session_id", None)
    return f"chatgpt:{session_id}" if session_id else None


def deferred_image_tool() -> types.Tool:
    # Hermes omits this MCP tool while the selected provider is rate-limited.
    # Capabilities come from Hermes itself; provider availability is checked at call time.
    from tools.image_generation_tool import _build_dynamic_image_schema

    schema = _build_dynamic_image_schema()
    return types.Tool(
        name="image_generate",
        description=schema["description"],
        input_schema=schema["parameters"],
    )


def call_deferred_image(arguments: dict[str, Any]) -> types.CallToolResult:
    from model_tools import handle_function_call

    raw = handle_function_call("image_generate", arguments)
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        payload = None
    failed = isinstance(payload, dict) and (
        payload.get("success") is False or "error" in payload
    )
    return types.CallToolResult(content=[types.TextContent(text=raw)], is_error=failed)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    tool: types.Tool
    source: str


def build_catalog(
    hermes_tools: list[types.Tool],
    memory_tools: list[types.Tool],
    gateway_tools: list[types.Tool] | None = None,
    *,
    capability_tools: list[types.Tool] | None = None,
    graph_tools: list[types.Tool] | None = None,
    lsp_tools: list[types.Tool] | None = None,
) -> dict[str, CatalogEntry]:
    catalog: dict[str, CatalogEntry] = {}
    groups = (
        ("hermes", hermes_tools, HERMES_ALLOWLIST),
        ("graph", graph_tools or [], {SAMCHON_GRAPH_TOOL}),
        ("lsp", lsp_tools or [], None),
        ("capability", capability_tools or [], None),
        ("memory", memory_tools, None),
        ("gateway", gateway_tools or [], None),
    )
    for source, tools, allowlist in groups:
        for tool in tools:
            name = tool.name
            if allowlist is not None and name not in allowlist:
                continue
            if name in catalog:
                raise ValueError(f"tool collision: {name}")
            public_tool = tool
            if source == "hermes" and name == "terminal":
                schema = deepcopy(tool.input_schema)
                timeout = schema.get("properties", {}).get("timeout")
                if isinstance(timeout, dict):
                    timeout["default"] = 30
                    timeout["maximum"] = 30
                    timeout["description"] = (
                        "Foreground only, maximum 30 seconds. "
                        "For long commands use job_start."
                    )
                public_tool = tool.model_copy(update={
                    "input_schema": schema,
                    "description": "Execution Guard: short foreground calls only (<=30s); "
                    "background and long commands require job_start. " + (tool.description or ""),
                })
            if source == "hermes" and (
                name == "vision_analyze" or name in BROWSER_TOOLS
            ):
                public_tool = public_tool.model_copy(update={"output_schema": None})
            catalog[name] = CatalogEntry(tool=public_tool, source=source)
    return catalog


def _structured_read_file_result(
    result: types.CallToolResult, *, limit: int
) -> types.CallToolResult:
    """Expose Hermes file content as MCP data, preserving pagination on large reads."""
    if result.is_error or result.structured_content is not None or len(result.content) != 1:
        return result
    item = result.content[0]
    if not isinstance(item, types.TextContent):
        return result
    try:
        payload = json.loads(item.text)
    except (ValueError, TypeError):
        return result
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
        return result

    def structured_value(data: dict[str, Any]) -> dict[str, Any]:
        fields = {k: v for k, v in data.items() if k not in {"content", "file_text"}}
        return {**fields, "file_text": data["content"]}

    structured = structured_value(payload)
    if len(json.dumps(_sanitize_structure(structured), ensure_ascii=False)) <= limit:
        return result.model_copy(update={"structured_content": structured})

    # Hermes already supplies numbered lines; only shrink the MCP response envelope.
    lines = payload["content"].splitlines(keepends=True)
    kept: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) > limit:
            break
        kept.append(line)
        size += len(line)

    while kept:
        numbered = kept[-1].split("|", 1)[0]
        if not numbered.isdecimal():
            return error_result("read_file pagination requires Hermes numbered lines", limit=limit)
        next_offset = int(numbered) + 1
        page = {
            **payload,
            "content": "".join(kept).rstrip("\n"),
            "truncated": True,
            "next_offset": next_offset,
            "hint": f"Gateway output limit reached; continue with offset={next_offset}.",
        }
        structured = structured_value(page)
        text = json.dumps(page, ensure_ascii=False)
        if (
            len(text) <= limit
            and len(json.dumps(_sanitize_structure(structured), ensure_ascii=False)) <= limit
        ):
            return types.CallToolResult(
                content=[types.TextContent(text=text)], structured_content=structured
            )
        kept.pop()

    return error_result(
        "read_file single line exceeds the Gateway output limit; "
        "use a more targeted native file operation.",
        limit=limit,
    )


class Gateway:
    def __init__(self, config: GatewayConfig | None = None):
        self.config = config or GatewayConfig()
        self.execution = NativeExecution(self.config)
        self.native_calls = NativeCalls()
        self.hermes = HermesToolsClient(self.config, native_calls=self.native_calls)
        self.graph = SamchonGraphClient(self.config)
        self.lsp = LSPAdapter(
            allowed_roots=self.config.samchon_graph_allowed_roots,
            timeout_seconds=self.config.timeout_seconds,
        )
        self.memory = MemoryAdapter(
            ai_peer=self.config.memory_ai_peer,
            session_id=self.config.memory_session,
            timeout=self.config.timeout_seconds,
            native_calls=self.native_calls,
        )
        self.capabilities = HermesCapabilities(native_calls=self.native_calls)
        self.startup = StartupContext(
            max_total_bytes=self.config.max_startup_bytes,
            max_output_chars=self.config.max_text_chars,
        )
        self.catalog: dict[str, CatalogEntry] = {}
        self.discovery: ToolDiscovery | None = None
        self.missing_final_tools: set[str] = set()
        self.web_tools_ready = False
        self._deferred_image = False
        self._web_probe_at = time.monotonic()
        self._web_probe_lock = asyncio.Lock()
        self.session_capabilities = SessionCapabilityStore()
        self.jobs: ResumableJobs | None = None
        self._job_reaper_task: asyncio.Task[None] | None = None
        self._session_reaper_task: asyncio.Task[None] | None = None
        self._inflight_requests = 0
        self._session_cleanup_tasks: set[asyncio.Task[None]] = set()

    @property
    def inflight_requests(self) -> int:
        """Active MCP tool handlers; health/readiness endpoints are not counted."""
        return self._inflight_requests

    @staticmethod
    def _payload_has_error(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("error"):
                return True
            if value.get("success") is False:
                return True
            return any(Gateway._payload_has_error(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(Gateway._payload_has_error(item) for item in value)
        return False

    @classmethod
    def _tool_result_has_error(cls, result: types.CallToolResult) -> bool:
        if result.is_error:
            return True
        if result.structured_content is not None and cls._payload_has_error(
            result.structured_content
        ):
            return True
        for item in result.content:
            if not isinstance(item, types.TextContent):
                continue
            try:
                payload = json.loads(item.text)
            except (TypeError, json.JSONDecodeError):
                continue
            if cls._payload_has_error(payload):
                return True
        return False

    async def probe_web_tools(self) -> bool:
        try:
            search, extract = await asyncio.gather(
                asyncio.wait_for(
                    self.hermes.call("web_search", {"query": "OpenAI", "limit": 1}),
                    timeout=self.config.timeout_seconds,
                ),
                asyncio.wait_for(
                    self.hermes.call(
                        "web_extract",
                        {"urls": ["https://example.com"], "char_limit": 2000},
                    ),
                    timeout=self.config.timeout_seconds,
                ),
            )
        except Exception:  # noqa: BLE001 - readiness probe must fail closed
            return False
        return not (
            self._tool_result_has_error(search) or self._tool_result_has_error(extract)
        )

    def _web_probe_fresh(self) -> bool:
        ttl = 300 if self.web_tools_ready else 30
        return time.monotonic() - self._web_probe_at < ttl

    async def refresh_web_tools(self) -> bool:
        # Cheap cached status; only a stale readiness check probes providers.
        if self._web_probe_fresh():
            return self.web_tools_ready
        async with self._web_probe_lock:
            if self._web_probe_fresh():
                return self.web_tools_ready
            try:
                # Provider-backed probes need the tool-call budget, not the short
                # budget used for local listener/discovery health checks. A valid
                # web request routinely exceeds three seconds.
                self.web_tools_ready = await asyncio.wait_for(
                    self.probe_web_tools(), timeout=self.config.timeout_seconds
                )
            except Exception:  # noqa: BLE001 - readiness must fail closed
                self.web_tools_ready = False
            self.missing_final_tools = set() if self.web_tools_ready else set(WEB_TOOLS)
            self._web_probe_at = time.monotonic()
            return self.web_tools_ready

    async def start(self) -> None:
        try:
            await self.hermes.start()
            hermes_tools, graph_tool = await asyncio.gather(
                self.hermes.discover(), self.graph.describe()
            )
            hermes_tools = [t for t in hermes_tools if t.name != "execute_code"]
            hermes_tools.append(execution_schema())
            hermes_names = {tool.name for tool in hermes_tools}
            if "image_generate" not in hermes_names:
                hermes_tools.append(deferred_image_tool())
                hermes_names.add("image_generate")
                self._deferred_image = True
            hermes_missing = missing_expected(hermes_names, HERMES_REQUIRED)
            if not (hermes_names & IMAGE_TOOLS):
                hermes_missing.add("image_generate")
            if hermes_missing:
                raise RuntimeError(
                    f"Hermes missing required tools: {', '.join(sorted(hermes_missing))}"
                )
            web_schema_missing = missing_expected(hermes_names, WEB_TOOLS)
            self.web_tools_ready = (
                not web_schema_missing and await self.probe_web_tools()
            )
            self._web_probe_at = time.monotonic()
            self.missing_final_tools = set() if self.web_tools_ready else set(WEB_TOOLS)
            await self.memory.start()
            self.jobs = ResumableJobs(self.hermes, Path(self.config.job_state_dir or self.config.hermes_home),
                                      allowed_roots=self.config.samchon_graph_allowed_roots)
            self.catalog = build_catalog(
                hermes_tools,
                memory_tool_schemas(),
                [
                    startup_tool_schema(),
                    session_open_tool_schema(),
                    session_close_tool_schema(),
                    *job_tool_schemas(),
                ],
                capability_tools=capability_tool_schemas(),
                graph_tools=[graph_tool],
                lsp_tools=lsp_tool_schemas(),
            )
            self.discovery = ToolDiscovery(self.catalog)
            self._job_reaper_task = asyncio.create_task(self._reap_jobs(), name="gateway-job-reaper")
            self._session_reaper_task = asyncio.create_task(
                self._reap_session_capabilities(),
                name="gateway-session-capability-reaper",
            )
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        if self._job_reaper_task is not None:
            self._job_reaper_task.cancel()
            try:
                await self._job_reaper_task
            except asyncio.CancelledError:
                pass
            self._job_reaper_task = None
        if self._session_reaper_task is not None:
            self._session_reaper_task.cancel()
            try:
                await self._session_reaper_task
            except asyncio.CancelledError:
                pass
            self._session_reaper_task = None
        await self._cleanup_session_records(self.session_capabilities.revoke_all())
        while self.native_calls.active_count or self._session_cleanup_tasks:
            await self.native_calls.wait_idle()
            cleanup_tasks = tuple(self._session_cleanup_tasks)
            if cleanup_tasks:
                await asyncio.shield(asyncio.gather(*cleanup_tasks, return_exceptions=True))
        await self.execution.close()
        await self.hermes.close()
        await self.graph.close()
        await self.native_calls.run(self.lsp.close)

    async def _cleanup_session_owner(self, task_id: str) -> None:
        """Clean only resources keyed by this gateway-generated owner ID."""
        def kill_owned_processes() -> None:
            from tools.process_registry import process_registry

            process_registry.kill_all(
                task_id=task_id,
                source="gateway_session_close",
                consume_output=True,
            )

        try:
            await self.native_calls.run(kill_owned_processes)
        except Exception:  # noqa: BLE001 - cleanup must not break revocation
            logger.warning("Could not clean processes for a gateway session")
        try:
            from tools.terminal_tool_lifecycle import cleanup_vm

            await self.native_calls.run(cleanup_vm, task_id)
        except Exception:  # noqa: BLE001 - do not retain a revoked owner's cwd/env
            logger.warning("Could not clean a gateway-owned terminal environment")
        try:
            from .upstreams import _reset_camofox_session

            await self.native_calls.run(_reset_camofox_session, task_id)
        except Exception:  # noqa: BLE001 - local browser binding cleanup is best effort
            logger.warning("Could not drop a gateway-owned browser binding")

    async def _cleanup_session_records(
        self, records: list[SessionCapability]
    ) -> None:
        # Schedule every revoked owner before yielding. Cancellation of the
        # reaper/request/shutdown waiter must not abandon any claimed cleanup.
        tasks = [self._start_session_cleanup(record) for record in records]
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks))

    def _start_session_cleanup(self, record: SessionCapability) -> asyncio.Task[None]:
        async def finish() -> None:
            await self._cleanup_session_owner(record.task_id)
            self.session_capabilities.cleanup_complete(record)

        # Do not inherit a request's native owner lease into owner cleanup.
        task = asyncio.create_task(
            finish(), name=f"gateway-session-cleanup:{record.task_id}", context=Context()
        )
        self._session_cleanup_tasks.add(task)
        task.add_done_callback(self._session_cleanup_finished)
        return task

    def _release_native_session_token(self, lease: SessionCapability) -> None:
        cleanup = self.session_capabilities.release(lease)
        if cleanup is not None:
            self._start_session_cleanup(cleanup)

    def _session_cleanup_finished(self, task: asyncio.Task[None]) -> None:
        self._session_cleanup_tasks.discard(task)
        if not task.cancelled():
            # Cleanup is best effort and must not echo upstream diagnostics.
            task.exception()

    async def _reap_jobs(self) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                if self.jobs is not None:
                    await self.jobs.maintain()
            except Exception:
                logger.exception("Resumable job maintenance failed")

    async def _reap_session_capabilities(self) -> None:
        while True:
            await asyncio.sleep(min(30, SESSION_TOKEN_TTL_SECONDS))
            await self._cleanup_session_records(self.session_capabilities.expire_due())

    async def acquire_session_token(
        self, token: object
    ) -> SessionCapability | None:
        # Do not acquire a lease before an await: cancellation while cleaning
        # expired owners would otherwise strand the newly acquired reference.
        await self._cleanup_session_records(self.session_capabilities.expire_due())
        return self.session_capabilities.acquire(token)

    async def release_session_token(self, lease: SessionCapability) -> None:
        cleanup = self.session_capabilities.release(lease)
        if cleanup is not None:
            await self._cleanup_session_records([cleanup])

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        task_id: str | None = None,
        session_capability: SessionCapability | None = None,
        via_bridge: bool = False,
    ) -> types.CallToolResult:
        if name in BRIDGE_NAMES:
            if via_bridge or self.discovery is None:
                return error_result(
                    "Tool discovery unavailable", limit=self.config.max_text_chars
                )
            try:
                if name == "tool_search":
                    payload = self.discovery.search(arguments)
                elif name == "tool_describe":
                    payload = self.discovery.describe(arguments)
                else:
                    target, params = self.discovery.validate_call(arguments)
                    return await self.call(
                        target,
                        params,
                        task_id=task_id,
                        session_capability=session_capability,
                        via_bridge=True,
                    )
                return normalize_result(
                    types.CallToolResult(
                        content=[
                            types.TextContent(
                                text=json.dumps(payload, ensure_ascii=False)
                            )
                        ]
                    ),
                    limit=self.config.max_text_chars,
                )
            except ValueError as exc:
                return error_result(str(exc), limit=self.config.max_text_chars)
        entry = self.catalog.get(name)
        if entry is None:
            return error_result(
                f"Unknown or disallowed tool: {name}", limit=self.config.max_text_chars
            )
        if (name in STATEFUL_HERMES_TOOLS or name in {"job_prepare", "job_start"}) and not task_id:
            return error_result(
                "MCP session identity is required for stateful browser/process tools and "
                "job preparation/start. On modern sessionless MCP, discover and call "
                "session_open, then pass its session_token at the top level of each "
                "tool_call. Legacy stateful clients may use their SDK session; "
                "independent commands can use execute_code.",
                limit=self.config.max_text_chars,
            )
        try:
            preserve_images = False
            if entry.source == "hermes":
                if name == "execute_code":
                    result = await self.execution.call(arguments)
                elif name == "image_generate" and self._deferred_image:
                    result = await self.native_calls.run(call_deferred_image, arguments)
                elif name == "vision_analyze":
                    result = await self.hermes.call_vision(arguments)
                    preserve_images = True
                elif name in BROWSER_TOOLS:
                    assert task_id is not None  # enforced before upstream dispatch
                    result = await self.hermes.call_browser(
                        name, arguments, task_id=task_id
                    )
                elif name in DIRECT_HERMES_TOOLS:
                    if name == "terminal":
                        guarded, reason = guard_terminal(arguments)
                        if reason:
                            return error_result(reason, limit=self.config.max_text_chars)
                        assert guarded is not None
                        arguments = guarded
                    result = await self.hermes.call_direct(
                        name, arguments, task_id=task_id
                    )
                    if name == "read_file":
                        result = _structured_read_file_result(
                            result, limit=self.config.max_text_chars
                        )
                else:
                    result = await self.hermes.call(name, arguments)
                if name in WEB_TOOLS and self._tool_result_has_error(result):
                    self.web_tools_ready = False
                    self.missing_final_tools = set(WEB_TOOLS)
                    self._web_probe_at = 0.0
            elif entry.source == "graph":
                result = await self.graph.call(arguments)
            elif entry.source == "lsp":
                payload = await self.native_calls.run(self.lsp.call, name, arguments)
                result = types.CallToolResult(
                    content=[
                        types.TextContent(
                            text=json.dumps(payload, ensure_ascii=False, default=str)
                        )
                    ],
                    structured_content=payload,
                )
            elif entry.source == "capability":
                result = await self.capabilities.call(name, arguments)
            elif entry.source == "memory":
                result = await self.memory.call(name, arguments)
            elif entry.source == "gateway":
                if name == "startup_context":
                    if arguments:
                        return error_result(
                            "startup_context does not accept arguments",
                            limit=self.config.max_text_chars,
                        )
                    payload = await self.native_calls.run(self.startup.read)
                elif name in {"job_prepare", "job_start", "job_status", "job_cancel"}:
                    if self.jobs is None:
                        return error_result("job service not started", limit=self.config.max_text_chars)
                    payload = await self.jobs.call(
                        name,
                        arguments,
                        caller_id=task_id if name in {"job_prepare", "job_start"} else None,
                    )
                    if name in {"job_prepare", "job_start"}:
                        # Credential issuance is intentional and must bypass output
                        # redaction; credentials are never included in logs.
                        return types.CallToolResult(
                            content=[types.TextContent(text=json.dumps(payload))]
                        )
                elif name == "session_open":
                    if arguments:
                        return error_result(
                            "session_open does not accept arguments",
                            limit=self.config.max_text_chars,
                        )
                    expired = self.session_capabilities.expire_due()
                    await self._cleanup_session_records(expired)
                    try:
                        token, _record = self.session_capabilities.issue()
                    except RuntimeError as exc:
                        return error_result(
                            str(exc), limit=self.config.max_text_chars
                        )
                    # This is the one intentional credential-bearing response.
                    # Do not run it through output redaction or log its contents.
                    return types.CallToolResult(
                        content=[
                            types.TextContent(
                                text=json.dumps(
                                    {
                                        "session_token": token,
                                        "expires_in_seconds": SESSION_TOKEN_TTL_SECONDS,
                                    }
                                )
                            )
                        ]
                    )
                elif name == "session_close":
                    if arguments:
                        return error_result(
                            "session_close does not accept arguments",
                            limit=self.config.max_text_chars,
                        )
                    if session_capability is None:
                        return error_result(
                            "session_close requires a valid session_token",
                            limit=self.config.max_text_chars,
                        )
                    self.session_capabilities.revoke(session_capability)
                    payload = {"closed": True}
                else:
                    raise ValueError(f"unsupported gateway tool: {name}")
                result = types.CallToolResult(
                    content=[
                        types.TextContent(text=json.dumps(payload, ensure_ascii=False))
                    ]
                )
            else:
                raise RuntimeError(f"unsupported catalog source: {entry.source}")
            return normalize_result(
                result,
                limit=self.config.max_text_chars,
                preserve_images=preserve_images,
            )
        except asyncio.CancelledError:
            # Cancellation is a request outcome, not an upstream tool error.
            raise
        except TimeoutError:
            if name in WEB_TOOLS:
                self.web_tools_ready = False
                self.missing_final_tools = set(WEB_TOOLS)
                self._web_probe_at = 0.0
            logger.warning(
                "gateway_tool_finished source=%s tool=%s category=timeout",
                entry.source,
                name,
            )
            return error_result(
                f"{entry.source} upstream timed out",
                limit=self.config.max_text_chars,
                category="timeout",
            )
        except Exception as exc:  # noqa: BLE001 - upstream boundary must normalize failures
            if name in WEB_TOOLS:
                self.web_tools_ready = False
                self.missing_final_tools = set(WEB_TOOLS)
                self._web_probe_at = 0.0
            logger.warning(
                "gateway_tool_finished source=%s tool=%s category=tool_error",
                entry.source,
                name,
            )
            return error_result(
                f"{entry.source} upstream unavailable or failed: {exc}",
                limit=self.config.max_text_chars,
            )

    async def health(self) -> dict[str, Any]:
        async def get_ok(url: str) -> bool:
            try:
                async with httpx.AsyncClient(
                    timeout=self.config.health_timeout_seconds
                ) as client:
                    response = await client.get(url)
                    return response.status_code == 200
            except Exception:  # noqa: BLE001 - health probe must degrade, not crash
                return False

        async def hermes_state() -> tuple[bool, bool]:
            try:
                tools = await asyncio.wait_for(
                    self.hermes.discover(), timeout=self.config.health_timeout_seconds
                )
                names = {tool.name for tool in tools}
                names.add("execute_code")
                if self._deferred_image and "image_generate" in self.catalog:
                    names.add("image_generate")
                missing = missing_expected(names, HERMES_REQUIRED)
                if not (names & IMAGE_TOOLS):
                    missing.add("image_generate")
                core_ok = not missing
                web_schema_ok = not missing_expected(names, WEB_TOOLS)
                return core_ok, bool(
                    core_ok and web_schema_ok and await self.refresh_web_tools()
                )
            except Exception:  # noqa: BLE001 - health probe must degrade, not crash
                return False, False

        async def graph_state() -> bool:
            try:
                return await asyncio.wait_for(
                    self.graph.healthy(), timeout=self.config.health_timeout_seconds
                )
            except Exception:  # noqa: BLE001 - health probe must degrade, not crash
                return False

        honcho_ok, hermes_status, graph_ready = await asyncio.gather(
            get_ok(self.config.honcho_health_url),
            hermes_state(),
            graph_state(),
        )
        hermes_core_ok, web_tools_ok = hermes_status
        components = {
            "hermes_tools": hermes_core_ok,
            "web_tools": web_tools_ok,
            "samchon_graph": graph_ready,
            "hermes_lsp": self.lsp.healthy(),
            "hermes_capabilities": self.capabilities.check(),
            "honcho": honcho_ok,
            "startup_context": self.startup.check(),
        }
        return {
            "status": "ok" if all(components.values()) else "degraded",
            "components": components,
            "missing_final_tools": sorted(self.missing_final_tools),
        }


def create_app(config: GatewayConfig | None = None):
    gateway = Gateway(config)

    @asynccontextmanager
    async def lifespan(_: Server[Any]):
        await gateway.start()
        try:
            yield {"gateway": gateway}
        finally:
            await gateway.close()

    async def list_tools(_ctx, _params):
        return types.ListToolsResult(tools=public_schemas(gateway.catalog))

    async def call_tool(_ctx, params):
        gateway._inflight_requests += 1
        request_id = secrets.token_hex(8)
        started = time.monotonic()
        result_category = "unknown"
        trace_name = "invalid"

        request = getattr(_ctx, "request", None)
        headers = getattr(request, "headers", None)
        try:
            inbound_id = headers.get("x-request-id") if headers is not None else None
        except Exception:  # noqa: BLE001 - correlation metadata is untrusted
            inbound_id = None
        correlation_id = (
            "sha256:" + hashlib.sha256(inbound_id.encode()).hexdigest()[:16]
            if isinstance(inbound_id, str)
            and len(inbound_id) <= 64
            and re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", inbound_id)
            else "none"
        )

        if params.name == "tool_call":
            initial_arguments = params.arguments or {}
            calls = initial_arguments.get("calls", []) if isinstance(initial_arguments, dict) else []
            selected = (
                calls[0].get("name")
                if isinstance(calls, list)
                and len(calls) == 1
                and isinstance(calls[0], dict)
                else None
            )
            trace_name = selected if selected in gateway.catalog else "invalid"
        elif params.name in PUBLIC_NAMES:
            trace_name = params.name

        def categorize(result: types.CallToolResult) -> str:
            if result.is_error:
                metadata = result.structured_content
                category = (
                    metadata.get("_gateway_error_category")
                    if isinstance(metadata, dict)
                    else None
                )
                if category in {"timeout", "tool_error"}:
                    return category
                return "tool_error"
            if gateway._tool_result_has_error(result):
                return "tool_error"
            values = [result.structured_content]
            for item in result.content:
                if isinstance(item, types.TextContent):
                    try:
                        values.append(json.loads(item.text))
                    except (TypeError, json.JSONDecodeError):
                        continue
            if any(isinstance(value, dict) and value.get("state") == "unknown" for value in values):
                return "unknown"
            return "success"

        async def dispatch():
            if params.name not in PUBLIC_NAMES:
                return error_result(
                    f"Unknown or disallowed tool: {params.name}",
                    limit=gateway.config.max_text_chars,
                )
            arguments = params.arguments or {}
            if params.name != "tool_call":
                return await gateway.call(params.name, arguments)

            arguments = dict(arguments)
            token_present = "session_token" in arguments
            token = arguments.pop("session_token", None)
            calls = arguments.get("calls", [])
            selected_for_auth = (
                calls[0].get("name")
                if isinstance(calls, list)
                and len(calls) == 1
                and isinstance(calls[0], dict)
                else None
            )
            recovery_call = selected_for_auth in {"job_status", "job_cancel"}
            lease = None
            if token_present and not recovery_call:
                lease = await gateway.acquire_session_token(token)
                if lease is None:
                    return error_result(
                        "Invalid or expired session token",
                        limit=gateway.config.max_text_chars,
                    )
            task_id = (
                lease.task_id
                if lease is not None
                else None
                if recovery_call
                else browser_task_id(_ctx)
            )
            try:
                if lease is not None and selected_for_auth in STATEFUL_HERMES_TOOLS:
                    with gateway.native_calls.owner_lease(
                        retain=lambda: gateway.session_capabilities.retain(lease),
                        release=lambda: gateway._release_native_session_token(lease),
                    ):
                        return await gateway.call(
                            params.name,
                            arguments,
                            task_id=task_id,
                            session_capability=lease,
                        )
                return await gateway.call(
                    params.name,
                    arguments,
                    task_id=task_id,
                    session_capability=lease,
                )
            finally:
                if lease is not None:
                    await asyncio.shield(gateway.release_session_token(lease))

        try:
            result = await dispatch()
            result_category = categorize(result)
            return result
        except asyncio.CancelledError:
            result_category = "cancelled"
            raise
        except TimeoutError:
            result_category = "timeout"
            return error_result(
                "Gateway request timed out",
                limit=gateway.config.max_text_chars,
                category="timeout",
            )
        except Exception:  # noqa: BLE001 - transport boundary returns a safe failure
            result_category = "unknown"
            return error_result(
                "Gateway request failed",
                limit=gateway.config.max_text_chars,
            )
        finally:
            gateway._inflight_requests -= 1
            logger.info(
                "gateway_request request_id=%s correlation_id=%s tool=%s elapsed_ms=%d result=%s",
                request_id,
                correlation_id,
                trace_name,
                int((time.monotonic() - started) * 1000),
                result_category,
            )

    server = Server(
        "hermes-mcp-gateway",
        version="0.1.0",
        instructions=(
            "Single default-deny MCP gateway for native execution, curated Hermes "
            "tools, cwd-scoped Samchon Graph semantics, guarded Hermes capabilities, and Honcho memory. "
            "Use tool_search, tool_describe, then tool_call for deferred tools. For browser, "
            "terminal and process continuity on modern sessionless MCP, discover and call "
            "session_open once, pass its returned session_token at the top level of every "
            "related tool_call, and use session_close when finished. For long commands, "
            "call job_prepare first, securely save its one-time recovery_token before "
            "starting work, then call job_start with the prepared "
            "job_id and recovery_token. job_prepare and job_start require a verified "
            "interactive session; renew an expired session with session_open before "
            "starting. Use job_status or job_cancel with the recovery credential from "
            "any session or no session. After disconnect, check job_status before any "
            "retry. Backend deadlines are not uniform: Hermes MCP reads and LSP RPCs "
            "default to 30 seconds, Samchon Graph calls may use 120 seconds, and "
            "native execute_code has a 30-second worker deadline. Thread-backed native "
            "browser, vision, and direct tool calls are not force-stopped by Gateway "
            "cancellation. Never blindly replay a mutation."
        ),
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )

    async def healthz(_request: Request):
        # Liveness must not run upstream discovery, LSP, or paid web probes.
        return JSONResponse(
            {"status": "ok", "inflight_requests": gateway.inflight_requests,
             "inflight_native_calls": gateway.native_calls.active_count}
        )

    async def readyz(_request: Request):
        payload = await gateway.health()
        return JSONResponse(
            payload, status_code=200 if payload["status"] == "ok" else 503
        )

    async def oauth_protected_resource(_request: Request):
        return JSONResponse(
            {"error": "not_supported", "error_description": "OAuth not configured"},
            status_code=404,
        )

    return server.streamable_http_app(
        streamable_http_path="/mcp",
        # MCP 2.0's modern JSON-only path does not watch HTTP disconnects.
        # The default JSON/SSE path cancels the handler and downstream HTTP call.
        json_response=False,
        stateless_http=False,
        host=gateway.config.host,
        custom_starlette_routes=[
            Route("/healthz", healthz),
            Route("/readyz", readyz),
            Route("/.well-known/oauth-protected-resource", oauth_protected_resource),
            Route("/.well-known/oauth-protected-resource/mcp", oauth_protected_resource),
        ],
    )


def main() -> None:
    config = GatewayConfig()
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
