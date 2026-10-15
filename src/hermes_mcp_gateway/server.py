from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from agent.redact import redact_sensitive_text
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
from .lsp import LSPAdapter, lsp_tool_schemas
from .memory import MemoryAdapter, memory_tool_schemas
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
            if source == "hermes" and (
                name == "vision_analyze" or name in BROWSER_TOOLS
            ):
                public_tool = public_tool.model_copy(update={"output_schema": None})
            catalog[name] = CatalogEntry(tool=public_tool, source=source)
    return catalog


def _sanitize_structure(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, dict):
        return {
            redact_sensitive_text(str(key), force=True): _sanitize_structure(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_structure(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_sensitive_text(str(value), force=True)


def normalize_result(
    result: types.CallToolResult, *, limit: int, preserve_images: bool = False
) -> types.CallToolResult:
    content: list[Any] = []
    for item in result.content:
        if isinstance(item, types.TextContent):
            text = redact_sensitive_text(item.text, force=True)
            if len(text) > limit:
                suffix = "\n...[truncated by hermes-mcp-gateway]"
                text = text[: max(0, limit - len(suffix))] + suffix
            content.append(types.TextContent(text=text))
        elif preserve_images and isinstance(item, types.ImageContent):
            content.append(item)
        else:
            marker = (
                f"[{type(item).__name__} omitted by hermes-mcp-gateway output policy]"
            )
            content.append(types.TextContent(text=marker[:limit]))
    structured = result.structured_content
    if structured is not None:
        structured = _sanitize_structure(structured)
        rendered = json.dumps(structured, ensure_ascii=False, default=str)
        if len(rendered) > limit:
            structured = {
                "truncated": True,
                "preview": rendered[: max(0, limit - 64)],
            }
    return types.CallToolResult(
        content=content, structured_content=structured, is_error=result.is_error
    )


def error_result(message: str, *, limit: int) -> types.CallToolResult:
    text = redact_sensitive_text(str(message), force=True)
    if len(text) > limit:
        text = text[:limit]
    return types.CallToolResult(content=[types.TextContent(text=text)], is_error=True)


class Gateway:
    def __init__(self, config: GatewayConfig | None = None):
        self.config = config or GatewayConfig()
        self.execution = NativeExecution(self.config)
        self.hermes = HermesToolsClient(self.config)
        self.graph = SamchonGraphClient(self.config)
        self.lsp = LSPAdapter(
            allowed_roots=self.config.samchon_graph_allowed_roots,
            timeout_seconds=self.config.timeout_seconds,
        )
        self.memory = MemoryAdapter(
            ai_peer=self.config.memory_ai_peer,
            session_id=self.config.memory_session,
            timeout=self.config.timeout_seconds,
        )
        self.capabilities = HermesCapabilities()
        self.startup = StartupContext(max_total_bytes=self.config.max_startup_bytes)
        self.catalog: dict[str, CatalogEntry] = {}
        self.discovery: ToolDiscovery | None = None
        self.missing_final_tools: set[str] = set()
        self.web_tools_ready = False
        self._deferred_image = False
        self._web_probe_at = time.monotonic()
        self._web_probe_lock = asyncio.Lock()
        self.session_capabilities = SessionCapabilityStore()
        self._session_reaper_task: asyncio.Task[None] | None = None

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
            self.catalog = build_catalog(
                hermes_tools,
                memory_tool_schemas(),
                [
                    startup_tool_schema(),
                    session_open_tool_schema(),
                    session_close_tool_schema(),
                ],
                capability_tools=capability_tool_schemas(),
                graph_tools=[graph_tool],
                lsp_tools=lsp_tool_schemas(),
            )
            self.discovery = ToolDiscovery(self.catalog)
            self._session_reaper_task = asyncio.create_task(
                self._reap_session_capabilities(),
                name="gateway-session-capability-reaper",
            )
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        if self._session_reaper_task is not None:
            self._session_reaper_task.cancel()
            try:
                await self._session_reaper_task
            except asyncio.CancelledError:
                pass
            self._session_reaper_task = None
        await self._cleanup_session_records(self.session_capabilities.revoke_all())
        await self.execution.close()
        await self.hermes.close()
        await self.graph.close()
        await asyncio.to_thread(self.lsp.close)

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
            await asyncio.to_thread(kill_owned_processes)
        except Exception:  # noqa: BLE001 - cleanup must not break revocation
            logger.warning("Could not clean processes for a gateway session")
        try:
            from tools.terminal_tool_lifecycle import cleanup_vm

            await asyncio.to_thread(cleanup_vm, task_id)
        except Exception:  # noqa: BLE001 - do not retain a revoked owner's cwd/env
            logger.warning("Could not clean a gateway-owned terminal environment")
        try:
            from .upstreams import _reset_camofox_session

            await asyncio.to_thread(_reset_camofox_session, task_id)
        except Exception:  # noqa: BLE001 - local browser binding cleanup is best effort
            logger.warning("Could not drop a gateway-owned browser binding")

    async def _cleanup_session_records(
        self, records: list[SessionCapability]
    ) -> None:
        for record in records:
            await self._cleanup_session_owner(record.task_id)

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
        if name in STATEFUL_HERMES_TOOLS and not task_id:
            return error_result(
                "MCP session identity is required for stateful browser/process tools. "
                "On modern sessionless MCP, discover and call session_open, then pass "
                "its session_token at the top level of each tool_call. Legacy stateful "
                "clients may use their SDK session; independent commands can use execute_code.",
                limit=self.config.max_text_chars,
            )
        try:
            preserve_images = False
            if entry.source == "hermes":
                if name == "execute_code":
                    result = await self.execution.call(arguments)
                elif name == "image_generate" and self._deferred_image:
                    result = await asyncio.to_thread(call_deferred_image, arguments)
                elif name == "vision_analyze":
                    result = await self.hermes.call_vision(arguments)
                    preserve_images = True
                elif name in BROWSER_TOOLS:
                    assert task_id is not None  # enforced before upstream dispatch
                    result = await self.hermes.call_browser(
                        name, arguments, task_id=task_id
                    )
                elif name in DIRECT_HERMES_TOOLS:
                    result = await self.hermes.call_direct(
                        name, arguments, task_id=task_id
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
                payload = await asyncio.to_thread(self.lsp.call, name, arguments)
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
                    payload = await asyncio.to_thread(self.startup.read)
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
        except Exception as exc:  # noqa: BLE001 - upstream boundary must normalize failures
            if name in WEB_TOOLS:
                self.web_tools_ready = False
                self.missing_final_tools = set(WEB_TOOLS)
                self._web_probe_at = 0.0
            logger.warning(
                "upstream tool call failed: source=%s tool=%s error=%s",
                entry.source,
                name,
                redact_sensitive_text(str(exc), force=True),
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
        lease = None
        if token_present:
            lease = await gateway.acquire_session_token(token)
            if lease is None:
                return error_result(
                    "Invalid or expired session token",
                    limit=gateway.config.max_text_chars,
                )
        task_id = lease.task_id if lease is not None else browser_task_id(_ctx)
        try:
            return await gateway.call(
                params.name,
                arguments,
                task_id=task_id,
                session_capability=lease,
            )
        finally:
            if lease is not None:
                await gateway.release_session_token(lease)

    server = Server(
        "hermes-mcp-gateway",
        version="0.1.0",
        instructions=(
            "Single default-deny MCP gateway for native execution, curated Hermes "
            "tools, cwd-scoped Samchon Graph semantics, guarded Hermes capabilities, and Honcho memory. "
            "Use tool_search, tool_describe, then tool_call for deferred tools. For browser, "
            "terminal and process continuity on modern sessionless MCP, discover and call "
            "session_open once, pass its returned session_token at the top level of every "
            "related tool_call, and use session_close when finished."
        ),
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )

    async def healthz(_request: Request):
        # Liveness must not run upstream discovery, LSP, or paid web probes.
        return JSONResponse({"status": "ok"})

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
