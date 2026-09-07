from __future__ import annotations

import asyncio
import json
import logging
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

from .config import (
    CONTEXT_REQUIRED,
    HERMES_ALLOWLIST,
    HERMES_REQUIRED,
    WEB_TOOLS,
    GatewayConfig,
)
from .capabilities import HermesCapabilities, capability_tool_schemas
from .memory import MemoryAdapter, memory_tool_schemas
from .startup import StartupContext, startup_tool_schema
from .upstreams import ContextModeClient, HermesToolsClient, missing_expected

logger = logging.getLogger("hermes_mcp_gateway")


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    tool: types.Tool
    source: str
    upstream_name: str


def build_catalog(
    context_tools: list[types.Tool],
    hermes_tools: list[types.Tool],
    memory_tools: list[types.Tool],
    gateway_tools: list[types.Tool] | None = None,
    *,
    capability_tools: list[types.Tool] | None = None,
) -> dict[str, CatalogEntry]:
    catalog: dict[str, CatalogEntry] = {}
    groups = (
        ("context", context_tools, None),
        ("hermes", hermes_tools, HERMES_ALLOWLIST),
        ("capability", capability_tools or [], None),
        ("memory", memory_tools, None),
        ("gateway", gateway_tools or [], None),
    )
    for source, tools, allowlist in groups:
        for tool in tools:
            if allowlist is not None and tool.name not in allowlist:
                continue
            if tool.name in catalog:
                raise ValueError(f"tool collision: {tool.name}")
            public_tool = tool
            if source == "hermes" and tool.name == "vision_analyze":
                public_tool = tool.model_copy(update={"output_schema": None})
            catalog[tool.name] = CatalogEntry(
                tool=public_tool, source=source, upstream_name=tool.name
            )
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
        self.context = ContextModeClient(
            self.config.context_url, self.config.timeout_seconds
        )
        self.hermes = HermesToolsClient(self.config)
        self.memory = MemoryAdapter(
            ai_peer=self.config.memory_ai_peer,
            session_id=self.config.memory_session,
            timeout=self.config.timeout_seconds,
        )
        self.capabilities = HermesCapabilities()
        self.startup = StartupContext(max_total_bytes=self.config.max_startup_bytes)
        self.catalog: dict[str, CatalogEntry] = {}
        self.missing_final_tools: set[str] = set()
        self.web_tools_ready = False

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
                        {"urls": ["https://example.com"], "format": "markdown"},
                    ),
                    timeout=self.config.timeout_seconds,
                ),
            )
        except Exception:  # noqa: BLE001 - readiness probe must fail closed
            return False
        return not (
            self._tool_result_has_error(search) or self._tool_result_has_error(extract)
        )

    async def start(self) -> None:
        try:
            await self.hermes.start()
            context_tools, hermes_tools = await asyncio.gather(
                self.context.discover(), self.hermes.discover()
            )
            context_names = {tool.name for tool in context_tools}
            hermes_names = {tool.name for tool in hermes_tools}
            context_missing = missing_expected(context_names, CONTEXT_REQUIRED)
            hermes_missing = missing_expected(hermes_names, HERMES_REQUIRED)
            if context_missing:
                raise RuntimeError(
                    f"Context Mode missing required tools: {', '.join(sorted(context_missing))}"
                )
            if hermes_missing:
                raise RuntimeError(
                    f"Hermes missing required tools: {', '.join(sorted(hermes_missing))}"
                )
            web_schema_missing = missing_expected(hermes_names, WEB_TOOLS)
            self.web_tools_ready = (
                not web_schema_missing and await self.probe_web_tools()
            )
            self.missing_final_tools = set() if self.web_tools_ready else set(WEB_TOOLS)
            await self.memory.start()
            self.catalog = build_catalog(
                context_tools,
                hermes_tools,
                memory_tool_schemas(),
                [startup_tool_schema()],
                capability_tools=capability_tool_schemas(),
            )
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        await self.context.close()
        await self.hermes.close()

    async def call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        entry = self.catalog.get(name)
        if entry is None:
            return error_result(
                f"Unknown or disallowed tool: {name}", limit=self.config.max_text_chars
            )
        try:
            preserve_images = False
            if entry.source == "context":
                result = await self.context.call(entry.upstream_name, arguments)
            elif entry.source == "hermes":
                if entry.upstream_name == "vision_analyze":
                    result = await self.hermes.call_vision(arguments)
                    preserve_images = True
                else:
                    result = await self.hermes.call(entry.upstream_name, arguments)
            elif entry.source == "capability":
                result = await self.capabilities.call(entry.upstream_name, arguments)
            elif entry.source == "memory":
                result = await self.memory.call(entry.upstream_name, arguments)
            elif entry.source == "gateway":
                if arguments:
                    return error_result(
                        "startup_context does not accept arguments",
                        limit=self.config.max_text_chars,
                    )
                payload = await asyncio.to_thread(self.startup.read)
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
                core_ok = not missing_expected(names, HERMES_REQUIRED)
                web_schema_ok = not missing_expected(names, WEB_TOOLS)
                return core_ok, bool(core_ok and web_schema_ok and self.web_tools_ready)
            except Exception:  # noqa: BLE001 - health probe must degrade, not crash
                return False, False

        context_ok, honcho_ok, hermes_state = await asyncio.gather(
            get_ok(self.config.context_ready_url),
            get_ok(self.config.honcho_health_url),
            hermes_state(),
        )
        hermes_core_ok, web_tools_ok = hermes_state
        components = {
            "context_mode": context_ok,
            "hermes_tools": hermes_core_ok,
            "web_tools": web_tools_ok,
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
        return types.ListToolsResult(
            tools=[entry.tool for entry in gateway.catalog.values()]
        )

    async def call_tool(_ctx, params):
        return await gateway.call(params.name, params.arguments or {})

    server = Server(
        "hermes-mcp-gateway",
        version="0.1.0",
        instructions=(
            "Single default-deny MCP gateway for Context Mode, curated Hermes "
            "tools, guarded Hermes capabilities, and Honcho memory."
        ),
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )

    async def healthz(_request: Request):
        return JSONResponse(await gateway.health())

    async def readyz(_request: Request):
        payload = await gateway.health()
        return JSONResponse(
            payload, status_code=200 if payload["status"] == "ok" else 503
        )

    return server.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=False,
        host=gateway.config.host,
        custom_starlette_routes=[Route("/healthz", healthz), Route("/readyz", readyz)],
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
