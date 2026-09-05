from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from typing import Any

from agent.redact import redact_sensitive_text
from mcp import types
from plugins.memory.honcho import ALL_TOOL_SCHEMAS
from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client
from plugins.memory.honcho.session import HonchoSessionManager

_DURABLE_KINDS = {"preference", "decision", "architecture", "project_state"}
_TEMPORARY_RE = re.compile(
    r"\b(for this chat|this conversation|temporary|temporarily|for now|today only|"
    r"bu konuşma|bu sohbet|geçici|şimdilik|sadece bugün)\b",
    re.IGNORECASE,
)


def resolve_memory_config(
    base: HonchoClientConfig, *, ai_peer: str
) -> HonchoClientConfig:
    return replace(base, ai_peer=ai_peer)


def memory_tool_schemas() -> list[types.Tool]:
    tools: list[types.Tool] = []
    for schema in ALL_TOOL_SCHEMAS:
        source_name = schema["name"]
        public_name = source_name.replace("honcho_", "memory_", 1)
        if public_name not in {
            "memory_profile",
            "memory_search",
            "memory_context",
            "memory_reasoning",
            "memory_conclude",
        }:
            continue
        input_schema = json.loads(json.dumps(schema["parameters"]))
        if public_name == "memory_profile":
            input_schema.get("properties", {}).pop("card", None)
        if public_name == "memory_conclude":
            input_schema.setdefault("properties", {})["kind"] = {
                "type": "string",
                "enum": sorted(_DURABLE_KINDS),
                "description": "Required when creating a conclusion: durable memory category.",
            }
        tools.append(
            types.Tool(
                name=public_name,
                description=(schema.get("description") or "").replace(
                    "Honcho", "memory"
                ),
                input_schema=input_schema,
            )
        )
    return tools


class MemoryWritePolicy:
    def validate(self, args: dict[str, Any]) -> tuple[bool, str]:
        conclusion = str(args.get("conclusion") or "").strip()
        delete_id = str(args.get("delete_id") or "").strip()
        list_mode = bool(args.get("list"))
        if sum((bool(conclusion), bool(delete_id), list_mode)) != 1:
            return (
                False,
                "Exactly one of conclusion, delete_id, or list must be provided.",
            )
        if not conclusion:
            return True, ""
        kind = str(args.get("kind") or "").strip()
        if kind not in _DURABLE_KINDS:
            return False, "A durable memory kind is required for conclusion writes."
        if len(conclusion) > 2000:
            return False, "Conclusion is too large for durable memory."
        if redact_sensitive_text(conclusion, force=True) != conclusion:
            return False, "Sensitive data must not be persisted to memory."
        if _TEMPORARY_RE.search(conclusion):
            return (
                False,
                "Temporary conversation details must not be persisted to memory.",
            )
        return True, ""


class MemoryAdapter:
    def __init__(
        self,
        *,
        ai_peer: str = "chatgpt",
        session_id: str = "chatgpt",
        timeout: float = 30.0,
    ):
        self.ai_peer = ai_peer
        self.session_id = session_id
        self.timeout = timeout
        self.config: HonchoClientConfig | None = None
        self.manager: HonchoSessionManager | None = None
        self.session_key = ""
        self.policy = MemoryWritePolicy()

    async def start(self) -> None:
        base = HonchoClientConfig.from_global_config()
        if not base.enabled or not (base.api_key or base.base_url):
            raise RuntimeError("Honcho memory is not configured")
        config = resolve_memory_config(base, ai_peer=self.ai_peer)
        session_key = (
            config.resolve_session_name(
                session_id=self.session_id,
                gateway_session_key=self.session_id,
            )
            or self.session_id
        )
        manager = HonchoSessionManager(
            honcho=get_honcho_client(config),
            config=config,
            context_tokens=config.context_tokens,
        )
        await asyncio.wait_for(
            asyncio.to_thread(manager.get_or_create, session_key), timeout=self.timeout
        )
        self.config = config
        self.manager = manager
        self.session_key = session_key

    async def _run(self, fn, *args, **kwargs):
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs), timeout=self.timeout
        )

    async def call(self, name: str, args: dict[str, Any]) -> types.CallToolResult:
        if self.manager is None or not self.session_key:
            raise RuntimeError("Honcho memory adapter is not started")
        manager = self.manager
        peer = str(args.get("peer") or "user")

        if name == "memory_profile":
            if "card" in args:
                return _text_result(
                    {
                        "error": "memory_profile is read-only; use memory_conclude for durable writes"
                    },
                    error=True,
                )
            card = await self._run(manager.get_peer_card, self.session_key, peer=peer)
            return _text_result({"result": card or []})

        if name == "memory_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return _text_result(
                    {"error": "Missing required parameter: query"}, error=True
                )
            max_tokens = min(int(args.get("max_tokens", 800)), 2000)
            result = await self._run(
                manager.search_context,
                self.session_key,
                query,
                max_tokens=max_tokens,
                peer=peer,
            )
            return _text_result({"result": result or "No relevant context found."})

        if name == "memory_reasoning":
            query = str(args.get("query") or "").strip()
            if not query:
                return _text_result(
                    {"error": "Missing required parameter: query"}, error=True
                )
            result = await self._run(
                manager.dialectic_query,
                self.session_key,
                query,
                reasoning_level=args.get("reasoning_level"),
                peer=peer,
                apply_injection_cap=False,
                raise_errors=True,
            )
            return _text_result({"result": result or "No reasoning result returned."})

        if name == "memory_context":
            ctx = await self._run(
                manager.get_session_context, self.session_key, peer=peer
            )
            return _text_result({"result": ctx or {}})

        if name == "memory_conclude":
            ok, error = self.policy.validate(args)
            if not ok:
                return _text_result({"error": error}, error=True)
            delete_id = str(args.get("delete_id") or "").strip()
            conclusion = str(args.get("conclusion") or "").strip()
            if bool(args.get("list")):
                conclusions = await self._run(
                    manager.list_conclusions,
                    self.session_key,
                    query=str(args.get("query") or "").strip() or None,
                    peer=peer,
                )
                return _text_result({"conclusions": conclusions})
            if delete_id:
                deleted = await self._run(
                    manager.delete_conclusion, self.session_key, delete_id, peer=peer
                )
                payload = (
                    {"result": f"Conclusion {delete_id} deleted."}
                    if deleted
                    else {"error": "Conclusion deletion failed"}
                )
                return _text_result(payload, error=not deleted)
            saved = await self._run(
                manager.create_conclusion, self.session_key, conclusion, peer=peer
            )
            payload = (
                {"result": f"Conclusion saved for {peer}."}
                if saved
                else {"error": "Conclusion write failed"}
            )
            return _text_result(payload, error=not saved)

        return _text_result({"error": f"Unknown memory tool: {name}"}, error=True)


def _text_result(payload: Any, *, error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(text=json.dumps(payload, ensure_ascii=False))],
        is_error=error,
    )
