from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from mcp import types
from tools.cronjob_tools import cronjob
from tools.delegate_tool import delegate_task
from tools.session_search_tool import session_search

_SESSION_ALLOWED = frozenset(
    {
        "query",
        "role_filter",
        "limit",
        "session_id",
        "around_message_id",
        "window",
        "sort",
        "detail",
    }
)
_CRON_ALLOWED = frozenset(
    {
        "action",
        "job_id",
        "prompt",
        "schedule",
        "name",
        "repeat",
        "deliver",
        "include_disabled",
        "skill",
        "skills",
        "context_from",
        "continuity",
        "workdir",
        "attach_to_session",
        "confirmed",
    }
)
_CRON_MUTATIONS = frozenset({"create", "update", "pause", "resume", "remove", "run"})
_DELEGATE_ALLOWED = frozenset({"goal", "context", "tasks", "confirmed"})
_SAFE_DELEGATION_TOOLSETS = ["web", "vision", "skills", "mcp-context-mode"]


def _text_result(value: str, *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=value)], is_error=is_error)


def _hermes_result(value: str) -> types.CallToolResult:
    is_error = False
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            is_error = payload.get("success") is False or bool(payload.get("error"))
    except (TypeError, json.JSONDecodeError):
        pass
    return _text_result(value, is_error=is_error)


def _error(message: str) -> types.CallToolResult:
    return _text_result(json.dumps({"success": False, "error": message}), is_error=True)


def _reject_unknown(
    arguments: dict[str, Any], allowed: frozenset[str]
) -> types.CallToolResult | None:
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        return _error(f"unsupported argument(s): {', '.join(unknown)}")
    return None


def capability_tool_schemas() -> list[types.Tool]:
    return [
        types.Tool(
            name="session_search",
            description=(
                "Search Hermes conversation history stored in the local session database. "
                "This is historical conversation context, not proof of current external state."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "role_filter": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
                    "session_id": {"type": "string"},
                    "around_message_id": {"type": "integer"},
                    "window": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                    "sort": {"type": "string"},
                    "detail": {
                        "type": "string",
                        "enum": ["adaptive", "full"],
                        "default": "adaptive",
                    },
                },
            },
        ),
        types.Tool(
            name="delegate_task",
            description=(
                "Run one or two Hermes leaf subagents synchronously in isolated contexts. "
                "Spawning costs model inference and requires confirmed=true. Children may use "
                "web, vision, skills, and Context Mode MCP tools; native shell/file/code tools "
                "and recursive delegation are not exposed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "context": {"type": "string"},
                    "tasks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "goal": {"type": "string"},
                                "context": {"type": "string"},
                            },
                            "required": ["goal"],
                            "additionalProperties": False,
                        },
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": (
                            "Must be true after explicit user approval because "
                            "delegation consumes model inference."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="cronjob",
            description=(
                "Manage Hermes scheduled jobs. list is read-only. create/update/pause/resume/remove/run "
                "are state-changing and require confirmed=true. Unsafe script/provider/model routing "
                "parameters are intentionally unavailable. Delivery defaults to local."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "update", "pause", "resume", "remove", "run"],
                    },
                    "job_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "schedule": {"type": "string"},
                    "name": {"type": "string"},
                    "repeat": {"type": "integer", "minimum": 1},
                    "deliver": {"type": "string", "default": "local"},
                    "include_disabled": {"type": "boolean", "default": True},
                    "skill": {"type": "string"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "context_from": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                    "continuity": {"type": "boolean"},
                    "workdir": {"type": "string"},
                    "attach_to_session": {"type": "boolean"},
                    "confirmed": {
                        "type": "boolean",
                        "description": (
                            "Must be true for create/update/pause/resume/remove/run "
                            "after explicit user approval."
                        ),
                    },
                },
                "required": ["action"],
            },
        ),
    ]


def _build_delegation_parent():
    from hermes_cli.config import load_config, split_model_config_default
    from hermes_cli.fallback_config import get_fallback_chain
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent
    from tools.mcp_tool import discover_mcp_tools

    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    if isinstance(model_cfg, str):
        cfg_model = model_cfg
    else:
        raw_default = model_cfg.get("default") or model_cfg.get("model") or ""
        if isinstance(raw_default, dict):
            cfg_model, _ = split_model_config_default(raw_default)
        else:
            cfg_model = str(raw_default or "")
    effective_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip() or cfg_model

    runtime = resolve_runtime_provider(
        requested=None,
        target_model=effective_model or None,
        explicit_base_url=None,
    )
    discover_mcp_tools()

    parent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        requested_provider=runtime.get("requested_provider"),
        api_mode=runtime.get("api_mode"),
        model=effective_model,
        credential_pool=runtime.get("credential_pool"),
        fallback_model=get_fallback_chain(cfg) or None,
        enabled_toolsets=list(_SAFE_DELEGATION_TOOLSETS),
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        load_soul_identity=False,
    )
    parent._get_session_db_for_recall()
    return parent


class HermesCapabilities:
    def __init__(self, *, max_concurrent_delegations: int = 2):
        self._delegation_semaphore = asyncio.Semaphore(max_concurrent_delegations)

    def check(self) -> bool:
        try:
            from cron.scheduler import create_job_with_scheduler_registration
            from hermes_state import _default_db_path
            from run_agent import AIAgent

            _ = create_job_with_scheduler_registration
            _ = AIAgent
            return _default_db_path().parent.exists()
        except Exception:
            return False

    async def call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        if name == "session_search":
            return await self._session_search(arguments)
        if name == "delegate_task":
            return await self._delegate_task(arguments)
        if name == "cronjob":
            return await self._cronjob(arguments)
        return _error(f"Unknown Hermes capability: {name}")

    async def _session_search(self, arguments: dict[str, Any]) -> types.CallToolResult:
        if rejected := _reject_unknown(arguments, _SESSION_ALLOWED):
            return rejected
        payload = await asyncio.to_thread(
            session_search,
            query=arguments.get("query") or "",
            role_filter=arguments.get("role_filter"),
            limit=arguments.get("limit", 3),
            session_id=arguments.get("session_id"),
            around_message_id=arguments.get("around_message_id"),
            window=arguments.get("window", 5),
            sort=arguments.get("sort"),
            detail=arguments.get("detail", "adaptive"),
        )
        return _hermes_result(payload)


    async def _delegate_task(self, arguments: dict[str, Any]) -> types.CallToolResult:
        if rejected := _reject_unknown(arguments, _DELEGATE_ALLOWED):
            return rejected
        if arguments.get("confirmed") is not True:
            return _error("delegate_task requires confirmed=true after explicit user approval")

        goal = arguments.get("goal")
        tasks = arguments.get("tasks")
        if bool(goal) == bool(tasks):
            return _error("delegate_task requires exactly one of goal or tasks")
        if tasks is not None:
            if not isinstance(tasks, list) or not 1 <= len(tasks) <= 2:
                return _error("delegate_task supports 1 or 2 batch tasks")
            for index, task in enumerate(tasks):
                if not isinstance(task, dict):
                    return _error(f"delegate_task task {index} must be an object")
                unknown = sorted(set(task) - {"goal", "context"})
                if unknown:
                    return _error(
                        f"delegate_task task {index} has unsupported argument(s): "
                        f"{', '.join(unknown)}"
                    )
                if not str(task.get("goal") or "").strip():
                    return _error(f"delegate_task task {index} requires goal")

        async with self._delegation_semaphore:
            parent = await asyncio.to_thread(_build_delegation_parent)
            try:
                payload = await asyncio.to_thread(
                    delegate_task,
                    goal=goal,
                    context=arguments.get("context"),
                    tasks=tasks,
                    role="leaf",
                    background=False,
                    parent_agent=parent,
                )
            finally:
                await asyncio.to_thread(parent.close)
        return _hermes_result(payload)

    async def _cronjob(self, arguments: dict[str, Any]) -> types.CallToolResult:
        if rejected := _reject_unknown(arguments, _CRON_ALLOWED):
            return rejected
        action = str(arguments.get("action") or "").strip().lower()
        if action in _CRON_MUTATIONS and arguments.get("confirmed") is not True:
            return _error(
                f"cronjob action '{action}' requires confirmed=true after "
                "explicit user approval"
            )
        payload = await asyncio.to_thread(
            cronjob,
            action=action,
            job_id=arguments.get("job_id"),
            prompt=arguments.get("prompt"),
            schedule=arguments.get("schedule"),
            name=arguments.get("name"),
            repeat=arguments.get("repeat"),
            deliver=arguments.get("deliver") or "local",
            include_disabled=arguments.get("include_disabled", True),
            skill=arguments.get("skill"),
            skills=arguments.get("skills"),
            context_from=arguments.get("context_from"),
            continuity=arguments.get("continuity"),
            workdir=arguments.get("workdir"),
            attach_to_session=arguments.get("attach_to_session"),
        )
        return _hermes_result(payload)
