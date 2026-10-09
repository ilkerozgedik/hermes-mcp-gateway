"""Stateless HTTP MCP front-end for the persistent coordinator."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from starlette.responses import JSONResponse
from starlette.routing import Route

from .core import Coordinator

TOOLS: dict[str, dict[str, Any]] = {
    "tasks_list": {"repo": "str"},
    "task_claim": {"repo": "str", "issue": "int", "agent": "str"},
    "task_status": {"repo": "str", "issue": "int"},
    "task_heartbeat": {"repo": "str", "issue": "int", "claim_id": "str"},
    "task_finish": {
        "repo": "str",
        "issue": "int",
        "claim_id": "str",
        "pr_number": "int",
    },
    "task_release": {"repo": "str", "issue": "int", "claim_id": "str"},
}
DESCRIPTIONS = {
    "tasks_list": "List agent:ready GitHub issues and owned tasks for an allowed repository.",
    "task_claim": "Claim one ready issue atomically; create an isolated Git worktree. Returns a private owner token.",
    "task_status": "Read task ownership, status, and worktree without revealing owner tokens.",
    "task_heartbeat": "Renew an active task lease using its private owner token.",
    "task_finish": "Mark work ready for review only after validating its open PR against main.",
    "task_release": "Mark owned work blocked for manual recovery; never delete worktrees automatically.",
}


def tool_schemas() -> list[types.Tool]:
    result = []
    for name, props in TOOLS.items():
        properties = {
            key: {"type": "integer" if kind == "int" else "string"}
            for key, kind in props.items()
        }
        for key in ("claim_id",):
            if key in properties:
                properties[key]["description"] = (
                    "Private capability from task_claim; do not log or publish."
                )
        result.append(
            types.Tool(
                name=name,
                description=DESCRIPTIONS[name],
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": list(props),
                    "additionalProperties": False,
                },
            )
        )
    return result


def default_coordinator() -> Coordinator:
    return Coordinator(
        db_path=os.getenv(
            "AGENT_COORDINATOR_DB", "/srv/agents/runtime/multi-agent/coordinator.sqlite"
        ),
        work_root=os.getenv("AGENT_COORDINATOR_WORK_ROOT", "/home/hermes/worktrees"),
        repos_root=os.getenv("AGENT_COORDINATOR_REPOS_ROOT", "/home/hermes/work"),
        owner=os.getenv("AGENT_COORDINATOR_OWNER", "ilkerozgedik"),
        repositories=tuple(
            x.strip()
            for x in os.getenv("AGENT_COORDINATOR_REPOS", "peacify").split(",")
            if x.strip()
        ),
    )


def create_app(coordinator: Coordinator | None = None):
    coordinator = coordinator or default_coordinator()

    async def list_tools(_ctx, _params):
        return types.ListToolsResult(tools=tool_schemas())

    async def call_tool(_ctx, params):
        try:
            name = params.name
            args = params.arguments or {}
            if name not in TOOLS:
                raise ValueError("unknown tool")
            if set(args) != set(TOOLS[name]):
                raise ValueError("incorrect tool arguments")
            for key, kind in TOOLS[name].items():
                expected = int if kind == "int" else str
                if type(args[key]) is not expected:
                    raise ValueError(f"{key} has incorrect type")
            methods = {
                "tasks_list": coordinator.tasks_list,
                "task_claim": coordinator.claim,
                "task_status": coordinator.status,
                "task_heartbeat": coordinator.heartbeat,
                "task_finish": coordinator.finish,
                "task_release": coordinator.release,
            }
            payload = await asyncio.to_thread(methods[name], **args)
            return types.CallToolResult(
                content=[types.TextContent(text=json.dumps(payload))],
                structured_content=payload,
            )
        except (
            ValueError,
            PermissionError,
            RuntimeError,
            TimeoutError,
            OSError,
        ) as exc:
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        text=json.dumps({"success": False, "error": str(exc)[:500]})
                    )
                ],
                is_error=True,
            )

    app = Server(
        "multi-agent-coordinator",
        version="0.1.0",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        instructions="Durable GitHub Issue claims; no autonomous merging or deployment.",
    )

    async def healthz(_request):
        return JSONResponse({"status": "ok", "database": str(coordinator.db_path)})

    return app.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        host="127.0.0.1",
        custom_starlette_routes=[Route("/healthz", healthz)],
    )


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=3061, access_log=False)


if __name__ == "__main__":
    main()
