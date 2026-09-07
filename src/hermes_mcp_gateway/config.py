from __future__ import annotations

from dataclasses import dataclass

CONTEXT_REQUIRED = frozenset(
    {
        "ctx_execute",
        "ctx_job_start",
        "ctx_job_status",
        "ctx_job_cancel",
        "ctx_execute_file",
        "ctx_index",
        "ctx_search",
        "ctx_fetch_and_index",
        "ctx_batch_execute",
        "ctx_doctor",
        "ctx_purge",
    }
)

WEB_TOOLS = frozenset({"web_search", "web_extract"})

HERMES_ALLOWLIST = frozenset(
    {
        "web_search",
        "web_extract",
        "vision_analyze",
        "skills_list",
        "skill_view",
    }
)

HERMES_REQUIRED = HERMES_ALLOWLIST - WEB_TOOLS

MEMORY_TOOLS = frozenset(
    {
        "memory_profile",
        "memory_search",
        "memory_context",
        "memory_reasoning",
        "memory_conclude",
    }
)

CAPABILITY_TOOLS = frozenset({"session_search", "delegate_task", "cronjob"})


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 3060
    context_url: str = "http://127.0.0.1:3050/mcp"
    context_health_url: str = "http://127.0.0.1:3050/healthz"
    context_ready_url: str = "http://127.0.0.1:3050/readyz"
    honcho_health_url: str = "http://127.0.0.1:8000/health"
    hermes_python: str = "/home/hermes/.hermes/venvs/hermes/bin/python"
    hermes_home: str = "/home/hermes/.hermes"
    hermes_cwd: str = "/home/hermes/.hermes"
    timeout_seconds: float = 30.0
    context_timeout_seconds: float = 50.0
    health_timeout_seconds: float = 3.0
    max_text_chars: int = 65_536
    max_startup_bytes: int = 64 * 1024
    memory_ai_peer: str = "chatgpt"
    memory_session: str = "chatgpt"
