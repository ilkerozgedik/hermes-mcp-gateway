from __future__ import annotations

from dataclasses import dataclass

WEB_TOOLS = frozenset({"web_search", "web_extract"})

SAMCHON_GRAPH_TOOL = "inspect_code_graph"

DIRECT_HERMES_TOOLSETS = ("file", "terminal", "video")
DIRECT_HERMES_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "patch",
        "search_files",
        "terminal",
        "process_manage",
        "video_analyze",
    }
)

BROWSER_TOOLS = frozenset(
    {
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_snapshot",
        "browser_scroll",
        "browser_back",
        "browser_get_images",
        "browser_console",
        "browser_vision",
    }
)

IMAGE_TOOLS = frozenset({"image_generate"})

# These tools retain resources across calls; foreground terminals retain cwd/env.
STATEFUL_HERMES_TOOLS = BROWSER_TOOLS | frozenset({"terminal", "process_manage"})

HERMES_ALLOWLIST = frozenset(
    {
        "web_search",
        "web_extract",
        "vision_analyze",
        "skills_list",
        "skill_view",
        "image_generate",
        "execute_code",
    }
) | BROWSER_TOOLS | DIRECT_HERMES_TOOLS

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

@dataclass(frozen=True, slots=True)
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 3060
    honcho_health_url: str = "http://127.0.0.1:8000/health"
    hermes_python: str = "/home/hermes/.hermes/venvs/hermes/bin/python"
    hermes_home: str = "/home/hermes/.hermes"
    hermes_cwd: str = "/home/hermes/.hermes"
    samchon_graph_command: str = "/home/hermes/.local/bin/samchon-graph"
    samchon_graph_schema_cwd: str = "/srv/agents/src/hermes-mcp-gateway"
    samchon_graph_allowed_roots: tuple[str, ...] = (
        "/home/hermes/work",
        "/home/hermes/worktrees",
        "/srv/agents/src",
    )
    samchon_graph_max_sessions: int = 4
    samchon_graph_timeout_seconds: float = 120.0
    timeout_seconds: float = 30.0
    execution_timeout_seconds: float = 300.0
    execution_max_concurrent: int = 2
    execution_max_code_bytes: int = 65_536
    execution_max_worker_bytes: int = 1_048_576
    health_timeout_seconds: float = 3.0
    max_text_chars: int = 65_536
    max_startup_bytes: int = 64 * 1024
    memory_ai_peer: str = "chatgpt"
    memory_session: str = "chatgpt"
