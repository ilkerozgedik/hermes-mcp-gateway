"""Fail-closed guard for ad-hoc MCP command execution.

Only Gateway-exposed terminal calls are guarded. The job adapter directly invokes
Hermes terminal(background=True) under its own durable, credentialed owner.
"""
from __future__ import annotations

FOREGROUND_TERMINAL_MAX_SECONDS = 30
REQUIRES_JOB = (
    "LONG_RUNNING_OPERATION_REQUIRES_JOB: this Gateway only permits short "
    "foreground terminal commands (maximum 30 seconds). For builds, test suites, "
    "deploys or background work, use job_start and recover with job_status."
)


def guard_terminal(arguments: dict) -> tuple[dict | None, str | None]:
    if not isinstance(arguments, dict):
        return None, "terminal arguments must be an object"
    if not isinstance(arguments.get("command"), str) or not arguments["command"].strip():
        return None, "terminal command must be a nonempty string"
    if arguments.get("background") is not False and "background" in arguments:
        return None, REQUIRES_JOB
    if arguments.get("pty") is not False and "pty" in arguments:
        return None, REQUIRES_JOB
    if any(arguments.get(key) not in (None, False, 0, []) for key in (
        "notify", "heartbeat", "persist_on_release", "watch_patterns"
    )):
        return None, REQUIRES_JOB
    duration = arguments.get("timeout", FOREGROUND_TERMINAL_MAX_SECONDS)
    if type(duration) is not int or not 1 <= duration <= FOREGROUND_TERMINAL_MAX_SECONDS:
        return None, REQUIRES_JOB
    # A caller cannot select a longer terminal deadline by omitting timeout.
    return {**arguments, "timeout": duration}, None
