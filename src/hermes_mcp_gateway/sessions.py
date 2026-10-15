from __future__ import annotations

import hashlib
import re
import secrets
import time
from dataclasses import dataclass
from typing import Callable

from mcp import types

SESSION_TOKEN_TTL_SECONDS = 15 * 60
MAX_ACTIVE_CAPABILITIES = 256


@dataclass(slots=True)
class SessionCapability:
    task_id: str
    token_digest: str
    expires_at: float
    active_calls: int = 0
    revoked: bool = False


class SessionCapabilityStore:
    """In-memory, bounded bearer capabilities for sessionless MCP clients."""

    def __init__(
        self,
        *,
        ttl_seconds: int = SESSION_TOKEN_TTL_SECONDS,
        max_active: int = MAX_ACTIVE_CAPABILITIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds < 1 or max_active < 1:
            raise ValueError("session capability TTL and capacity must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_active = max_active
        self._clock = clock
        self._by_digest: dict[str, SessionCapability] = {}
        self._by_owner: dict[str, SessionCapability] = {}

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _revoke(self, record: SessionCapability) -> SessionCapability | None:
        if not record.revoked:
            record.revoked = True
            self._by_digest.pop(record.token_digest, None)
        if record.active_calls == 0:
            self._by_owner.pop(record.task_id, None)
            return record
        return None

    def expire_due(self) -> list[SessionCapability]:
        now = self._clock()
        expired = [
            record
            for record in tuple(self._by_owner.values())
            if not record.revoked and record.expires_at <= now
        ]
        cleanup = []
        for record in expired:
            if (candidate := self._revoke(record)) is not None:
                cleanup.append(candidate)
        return cleanup

    def issue(self) -> tuple[str, SessionCapability]:
        if len(self._by_owner) >= self.max_active:
            raise RuntimeError("Session capability capacity reached; close or wait for an existing session")
        token = secrets.token_urlsafe(32)
        record = SessionCapability(
            task_id=f"gateway-session-{secrets.token_hex(24)}",
            token_digest=self._digest(token),
            expires_at=self._clock() + self.ttl_seconds,
        )
        self._by_digest[record.token_digest] = record
        self._by_owner[record.task_id] = record
        return token, record

    def acquire(self, token: object) -> SessionCapability | None:
        # Only accept the exact alphabet/length we issue. Reject overlong or
        # surrogate-containing JSON strings before hashing/UTF-8 encoding.
        if not isinstance(token, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", token) is None:
            return None
        record = self._by_digest.get(self._digest(token))
        if record is None or record.revoked or record.expires_at <= self._clock():
            return None
        record.active_calls += 1
        return record

    def revoke(self, record: SessionCapability) -> SessionCapability | None:
        return self._revoke(record)

    def release(self, record: SessionCapability) -> SessionCapability | None:
        if record.active_calls <= 0:
            return None
        record.active_calls -= 1
        if record.revoked and record.active_calls == 0:
            self._by_owner.pop(record.task_id, None)
            return record
        return None

    def revoke_all(self) -> list[SessionCapability]:
        cleanup = []
        for record in tuple(self._by_owner.values()):
            if (candidate := self._revoke(record)) is not None:
                cleanup.append(candidate)
        self._by_digest.clear()
        return cleanup


def session_open_tool_schema() -> types.Tool:
    return types.Tool(
        name="session_open",
        description=(
            "Open a private, short-lived session for modern sessionless MCP clients. "
            "Call once, keep the returned session_token exactly, and pass it as the "
            "top-level session_token on every tool_call that should share browser tabs, "
            "terminal state, or process handles. The token expires after 15 minutes; "
            "open a new session when it expires, or close it with session_close."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )


def session_close_tool_schema() -> types.Tool:
    return types.Tool(
        name="session_close",
        description=(
            "Close the current private session. Call through tool_call with its "
            "top-level session_token. This revokes the token, terminates only processes "
            "owned by this session, and drops this session's local browser binding."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )
