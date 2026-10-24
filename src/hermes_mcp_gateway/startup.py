from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from agent.redact import redact_sensitive_text
from mcp import types

STARTUP_PATHS = (
    Path("/home/hermes/.hermes/.hermes.md"),
    Path("/home/hermes/.hermes/SOUL.md"),
    Path("/home/hermes/.hermes/memories/MEMORY.md"),
    Path("/home/hermes/.hermes/memories/USER.md"),
)


def startup_tool_schema() -> types.Tool:
    return types.Tool(
        name="startup_context",
        description=(
            "Read the fixed Hermes startup instruction files. This tool accepts no path "
            "and cannot read arbitrary filesystem content."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(read_only_hint=True),
    )


class StartupContext:
    def __init__(
        self,
        *,
        paths: tuple[Path, ...] = STARTUP_PATHS,
        max_total_bytes: int = 64 * 1024,
        max_output_chars: int = 65_536,
    ) -> None:
        self.paths = paths
        self.max_total_bytes = max_total_bytes
        self.max_output_chars = max_output_chars

    def _validated_stat(self, path: Path):
        if path.is_symlink():
            raise RuntimeError(f"startup file must not be a symlink: {path}")
        try:
            info = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RuntimeError(f"required startup file is missing: {path}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"startup path is not a regular file: {path}")
        return info

    def check(self) -> bool:
        # Read the same bounded, no-follow payload the tool would return. Metadata
        # alone cannot prove readability, UTF-8 validity or serialized output size.
        try:
            self.read()
            return True
        except (RuntimeError, OSError):
            return False

    def _read_file(self, path: Path, expected) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            actual = os.fstat(stream.fileno())
            if not stat.S_ISREG(actual.st_mode):
                raise RuntimeError(f"startup path is not a regular file: {path}")
            if (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns) != (
                expected.st_dev, expected.st_ino, expected.st_size, expected.st_mtime_ns
            ):
                raise RuntimeError(f"startup file changed before read: {path}")
            raw = stream.read(self.max_total_bytes + 1)
            after = os.fstat(stream.fileno())
            if len(raw) != expected.st_size or after.st_mtime_ns != actual.st_mtime_ns:
                raise RuntimeError(f"startup file changed during read: {path}")
        return raw

    def read(self) -> dict[str, Any]:
        stats = []
        total = 0
        for path in self.paths:
            info = self._validated_stat(path)
            total += info.st_size
            if total > self.max_total_bytes:
                raise RuntimeError("startup context exceeds configured size limit")
            stats.append((path, info))

        files = []
        for path, info in stats:
            raw = self._read_file(path, info)
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"startup file is not valid UTF-8: {path}") from exc
            files.append(
                {
                    "path": str(path),
                    "content": redact_sensitive_text(content, force=True),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "mtime_ns": info.st_mtime_ns,
                }
            )
        payload = {"files": files}
        # The envelope and JSON escaping consume the same budget as content.
        # Fail explicitly instead of returning success with a cut JSON document.
        if len(json.dumps(payload, ensure_ascii=False)) > self.max_output_chars:
            raise RuntimeError("startup context serialized output exceeds configured size limit")
        return payload
