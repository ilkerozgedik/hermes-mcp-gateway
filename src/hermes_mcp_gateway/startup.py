from __future__ import annotations

import hashlib
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
    Path("/home/hermes/.hermes/skills/ponytail/SKILL.md"),
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
    ) -> None:
        self.paths = paths
        self.max_total_bytes = max_total_bytes

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
        try:
            total = 0
            for path in self.paths:
                info = self._validated_stat(path)
                total += info.st_size
                if total > self.max_total_bytes:
                    return False
            return True
        except RuntimeError:
            return False

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
            raw = path.read_bytes()
            if len(raw) != info.st_size:
                raise RuntimeError(f"startup file changed during read: {path}")
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
        return {"files": files}
