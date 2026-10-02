from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agent.lsp.manager import LSPService
from agent.lsp.servers import language_id_for
from mcp import types

LSP_TOOLS = frozenset(
    {
        "lsp_diagnostics",
        "lsp_hover",
        "lsp_definition",
        "lsp_references",
        "lsp_document_symbols",
        "lsp_status",
    }
)


def _file_schema(*, position: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "file": {
            "type": "string",
            "description": "Absolute path to an existing source file under an allowed project root.",
        }
    }
    required = ["file"]
    if position:
        properties.update(
            {
                "line": {"type": "integer", "minimum": 1, "description": "1-based line number."},
                "column": {"type": "integer", "minimum": 1, "description": "1-based column number."},
            }
        )
        required.extend(("line", "column"))
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def lsp_tool_schemas() -> list[types.Tool]:
    return [
        types.Tool(
            name="lsp_diagnostics",
            description="Return current read-only LSP diagnostics for one source file.",
            input_schema=_file_schema(),
        ),
        types.Tool(
            name="lsp_hover",
            description="Return read-only LSP hover/type/documentation information at a source position.",
            input_schema=_file_schema(position=True),
        ),
        types.Tool(
            name="lsp_definition",
            description="Return read-only LSP definition locations for a source position.",
            input_schema=_file_schema(position=True),
        ),
        types.Tool(
            name="lsp_references",
            description="Return read-only LSP references for a source position, including the declaration.",
            input_schema=_file_schema(position=True),
        ),
        types.Tool(
            name="lsp_document_symbols",
            description="Return the LSP document-symbol tree for one source file.",
            input_schema=_file_schema(),
        ),
        types.Tool(
            name="lsp_status",
            description="Return Hermes LSP service status. Does not install or modify language servers.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
    ]


class LSPAdapter:
    def __init__(
        self,
        *,
        allowed_roots: tuple[str, ...],
        service: LSPService | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._roots = tuple(Path(root).resolve(strict=True) for root in allowed_roots)
        self._timeout = timeout_seconds
        if service is None:
            service = LSPService.create_from_config()
            if service is None:
                service = LSPService(
                    enabled=True,
                    wait_mode="document",
                    wait_timeout=min(5.0, max(0.5, timeout_seconds / 4)),
                    install_strategy="manual",
                    idle_timeout=600.0,
                    broken_retry_seconds=30.0,
                )
        self._service = service
        # Read-only gateway calls may probe installed binaries but must never install one.
        self._service._install_strategy = "manual"

        # Preserve Hermes config, but provide its managed TypeScript SDK as a fallback
        # for projects that do not vendor their own TypeScript package.
        tsserver = (
            Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
            / "lsp/node_modules/typescript/lib/tsserver.js"
        )
        if tsserver.is_file():
            ts_init = self._service._init_overrides.setdefault("typescript", {})
            if isinstance(ts_init, dict):
                tsserver_init = ts_init.get("tsserver")
                if tsserver_init is None:
                    ts_init["tsserver"] = {"path": str(tsserver)}
                elif isinstance(tsserver_init, dict):
                    tsserver_init.setdefault("path", str(tsserver))

    def healthy(self) -> bool:
        return bool(self._service and self._service.is_active())

    def close(self) -> None:
        if self._service is not None:
            self._service.shutdown()

    def _resolve_file(self, value: Any) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("file must be a non-empty absolute path")
        raw = Path(value)
        if not raw.is_absolute():
            raise ValueError("file must be an absolute path")
        path = raw.resolve(strict=True)
        if not path.is_file():
            raise ValueError("file must point to an existing regular file")
        if not any(path.is_relative_to(root) for root in self._roots):
            raise PermissionError("file is outside the allowed LSP project roots")
        return path

    @staticmethod
    def _position(arguments: dict[str, Any]) -> dict[str, int]:
        line = arguments.get("line")
        column = arguments.get("column")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            raise ValueError("line must be an integer >= 1")
        if not isinstance(column, int) or isinstance(column, bool) or column < 1:
            raise ValueError("column must be an integer >= 1")
        return {"line": line - 1, "character": column - 1}

    def _display_path(self, path: Path) -> str:
        for root in self._roots:
            if path.is_relative_to(root):
                return str(path.relative_to(root))
        return str(path)

    def _normalize_lsp_value(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._normalize_lsp_value(item) for item in value]
        if not isinstance(value, dict):
            return value

        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"uri", "targetUri"} and isinstance(item, str):
                parsed = urlparse(item)
                if parsed.scheme == "file":
                    try:
                        out["file" if key == "uri" else "targetFile"] = self._display_path(Path(unquote(parsed.path)).resolve())
                    except (OSError, RuntimeError, ValueError):
                        out[key] = item
                else:
                    out[key] = item
                continue
            if key in {"start", "end"} and isinstance(item, dict):
                line = item.get("line")
                character = item.get("character")
                if isinstance(line, int) and isinstance(character, int):
                    out[key] = {"line": line + 1, "column": character + 1}
                    continue
            out[key] = self._normalize_lsp_value(item)
        return out

    async def _request_async(
        self, file_path: Path, method: str, params: dict[str, Any]
    ) -> Any:
        client = await self._service._get_or_spawn(str(file_path))
        if client is None:
            raise RuntimeError(
                "no installed/available Hermes LSP server for this file; automatic installation is disabled"
            )
        server = self._service._server_for(str(file_path))
        await client.open_file(
            str(file_path),
            language_id=language_id_for(str(file_path), server),
        )
        return await client._send_request_with_retry(
            method, params, timeout=self._timeout
        )


    async def _diagnostics_async(self, file_path: Path) -> list[dict[str, Any]] | None:
        client = await self._service._get_or_spawn(str(file_path))
        if client is None:
            raise RuntimeError(
                "no installed/available Hermes LSP server for this file; automatic installation is disabled"
            )
        server = self._service._server_for(str(file_path))
        version = await client.open_file(
            str(file_path),
            language_id=language_id_for(str(file_path), server),
        )

        # Hermes marks the first TypeScript diagnostics push as a baseline seed.
        # For a read-only query we need a current verdict, so wait for that seed
        # and send one no-op didChange to obtain the next, fresh diagnostics set.
        if server is not None and server.seed_first_push:
            loop = asyncio.get_running_loop()
            seed_deadline = loop.time() + min(5.0, max(0.5, self._timeout / 3))
            abs_path = str(file_path)
            while loop.time() < seed_deadline:
                doc = client._docs.get(abs_path)
                if doc is not None and doc.seed_seen:
                    version = await client.open_file(
                        abs_path, language_id=language_id_for(abs_path, server)
                    )
                    break
                await asyncio.sleep(0.05)

        await client.save_file(str(file_path))
        fresh = await client.wait_for_diagnostics(
            str(file_path), version, mode="document", timeout=self._timeout
        )
        if not fresh:
            return None
        return list(client.diagnostics_for(str(file_path), fresh_only=True))

    def _request(
        self, file_path: Path, method: str, arguments: dict[str, Any], *, position: bool
    ) -> Any:
        params: dict[str, Any] = {"textDocument": {"uri": file_path.as_uri()}}
        if position:
            params["position"] = self._position(arguments)
        if method == "textDocument/references":
            params["context"] = {"includeDeclaration": True}
        value = self._service._loop.run(
            self._request_async(file_path, method, params), timeout=self._timeout
        )
        return self._normalize_lsp_value(value)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in LSP_TOOLS:
            raise ValueError(f"unsupported LSP tool: {name}")
        if name == "lsp_status":
            if arguments:
                raise ValueError("lsp_status does not accept arguments")
            return {"install_strategy": "manual", "status": self._service.get_status()}

        file_path = self._resolve_file(arguments.get("file"))
        if name == "lsp_diagnostics":
            diagnostics = self._service._loop.run(
                self._diagnostics_async(file_path), timeout=self._timeout + 6.0
            )
            if diagnostics is None:
                raise RuntimeError(
                    "LSP server did not return a fresh diagnostics verdict within the timeout"
                )
            return {
                "file": self._display_path(file_path),
                "diagnostics": self._normalize_lsp_value(diagnostics),
            }

        methods = {
            "lsp_hover": ("textDocument/hover", True),
            "lsp_definition": ("textDocument/definition", True),
            "lsp_references": ("textDocument/references", True),
            "lsp_document_symbols": ("textDocument/documentSymbol", False),
        }
        method, needs_position = methods[name]
        return {
            "file": self._display_path(file_path),
            "result": self._request(
                file_path, method, arguments, position=needs_position
            ),
        }
