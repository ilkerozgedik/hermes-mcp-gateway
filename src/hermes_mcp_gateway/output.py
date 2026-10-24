"""Shared output redaction and response budgets for HTTP and stdio MCP."""
from __future__ import annotations

import json
from typing import Any

from agent.redact import redact_sensitive_text
from mcp import types


def _sanitize_structure(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, dict):
        return {
            redact_sensitive_text(str(key), force=True): _sanitize_structure(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_structure(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_sensitive_text(str(value), force=True)



def normalize_result(
    result: types.CallToolResult, *, limit: int, preserve_images: bool = False
) -> types.CallToolResult:
    content: list[Any] = []
    for item in result.content:
        if isinstance(item, types.TextContent):
            text = redact_sensitive_text(item.text, force=True)
            if len(text) > limit:
                suffix = "\n...[truncated by hermes-mcp-gateway]"
                text = text[: max(0, limit - len(suffix))] + suffix
            content.append(types.TextContent(text=text))
        elif preserve_images and isinstance(item, types.ImageContent):
            content.append(item)
        else:
            marker = (
                f"[{type(item).__name__} omitted by hermes-mcp-gateway output policy]"
            )
            content.append(types.TextContent(text=marker[:limit]))
    structured = result.structured_content
    if structured is not None:
        structured = _sanitize_structure(structured)
        rendered = json.dumps(structured, ensure_ascii=False, default=str)
        if len(rendered) > limit:
            structured = {
                "truncated": True,
                "preview": rendered[: max(0, limit - 64)],
            }
    return types.CallToolResult(
        content=content, structured_content=structured, is_error=result.is_error
    )



def error_result(
    message: str, *, limit: int, category: str | None = None
) -> types.CallToolResult:
    text = redact_sensitive_text(str(message), force=True)
    if len(text) > limit:
        text = text[:limit]
    metadata = {"_gateway_error_category": category} if category else None
    return types.CallToolResult(
        content=[types.TextContent(text=text)],
        structured_content=metadata,
        is_error=True,
    )



def json_result(payload: Any, *, limit: int) -> types.CallToolResult:
    """Redact before serialization and keep truncated JSON syntactically valid."""
    rendered = json.dumps(_sanitize_structure(payload), ensure_ascii=False, default=str)
    if len(rendered) > limit:
        def envelope(size: int) -> str:
            return json.dumps({"truncated": True, "preview": rendered[:size]}, ensure_ascii=False)

        low, high = 0, min(len(rendered), limit)
        if len(envelope(0)) > limit:
            raise ValueError("JSON output limit is too small for a truncation receipt")
        while low < high:
            middle = (low + high + 1) // 2
            if len(envelope(middle)) <= limit:
                low = middle
            else:
                high = middle - 1
        rendered = envelope(low)
    return types.CallToolResult(content=[types.TextContent(text=rendered)])
