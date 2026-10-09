from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from mcp import types
from tools.tool_search_catalog import build_catalog as hermes_build_catalog
from tools.tool_search_catalog import search_catalog

# Keep session bootstrap cheap; every other formerly public tool is discoverable.
ALWAYS_VISIBLE = frozenset({"memory_context", "startup_context", "skills_list"})
BRIDGE_NAMES = frozenset({"tool_search", "tool_describe", "tool_call"})
PUBLIC_NAMES = ALWAYS_VISIBLE | BRIDGE_NAMES
MAX_SEARCH_QUERIES = 7
MAX_DESCRIBE_NAMES = 10


def bridge_schemas() -> list[types.Tool]:
    return [
        types.Tool(
            name="tool_search",
            description=(
                "Search the Gateway's permitted tools by keywords, e.g. 'browser click', "
                "'context execute', 'github issue claim'. Additional permitted tools "
                "are available on demand. Follow with tool_describe for their input schemas, "
                "then tool_call. Only catalogued tools can be invoked."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_SEARCH_QUERIES,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "default": 5,
                    },
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
            annotations=types.ToolAnnotations(read_only_hint=True),
        ),
        types.Tool(
            name="tool_describe",
            description="Return full input schemas for permitted tool names. Batch up to ten names.",
            input_schema={
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_DESCRIBE_NAMES,
                    },
                },
                "required": ["names"],
                "additionalProperties": False,
            },
            annotations=types.ToolAnnotations(read_only_hint=True),
        ),
        types.Tool(
            name="tool_call",
            description=(
                "Run exactly one tool discovered via tool_search/tool_describe, with the "
                "arguments specified by its schema. Existing Gateway permissions, approvals, "
                "session isolation, output filtering and no-retry policy still apply."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "calls": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "arguments": {"type": "object"},
                            },
                            "required": ["name", "arguments"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["calls"],
                "additionalProperties": False,
            },
        ),
    ]


def public_schemas(catalog: dict[str, Any]) -> list[types.Tool]:
    missing = ALWAYS_VISIBLE - set(catalog)
    if missing:
        raise RuntimeError(
            f"Gateway public tools missing: {', '.join(sorted(missing))}"
        )
    return bridge_schemas() + [
        entry.tool for name, entry in catalog.items() if name in ALWAYS_VISIBLE
    ]


class ToolDiscovery:
    """Hermes BM25 retrieval over an immutable, Gateway-allowlisted MCP catalog."""

    def __init__(self, catalog: dict[str, Any]):
        self.catalog = {
            name: entry for name, entry in catalog.items() if name not in ALWAYS_VISIBLE
        }
        # Search is indexed once at process start, matching Gateway's frozen MCP schema contract.
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{entry.source} {entry.tool.description or ''}",
                    "parameters": entry.tool.input_schema,
                },
            }
            for name, entry in self.catalog.items()
        ]
        self.index = hermes_build_catalog(definitions)

    @staticmethod
    def _parse(args: Any, schema: dict[str, Any]) -> str | None:
        try:
            Draft202012Validator(schema).validate(args)
        except ValidationError as exc:
            return str(exc).splitlines()[0][:500]
        return None

    @staticmethod
    def _list_arg(
        args: dict[str, Any], key: str, limit: int
    ) -> tuple[list[str], str | None]:
        items = args.get(key)
        if not isinstance(items, list) or not 1 <= len(items) <= limit:
            return [], f"{key} must contain 1-{limit} strings"
        if any(not isinstance(x, str) or not x.strip() for x in items):
            return [], f"{key} must contain nonempty strings"
        return [x.strip() for x in items], None

    def search(self, args: dict[str, Any]) -> dict[str, Any]:
        queries, err = self._list_arg(args, "queries", MAX_SEARCH_QUERIES)
        if err:
            raise ValueError(err)
        raw_limit = args.get("limit", 5)
        if type(raw_limit) is not int or not 1 <= raw_limit <= 25:
            raise ValueError("limit must be an integer between 1 and 25")
        groups = []
        matches = {}
        for query in queries:
            hits = search_catalog(self.index, query, limit=raw_limit)
            groups.append({"query": query, "matches": [hit.name for hit in hits]})
            for hit in hits:
                entry = self.catalog[hit.name]
                matches[hit.name] = {
                    "source": entry.source,
                    "description": (entry.tool.description or "")[:500],
                    "required": entry.tool.input_schema.get("required", []),
                }
        return {
            "total_available": len(self.catalog),
            "results": groups,
            "tools": matches,
        }

    def describe(self, args: dict[str, Any]) -> dict[str, Any]:
        names, err = self._list_arg(args, "names", MAX_DESCRIBE_NAMES)
        if err:
            raise ValueError(err)
        return {
            "tools": {
                name: {
                    "description": self.catalog[name].tool.description or "",
                    "parameters": self.catalog[name].tool.input_schema,
                    "source": self.catalog[name].source,
                }
                for name in dict.fromkeys(names)
                if name in self.catalog
            },
            "not_found": [
                name for name in dict.fromkeys(names) if name not in self.catalog
            ],
        }

    def validate_call(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        schemas = bridge_schemas()
        err = self._parse(args, schemas[2].input_schema)
        if err:
            raise ValueError(f"invalid tool_call: {err}")
        call = args["calls"][0]
        name = call["name"]
        if name not in self.catalog:
            raise ValueError(f"unknown or disallowed deferred tool: {name}")
        params = call["arguments"]
        err = self._parse(params, self.catalog[name].tool.input_schema)
        if err:
            raise ValueError(f"invalid arguments for {name}: {err}")
        return name, params
