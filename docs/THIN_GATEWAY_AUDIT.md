# Hermes → ChatGPT MCP boundary audit

Checked against pinned Hermes and Gateway at Gateway base `cb9e53e` (2026-10-17).

## Architectural rule

Hermes owns tool behavior. Gateway owns MCP framing, tool discovery, explicit
default-deny access, output redaction/budgets, request/session isolation,
upstream compatibility, and deployment/readiness checks. Do not fork Hermes
feature implementations into Gateway. Prefer removing unnecessary adapters
only when parity tests prove unchanged behavior.

## Native tool inventory

Live `model_tools.get_tool_definitions(quiet_mode=True, skip_tool_search_assembly=True)`
reported **49** named Hermes tools. Gateway's `HERMES_ALLOWLIST` contains
**24** names, all present in the pinned Hermes definitions. All seven
`DIRECT_HERMES_TOOLS` are also discoverable. The remaining 25 Hermes
tools are **deliberately not exposed through the Hermes allowlist** (e.g.
GUI tools, computer use, credential vault access, todo, Kanban-related
capabilities, TTS). Some, notably `delegate_task` and
`session_search`, are provided separately through guarded adapters.

The public MCP `tools/list` has six stable discovery/context tools;
other tools are reached through `tool_search` → `tool_describe` →
`tool_call`. A Hermes version update needs a fresh inventory review,
not automatic allowlist widening.

| Boundary | Implementation | Keep / simplify |
|---|---|---|
| File operations | Hermes native `read_file`, `write_file`, `patch`, `search_files` | Keep native behavior. Only `read_file` requires the ChatGPT structured-output compatibility envelope and page-safe Gateway output cap. |
| Terminal/process | Native Hermes handlers with Gateway session capabilities | Keep ownership/isolation; never replay mutations on failure. |
| Browser | Hermes Camofox calls | Keep authenticated owner and stale-tab recovery; no second browser system. |
| Web/media/skills | Native Hermes providers/schemas | Forward; no replacement web/media implementation. |
| `execute_code` | Native Hermes execution in per-request worker | Keep execution isolation and resource limits. No persistent Python state. |
| Cron | Hermes `cronjob` called by guarded adapter | Keep `list` read-only and direct; mutations restricted; no second job database. |
| Session history | Hermes `session_search` | Forward and forbid cross-profile overrides. |
| Delegation | Hermes `delegate_task` via leaf adapter | Keep leaf, concurrency and feature limits; no independent agent engine. |
| Memory | Existing Hermes Honcho plugin, distinct ChatGPT assistant peer | Preserve current user/workspace identity; don't duplicate Honcho. |
| Samchon Graph | One project-scoped Samchon MCP tool | Keep as separate integration, not Hermes feature reimplementation. |
| LSP | Hermes LSP adapter | Preserve validated diagnostic freshness fix; no new language server stack. |

## Verified parity and expected differences

- Current Hermes cron store returned exactly two enabled İlkerce Müzik
  jobs; Gateway `cronjob(action=list, include_disabled=true)` returned
  the same two. Gateway must not fabricate missing tasks from ChatGPT
  Tasks or other Hermes profiles.
- Honcho memory schemas are derived from Hermes' installed plugin;
  Gateway removes profile-writing support and requires durable categories
  for conclusion writes. This is deliberate, not full write parity.
- Native `read_file` returns a JSON string with `content`; the ChatGPT
  client previously surfaced it as `{}`. Gateway preserves that legacy
  text result while exposing `structuredContent.file_text`. For a read
  exceeding the Gateway output budget, page at Hermes-numbered line
  boundaries and provide `next_offset`. Do not discard source lines
  silently. Reject an oversized individual line explicitly.
- A tool may be advertised but intentionally blocked by policy; strict
  `tools/list`, per-tool validation, and owner isolation are security
  requirements, not a missing feature.

## Acceptance gates

1. Native-vs-Gateway inventory verified for pinned Hermes; default-deny
   and public six-tool surface unchanged.
2. `read_file`: basic, continuation, EOF, missing file, large page,
   exceptional single-line overflow, secret masking, and legacy text
   compatibility.
3. Cron `list`, Honcho read tools, and session search return existing
   backend results without modifications to their stores.
4. LSP repeated diagnostics and symbol lookup, Samchon Graph lookup,
   browser/session continuity, native execution cancellation, and
   denial of hidden/foreign-session actions remain green.
5. Unit suite, Ruff, Streamable HTTP acceptance, service readiness,
   pinned bootstrap deploy, and direct ChatGPT smoke readbacks.

Avoid refactoring healthy cron/memory/LSP/browser code absent a
specific failing parity test. Do not broaden execution permissions to
fix transport limits.
