# Hermes MCP Gateway

Single localhost-only MCP endpoint that aggregates the existing Context Mode MCP, a strict allowlist from Hermes native tools, a read-only semantic subset from Serena, three guarded Hermes capabilities, and Hermes/Honcho memory without patching any upstream source.

## Endpoint

- MCP: `http://127.0.0.1:3060/mcp`
- Process health: `http://127.0.0.1:3060/healthz`
- Final-cutover readiness: `http://127.0.0.1:3060/readyz`

`healthz` stays available when a running upstream later degrades. `readyz` is `200` only when Context Mode, the curated Hermes tools including web, the guarded Hermes capability runtime, and Honcho are all available. Tool discovery is frozen for the process lifetime; provider/config changes require a gateway restart, keeping MCP schemas stable within a running process.

## Tool policy

Context Mode forwards discovered `ctx_*` tools and remains the primary compact execution/indexing backend. The gateway also exposes a deliberately small native Hermes file/terminal surface for direct host work.

Hermes is default-deny and may expose only:

- `web_search`, `web_extract`
- `vision_analyze`
- `skills_list`, `skill_view`
- native filesystem: `read_file`, `write_file`, `patch`, `search_files`
- native terminal/process: `terminal`, `process`
- media: `image_generate`, `video_analyze`
- Camofox-compatible native browser tools: `browser_navigate`, `browser_click`, `browser_type`, `browser_press`, `browser_snapshot`, `browser_scroll`, `browser_back`, `browser_get_images`, `browser_console`, `browser_vision`

The gateway does not expose `browser_exec`, `browser_cdp`, `computer_use`, raw Hermes memory/todo state, TTS, or Kanban tools. `session_search`, `delegate_task`, and `cronjob` are exposed only through gateway-owned guarded adapters, not through the stateless Hermes allowlist.

Serena is a separate stdio upstream. Its public surface is intentionally limited to `activate_project`, `find_declaration`, `find_implementations`, `find_referencing_symbols`, `find_symbol`, `get_diagnostics_for_file`, and `get_symbols_overview`. Serena shell/file-edit/memory tools remain default-deny.

Guarded Hermes capabilities are:

- `session_search` — read-only search over the local Hermes session database. Cross-profile override is intentionally unavailable; results are historical conversation context, not proof of current external state.
- `delegate_task` — synchronous leaf delegation only, one or two children maximum, and `confirmed=true` after explicit user approval because it spends model inference. Children inherit only Hermes `web`, `vision`, `skills`, and the `mcp-context-mode` toolset. Context Mode MCP tools are reached through Hermes' scoped `tool_search`/`tool_describe`/`tool_call` bridge; native Hermes terminal/file/code tools and recursive delegation remain out of scope.
- `cronjob` — read-only `list`, plus guarded `create`, `update`, `pause`, `resume`, `remove`, and `run`. Mutations require `confirmed=true`. Model/provider/base-URL overrides and script/no-agent/monitor execution fields are not exposed; delivery defaults to `local`.

With the currently pinned Context Mode surface, the public MCP contract is 50 tools: 11 Context Mode + 23 curated Hermes tools (including 10 Camofox browser tools, native file/terminal/process, image generation, and video analysis) + 7 Serena semantic-code tools + 3 guarded Hermes capabilities + 5 memory + `startup_context`.

Gateway-owned startup tool:

- `startup_context` — reads only the five fixed Hermes startup files (`.hermes.md`, `SOUL.md`, `MEMORY.md`, `USER.md`, and canonical Ponytail `SKILL.md`). It accepts no path and is not a generic filesystem surface.

Memory tools are:

- `memory_profile` (read-only)
- `memory_search`
- `memory_context`
- `memory_reasoning`
- `memory_conclude`

Memory identity is resolved through Hermes Honcho configuration. The existing `workspace=hermes` and user peer are preserved; only this adapter's assistant peer is `chatgpt`. Conclusion writes require a durable kind (`preference`, `decision`, `architecture`, `project_state`) and reject secret-like or explicitly temporary content. List/delete operations retain Honcho semantics.

## Browser

The unified ChatGPT gateway exposes the 10 Hermes native browser tools that are available in the pinned Camofox mode. The gateway subprocess uses the same `HERMES_HOME=/home/hermes/.hermes` and inherits `CAMOFOX_URL=http://127.0.0.1:9377`, so ChatGPT and Hermes Agent share the same Hermes-managed persistent Camofox `userId`/profile. Tool calls still use Hermes/Camofox task-session semantics rather than a second browser profile. `browser_exec` and `browser_cdp` remain absent because the pinned Camofox backend does not provide those Hermes backends.

## Web prerequisite

Hermes natively supports `web_search` and `web_extract` through its managed Nous Tool Gateway using the Firecrawl provider. The gateway does not fabricate these schemas and does not implement a second web stack.

The narrow upstream-supported activation path is:

1. Authenticate the existing Hermes profile with `hermes auth add nous --type oauth --no-browser`. This adds Nous OAuth state without requiring the default inference provider to switch to Nous.
2. Select the managed web backend in Hermes config:

   ```yaml
   web:
     backend: firecrawl
     use_gateway: true
   ```

Hermes may advertise the `web_search`/`web_extract` schemas as soon as the managed backend is selected, even before Nous authentication is usable. The gateway therefore performs one real `web_search` and one real `web_extract` smoke call at startup and treats nested provider errors as not-ready. `/readyz` becomes `200` only when those calls succeed; otherwise the process stays degraded and tunnel cutover is forbidden. Provider/auth changes require a gateway restart so readiness and the MCP schema remain stable for the process lifetime.

## Tests

```bash
PYTHONPATH=src:/srv/agents/src/hermes-agent \
  /home/hermes/.hermes/venvs/hermes/bin/python -m unittest discover -s tests -v
```

Local integration gates before tunnel cutover:

1. Context Mode `ctx_execute` smoke.
2. `vision_analyze`, `skills_list`, native file/terminal/process tools, `image_generate`, `video_analyze`, and the exact 10 Camofox-compatible `browser_*` tools; explicit absence of `computer_use`, `browser_exec`, and `browser_cdp`.
3. Guarded capability checks: real `session_search`, read-only `cronjob list`, and scoped delegation-tool policy; one small delegated child smoke when resource headroom permits.
4. Honcho profile/context/search and controlled `memory_conclude` create/readback/delete.
5. Gateway MCP initialize + exact 50-tool `tools/list` + representative calls, including a read-only Camofox browser navigation/snapshot smoke and Serena semantic symbol/reference smoke.
6. `healthz` and `readyz` readback, including Serena readiness.
7. systemd restart and enabled-state readback.
8. Only after `readyz=200`: point Tunnel Client `main` from `3050/mcp` to `3060/mcp`; rollback is the inverse URL change plus Tunnel Client restart.

## systemd

The repository ships `systemd/hermes-mcp-gateway.service`. It uses `Wants`/`After` rather than hard `Requires`, binds the application itself to loopback, imports the same Hermes runtime environment files, and keeps upstream failures from cascading through systemd dependency teardown.
