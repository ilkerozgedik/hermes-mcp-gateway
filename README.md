# Hermes MCP Gateway

Single localhost-only MCP endpoint that aggregates the existing Context Mode MCP, a strict allowlist from Hermes native tools, three guarded Hermes capabilities, and Hermes/Honcho memory without patching any upstream source.

## Endpoint

- MCP: `http://127.0.0.1:3060/mcp`
- Process health: `http://127.0.0.1:3060/healthz`
- Final-cutover readiness: `http://127.0.0.1:3060/readyz`

`healthz` stays available when a running upstream later degrades. `readyz` is `200` only when Context Mode, the curated Hermes tools including web, the guarded Hermes capability runtime, and Honcho are all available. Tool discovery is frozen for the process lifetime; provider/config changes require a gateway restart, keeping MCP schemas stable within a running process.

## Tool policy

Context Mode forwards discovered `ctx_*` tools and remains the only shell/filesystem/code-execution backend.

Hermes is default-deny and may expose only:

- `web_search`, `web_extract`
- `vision_analyze`
- `skills_list`, `skill_view`

The gateway never exposes Hermes `browser_*`, `terminal`, native file mutation/search/process tools, raw memory/todo state, image generation, TTS, or Kanban tools. `session_search`, `delegate_task`, and `cronjob` are exposed only through gateway-owned guarded adapters, not through the stateless Hermes allowlist.

Guarded Hermes capabilities are:

- `session_search` — read-only search over the local Hermes session database. Cross-profile override is intentionally unavailable; results are historical conversation context, not proof of current external state.
- `delegate_task` — synchronous leaf delegation only, one or two children maximum, and `confirmed=true` after explicit user approval because it spends model inference. Children inherit only Hermes `web`, `vision`, `skills`, and the `mcp-context-mode` toolset. Context Mode MCP tools are reached through Hermes' scoped `tool_search`/`tool_describe`/`tool_call` bridge; native Hermes terminal/file/code tools and recursive delegation remain out of scope.
- `cronjob` — read-only `list`, plus guarded `create`, `update`, `pause`, `resume`, `remove`, and `run`. Mutations require `confirmed=true`. Model/provider/base-URL overrides and script/no-agent/monitor execution fields are not exposed; delivery defaults to `local`.

With the currently pinned Context Mode surface, the public MCP contract is 25 tools: 11 Context Mode + 5 curated Hermes stateless + 3 guarded Hermes capabilities + 5 memory + `startup_context`.

Gateway-owned startup tool:

- `startup_context` — reads only the five fixed Hermes startup files (`.hermes.md`, `SOUL.md`, `MEMORY.md`, `USER.md`, and Ponytail `SKILL.md`). It accepts no path and is not a generic filesystem surface.

Memory tools are:

- `memory_profile` (read-only)
- `memory_search`
- `memory_context`
- `memory_reasoning`
- `memory_conclude`

Memory identity is resolved through Hermes Honcho configuration. The existing `workspace=hermes` and user peer are preserved; only this adapter's assistant peer is `chatgpt`. Conclusion writes require a durable kind (`preference`, `decision`, `architecture`, `project_state`) and reject secret-like or explicitly temporary content. List/delete operations retain Honcho semantics.

## Browser

The unified ChatGPT gateway intentionally exposes no Hermes `browser_*` tools and has no CDP readiness dependency. Browser automation remains a Hermes terminal workflow through the `agent-browser` skill and canonical `agent-browser-hermes` wrapper, which reuse the existing CloakBrowser profile/CDP managed by the VPS bootstrap.

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
2. `vision_analyze`, `skills_list` discovery and explicit absence of every `browser_*` tool.
3. Guarded capability checks: real `session_search`, read-only `cronjob list`, and scoped delegation-tool policy; one small delegated child smoke when resource headroom permits.
4. Honcho profile/context/search and controlled `memory_conclude` create/readback/delete.
5. Gateway MCP initialize + exact 25-tool `tools/list` + representative calls.
6. `healthz` and `readyz` readback.
7. systemd restart and enabled-state readback.
8. Only after `readyz=200`: point Tunnel Client `main` from `3050/mcp` to `3060/mcp`; rollback is the inverse URL change plus Tunnel Client restart.

## systemd

The repository ships `systemd/hermes-mcp-gateway.service`. It uses `Wants`/`After` rather than hard `Requires`, binds the application itself to loopback, imports the same Hermes runtime environment files, and keeps upstream failures from cascading through systemd dependency teardown.
