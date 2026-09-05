# Hermes MCP Gateway

Single localhost-only MCP endpoint that aggregates the existing Context Mode MCP, a strict allowlist from Hermes native tools, and Hermes/Honcho memory without patching any upstream source.

## Endpoint

- MCP: `http://127.0.0.1:3060/mcp`
- Process health: `http://127.0.0.1:3060/healthz`
- Final-cutover readiness: `http://127.0.0.1:3060/readyz`

`healthz` stays available when a running upstream later degrades. `readyz` is `200` only when Context Mode, the curated Hermes tools including web, Honcho, and CloakBrowser CDP are all available. Tool discovery is frozen for the process lifetime; provider/config changes require a gateway restart, keeping MCP schemas stable within a running process.

## Tool policy

Context Mode forwards discovered `ctx_*` tools and remains the only shell/filesystem/code-execution backend.

Hermes is default-deny and may expose only:

- `web_search`, `web_extract`
- `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_press`, `browser_scroll`, `browser_back`, `browser_get_images`, `browser_console`, `browser_vision`
- `vision_analyze`
- `skills_list`, `skill_view`

The gateway never exposes Hermes `terminal`, file mutation/search/process tools, delegation, session/todo state, image generation, TTS, or Kanban tools.

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

Hermes built-in `browser_*` tools attach to the existing CloakBrowser CDP at `127.0.0.1:9222`. Hermes `browser.backend` must be `off` so the Browser Use CLI does not replace the built-in browser toolset. The VPS bootstrap owns that desired state together with `browser.cdp_url=http://127.0.0.1:9222`.

No alternate browser, profile, port, or browser implementation is created.

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

Hermes advertises `web_search`/`web_extract` only when the Nous account is logged in, entitled to Tool Gateway access, and managed Firecrawl resolves successfully. Until then the process may run in degraded mode, but `/readyz` remains `503` and tunnel cutover is forbidden.

## Tests

```bash
PYTHONPATH=src:/srv/agents/src/hermes-agent \
  /home/hermes/.hermes/venvs/hermes/bin/python -m unittest discover -s tests -v
```

Local integration gates before tunnel cutover:

1. Context Mode `ctx_execute` smoke.
2. Hermes browser `navigate` + `snapshot`, proving the CDP/browser PID is unchanged.
3. `vision_analyze`, `skills_list` discovery.
4. Honcho profile/context/search and controlled `memory_conclude` create/readback/delete.
5. Gateway MCP initialize + `tools/list` + representative calls.
6. `healthz` and `readyz` readback.
7. systemd restart and enabled-state readback.
8. Only after `readyz=200`: point Tunnel Client `main` from `3050/mcp` to `3060/mcp`; rollback is the inverse URL change plus Tunnel Client restart.

## systemd

The repository ships `systemd/hermes-mcp-gateway.service`. It uses `Wants`/`After` rather than hard `Requires`, binds the application itself to loopback, imports the same Hermes runtime environment files, and keeps upstream failures from cascading through systemd dependency teardown.
