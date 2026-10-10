# Hermes MCP Gateway

Single localhost-only MCP endpoint that aggregates Context Mode, a strict Hermes tool allowlist, Samchon Graph, guarded capabilities and Honcho memory through progressive tool disclosure without patching upstream source.

## Endpoint

- MCP: `http://127.0.0.1:3060/mcp`
- Process health: `http://127.0.0.1:3060/healthz`
- Final-cutover readiness: `http://127.0.0.1:3060/readyz`

`healthz` stays available when a running upstream later degrades. `readyz` is `200` only when Context Mode, the curated Hermes tools including web, the guarded Hermes capability runtime, and Honcho are all available. Tool discovery is frozen for the process lifetime; provider/config changes require a gateway restart, keeping MCP schemas stable within a running process.

## Tool policy

The complete 56-tool allowlisted catalog remains available, but `tools/list` exposes only six tools: `tool_search`, `tool_describe`, `tool_call`, `startup_context`, `memory_context`, and `skills_list`. The other 53 tools are accessible only through the discovery bridge. Search uses Hermes' native BM25 implementation over the frozen Gateway catalog (no separate search service/index); `tool_describe` returns authoritative MCP input schemas. `tool_call` executes **one** described tool via the existing Gateway dispatcher. Direct MCP calls to hidden tools fail closed. Gateway safety policies and backend ownership do not change.

Hermes is default-deny and may expose only:

- `web_search`, `web_extract`
- `vision_analyze`
- `skills_list`, `skill_view`
- native filesystem: `read_file`, `write_file`, `patch`, `search_files`
- native terminal/process: `terminal`, `process_manage`
- media: `image_generate`, `video_analyze`
- Camofox-compatible native browser tools: `browser_navigate`, `browser_click`, `browser_type`, `browser_press`, `browser_snapshot`, `browser_scroll`, `browser_back`, `browser_get_images`, `browser_console`, `browser_vision`

The gateway does not expose `browser_exec`, `browser_cdp`, `computer_use`, raw Hermes memory/todo state, TTS, or Kanban tools. `session_search`, `delegate_task`, and `cronjob` are exposed only through gateway-owned guarded adapters, not through the stateless Hermes allowlist.

Samchon Graph is exposed as one `inspect_code_graph` tool. The caller supplies an absolute `cwd`; the gateway resolves it inside the allowlisted work/source roots and reuses a bounded resident MCP session per project root. The upstream MCP surface must contain exactly that one graph tool.

Guarded Hermes capabilities are:

- `session_search` — read-only search over the local Hermes session database. Cross-profile override is intentionally unavailable; results are historical conversation context, not proof of current external state.
- `delegate_task` — synchronous leaf delegation only, one or two children maximum, and `confirmed=true` after explicit user approval because it spends model inference. Children inherit only Hermes `web`, `vision`, `skills`, and the `mcp-context-mode` toolset. Context Mode MCP tools are reached through Hermes' scoped `tool_search`/`tool_describe`/`tool_call` bridge; native Hermes terminal/file/code tools and recursive delegation remain out of scope.
- `cronjob` — read-only `list`, plus guarded `create`, `update`, `pause`, `resume`, `remove`, and `run`. Mutations require `confirmed=true`. Model/provider/base-URL overrides and script/no-agent/monitor execution fields are not exposed; delivery defaults to `local`.

The underlying catalog is 50 tools without the opt-in Coordinator, or 56 with its six tools enabled. Both configurations expose the same six public tools. No tools are dynamically added to the catalog during a session; a Gateway restart refreshes discovery. A new ChatGPT MCP connection may be required to refresh the public schemas.

Gateway-owned startup tool:

- `startup_context` — reads only the four fixed Hermes startup files (`.hermes.md`, `SOUL.md`, `MEMORY.md`, and `USER.md`). It accepts no path and is not a generic filesystem surface.

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
2. `vision_analyze`, `skills_list`, native file/terminal/`process_manage` tools, `image_generate`, `video_analyze`, and the exact 10 Camofox-compatible `browser_*` tools; explicit absence of `computer_use`, `browser_exec`, and `browser_cdp`.
3. Guarded capability checks: real `session_search`, read-only `cronjob list`, and scoped delegation-tool policy; one small delegated child smoke when resource headroom permits.
4. Honcho profile/context/search and controlled `memory_conclude` create/readback/delete.
5. Gateway MCP initialize + exact six-tool `tools/list`, lexical search + describe, denied direct hidden calls, and representative `tool_call` calls covering LSP, Camofox browser navigation/snapshot, and cwd-scoped Samchon Graph.
6. `healthz` and `readyz` readback, including Samchon Graph readiness.
7. systemd restart and enabled-state readback.
8. Only after `readyz=200`: point Tunnel Client `main` from `3050/mcp` to `3060/mcp`; rollback is the inverse URL change plus Tunnel Client restart.

## systemd

The repository ships `systemd/hermes-mcp-gateway.service`. It uses `Wants`/`After` rather than hard `Requires`, binds the application itself to loopback, imports the same Hermes runtime environment files, and keeps upstream failures from cascading through systemd dependency teardown.

## Native Git workspaces (new; coordinator still available during pilot)

The Gateway exposes two discoverable tools: `workspace_open(repo)` and
`workspace_close(path)`. Repositories are direct Git checkouts beneath
`/home/hermes/work`; each open creates a unique `agent/<id>` branch and a
worktree beneath `/home/hermes/worktrees/<repo>/<id>`. Non-Git directories
return `not_git`; read-only analysis needs no worktree. Use the returned
worktree as the working directory for all edits, tests, and commits.

Close refuses dirty/ignored files and commits absent from other local or
remote-tracking refs, never forces removal, and leaves branches intact.
Git worktrees are *not* operating-system sandboxes. Two distinct ChatGPT
sessions must still be piloted before retiring the existing Coordinator.

## Multi-agent coordinator (opt-in)

A separate loopback-only MCP server on `127.0.0.1:3061/mcp` owns GitHub Issues and isolated Git worktrees. It shares **no in-process session state** with Hermes or ChatGPT; SQLite transactions control exclusive ownership. The Gateway adds its six tools to the hidden catalog only with `AGENT_COORDINATOR_ENABLED=1` (56 underlying tools). Hermes connects directly through the `multi-agent-coordinator` MCP server entry in `~/.hermes/config.yaml`; never point Hermes back to the aggregator on `:3060`.

Tools: `tasks_list(repo)`, `task_claim(repo, issue, agent)`, `task_status(repo, issue)`, `task_heartbeat(repo, issue, claim_id)`, `task_finish(repo, issue, claim_id, pr_number)`, and `task_release(repo, issue, claim_id)`. The opaque `claim_id` grants ownership for a single Issue; it must remain private and must not be committed, copied into GitHub comments, or logged. A claim creates `agent/issue-N` from `origin/main` in `/home/hermes/worktrees/<repo>/issue-N`. GitHub Issues need an `agent:ready` label; state is projected to `agent:active`, `agent:review`, or `agent:blocked`. GitHub label synchronization failures are reported, not retried as task claims.

Agents should list available Issues, claim one, work only in their returned worktree, maintain the lease via heartbeat, commit/push their feature branch, create a PR against `main`, and call `task_finish`. The Coordinator does **not** merge PRs, deploy, delete worktrees, or automatically reassign expired leases. Expired active claims become blocked for manual recovery; completed PR reviews stay owned. Only the existing CI-gated GitOps promotion deploys production. SQLite stores hashed ownership capabilities and lives under `/srv/agents/runtime/multi-agent`; never put it in Git.

The `systemd/multi-agent-coordinator.service` unit and the Hermes bootstrap install/start it before the Gateway. GitHub CLI authorization and the existing pinned checkout remain prerequisites. For a local integration test:

```bash
PYTHONPATH=src:/srv/agents/src/hermes-agent /home/hermes/.hermes/venvs/hermes/bin/python -m unittest discover -s tests -p 'test_coordinator*.py' -v
```

**Boundary:** Worktrees isolate Git state, not unrestricted shell/filesystem access. Fully unattended untrusted code execution requires separate worker OS sandboxes before enabling autonomous merges or broad write tools. Interactive ChatGPT conversations cannot be launched by the coordinator automatically.
