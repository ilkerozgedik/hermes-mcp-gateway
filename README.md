# Hermes MCP Gateway

Single localhost-only MCP endpoint that aggregates native Hermes execution, a strict Hermes tool allowlist, Samchon Graph, guarded capabilities and Honcho memory through progressive tool disclosure without patching upstream source.

## Endpoint

- MCP: `http://127.0.0.1:3060/mcp`
- Process health: `http://127.0.0.1:3060/healthz`
- Final-cutover readiness: `http://127.0.0.1:3060/readyz`

`healthz` is a local liveness check that never calls providers. `readyz` requires the curated native tools, web provider, Samchon Graph, LSP adapter, guarded capabilities, Honcho and startup files. The catalog is frozen until restart. Modern MCP `2026-07-28` HTTP is sessionless: `execute_code` uses an independent server-generated owner per request. Browser, direct `terminal` and `process_manage` require a validated legacy SDK session or a gateway-issued `session_token`; raw client headers and caller-provided owner IDs never grant ownership. Modern clients should discover `session_open`, call it once, and pass its returned token as the top-level `tool_call.session_token` on calls that need browser/terminal/process continuity. Tokens are random, revocable, expire after 15 minutes, and are capped at 256 active capabilities. `session_close` revokes the capability, kills only its owner's native processes, and drops only its local browser binding. Independent shell pipelines can use `execute_code` with `hermes_tools.terminal` without a separate Gateway confirmation parameter.

## Tool policy

Native direct tools bind Hermes' existing session ContextVars inside the request worker, so terminal cwd/environment and process ownership remain client-scoped. Per-call hook IDs are independently generated; they distinguish concurrent executions without granting resource ownership or weakening approval hooks.

HTTP disconnects/timeouts kill and reap native execution worker trees, including tracked descendants in separate process groups. The Gateway never retries execution. Other thread-backed native tools retain best-effort cancellation: cancellation is not rollback. Arbitrary Python has the service user's privileges; this is not an OS sandbox.

`tools/list` exposes only six tools: `tool_search`, `tool_describe`, `tool_call`, `startup_context`, `memory_context`, and `skills_list`. Deferred tools are available only through the validated one-call discovery bridge. Search uses Hermes native BM25 over the frozen catalog; descriptions expose authoritative input schemas.

Hermes is default-deny and may expose only:

- `web_search`, `web_extract`
- `vision_analyze`
- `skills_list`, `skill_view`
- native filesystem: `read_file`, `write_file`, `patch`, `search_files`
- native code: `execute_code` through isolated per-request worker processes
- native terminal/process: `terminal`, `process_manage`
- media: `image_generate`, `video_analyze`
- Camofox-compatible native browser tools: `browser_navigate`, `browser_click`, `browser_type`, `browser_press`, `browser_snapshot`, `browser_scroll`, `browser_back`, `browser_get_images`, `browser_console`, `browser_vision`

The gateway does not expose `browser_exec`, `browser_cdp`, `computer_use`, raw Hermes memory/todo state, TTS, or Kanban tools. `session_search`, `delegate_task`, and `cronjob` are exposed only through gateway-owned guarded adapters, not through the stateless Hermes allowlist.

Hermes Agent also uses the existing read-only LSP adapter directly through a separate stdio MCP entrypoint: `python -m hermes_mcp_gateway.lsp_stdio`. It exposes only `lsp_diagnostics`, `lsp_hover`, `lsp_definition`, `lsp_references`, `lsp_document_symbols`, and `lsp_status`. The same project-root allowlist, manual-only language-server installation policy, and LSP workspace trust rules apply. This is a native Hermes MCP connection, not a loopback call into the ChatGPT Gateway.

Samchon Graph is exposed as one `inspect_code_graph` tool. The caller supplies an absolute `cwd`; the gateway resolves it inside the allowlisted work/source roots and reuses a bounded resident MCP session per project root. The upstream MCP surface must contain exactly that one graph tool.

Guarded Hermes capabilities are:

- `session_search` — read-only search over the local Hermes session database. Cross-profile override is intentionally unavailable; results are historical conversation context, not proof of current external state.
- `delegate_task` — synchronous leaf delegation, one or two children, subject to host-side permission controls. Children inherit `web`, `vision`, `skills`, `file`, `terminal`, and `code_execution` native toolsets. Recursive delegation is unavailable; native child session/security policies apply.
- `cronjob` — read-only `list`, plus guarded `create`, `update`, `pause`, `resume`, `remove`, and `run`. Mutations use host-side permission controls. Model/provider/base-URL overrides and script/no-agent/monitor execution fields are not exposed; delivery defaults to `local`.

The catalog exposes exactly six public tools. Native Git worktrees replace the retired coordinator; no optional coordinator service is needed. No tools are dynamically added to the catalog during a session; a Gateway restart refreshes discovery. A new ChatGPT MCP connection may be required to refresh the public schemas.

Gateway-owned startup tool:

- `startup_context` — reads only the four fixed Hermes startup files (`.hermes.md`, `SOUL.md`, `MEMORY.md`, and `USER.md`). It accepts no path and is not a generic filesystem surface.

Memory tools are:

- `memory_profile` (read-only)
- `memory_search`
- `memory_context`
- `memory_reasoning`
- `memory_conclude`

Memory identity is resolved through Hermes Honcho configuration. The existing `workspace=hermes` and user peer are preserved; only this adapter's assistant peer is `chatgpt`. Conclusion writes require a durable kind (`preference`, `decision`, `architecture`, `project_state`) and reject secret-like or explicitly temporary content. List/delete operations retain Honcho semantics.

## Execution Guard

Direct MCP `terminal` is restricted to short foreground commands: at most
**30 seconds**, with `timeout=30` explicitly supplied to Hermes if omitted.
Any direct background request, PTY mode, foreground request above the cap,
or background-only notification/persistence option fails **before execution**
with `LONG_RUNNING_OPERATION_REQUIRES_JOB`. Use `job_start` for build,
test, deploy or other potentially lengthy work. Native terminal enforces
its subprocess deadline; cancelling only a Python thread would **not** safely
stop its child processes, so the gateway deliberately does not use
`asyncio.wait_for(to_thread(...))` as a hard-kill mechanism.

`execute_code` retains its isolated worker/process-tree cleanup and now
has a **30-second** total deadline, including queue wait. Neither tool's
timeout rolls back effects already performed. Tool selection remains the
agent's choice, but an incorrect long/background choice is rejected rather
than silently started unrecoverably. The dedicated job adapter is exempt
from these direct terminal restrictions.

## Resumable background jobs

For a build, test suite, or other command longer than a brief terminal call, use the
deferred `job_start` tool instead of waiting on a foreground `terminal` request.
Open an MCP session with `session_open` first. `job_start` accepts a command,
a workdir under the configured allowlisted project roots, an optional random
`idempotency_key` (16–128 characters), and a `max_runtime` of 1–7200 seconds
(default: 1800). It uses existing Hermes `terminal(background=true,
persist_on_release=true)`; the Gateway is **not** a second command executor.

**Parallel agents:** Four jobs globally, at most two per verified MCP session.
Admission is atomic in SQLite even when two Gateway instances race. Active
jobs cannot use the same or nested working directories: create a separate
Git worktree for each agent, even when working on the same repository.
Status/cancel synchronize per job, so a slow native poll cannot block an
unrelated job. A full quota or busy workspace returns an explicit error,
not an unbounded queue. The per-session allowance is **not a durable
per-person quota**: a new session gets a fresh identity. Enforcing limits
across all sessions for one person would require a trusted stable principal
from the authenticated ingress. Interactive terminal, browser, Git remotes,
databases, and manual filesystem changes are not workspace-locked by this
job admission mechanism; coordinate shared operations separately.

**Save the returned `job_id` and one-time `recovery_token` securely.**
A new MCP session may call `job_status` or `job_cancel` with both values;
the 15-minute interactive session's expiry does not kill an accepted job.
The ID alone grants no access. The Gateway stores only a hash of the token
and bounded, redacted output in a permission-restricted SQLite receipt.
A status may be `running`, `succeeded`, `failed`, `cancelled`,
`timed_out`, or `unknown`. A completed result remains available for seven
days, subject to Hermes' existing native process/result retention.

Never blindly retry `git push`, deploy, migrations, or other side effects
after an ambiguous or missing response. The same idempotency key cannot
launch the same command again; a duplicate returns the earlier job ID,
**not** the recovery token. Losing the one-time token can make recovery
impossible. Check the external system's actual state before any new attempt.

Gateway restarts may terminate live processes with their systemd cgroup.
A newly started Gateway can recover durable receipts and query Hermes'
native process registry if the underlying process still exists, but it
must report `unknown` when actual completion cannot be established.
This mechanism does not guarantee recovery from a ChatGPT application-side
pause or an outage of the VPS itself. Original `terminal` calls retain
their current behavior; only work launched via `job_start` is resumable.

## Native execution limits

Discover and describe `execute_code`, then call it through `tool_call` with `code`. The Gateway does not require a redundant client-supplied confirmation flag: user consent is handled by the ChatGPT host. This is not authorization proof at the MCP transport; access remains restricted to loopback and the authenticated Tunnel Client. The Gateway's allowlists, owner isolation, and resource limits remain in force.

Every call invokes unmodified upstream Hermes `execute_code` inside a new Linux worker. A child subreaper keeps double-forked descendants owned until cleanup. Interpreter state **does not persist** between Gateway requests; `reset` is accepted for native schema compatibility but each worker already starts fresh. The native whole-script approval guard runs in a request-bound context; the callback grants only the dispatched script once, and denies further approval requests. No session/always/YOLO grant is installed. Existing operator approval configuration still applies.

Limits: 64 KiB UTF-8 code, two concurrent workers, **30-second total deadline** including queue wait, 1 MiB per worker stdout/stderr transport stream, and 65,536-character redacted Gateway output. Native defaults additionally cap tool calls at 50 and stdout at 50 KiB; configured native limits can be stricter. Child environment scrubbing and the explicit native RPC allowlist remain active. Only `read_file`, `write_file`, `search_files`, `patch`, foreground `terminal`, `web_search`, and `web_extract` are granted over RPC. Direct Python imports/subprocesses are not restricted by that allowlist and can access user files; execution is not a security jail. Context Mode indexing/search is intentionally not emulated.

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

1. Modern sessionless HTTP `execute_code` smoke: no confirmation field required, native file/search/terminal RPC succeeds, requests do not share state, cancellation kills native descendants.
2. `vision_analyze`, `skills_list`, native file/terminal/`process_manage` tools, `image_generate`, `video_analyze`, and the exact 10 Camofox-compatible `browser_*` tools; explicit absence of `computer_use`, `browser_exec`, and `browser_cdp`.
3. Guarded capability checks: real `session_search`, read-only `cronjob list`, and scoped delegation-tool policy; one small delegated child smoke when resource headroom permits.
4. Honcho profile/context/search and controlled `memory_conclude` create/readback/delete.
5. Gateway MCP initialize + exact six-tool `tools/list`, lexical search + describe, denied direct hidden calls, and representative `tool_call` calls covering LSP, Camofox browser navigation/snapshot, and cwd-scoped Samchon Graph.
6. `healthz` and `readyz` readback, including Samchon Graph readiness.
7. systemd restart and enabled-state readback.
8. Keep Tunnel Client `main` pointed only at `3060/mcp`; its historical profile name is an identity label, not a dependency. Retire old components only after readiness and native execution gates pass.

## systemd

The repository ships `systemd/hermes-mcp-gateway.service`. It uses `Wants`/`After` rather than hard `Requires`, binds the application itself to loopback, imports the same Hermes runtime environment files, and keeps upstream failures from cascading through systemd dependency teardown.

## Native Git worktrees

Coding agents use Git's built-in `worktree` commands directly in their own terminal sessions. No task coordinator, workspace MCP tools, or separate database is required. Each agent works in a unique branch/worktree and must preserve uncommitted or unpushed changes. Worktrees isolate Git state, not OS-level filesystem access.
