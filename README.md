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
- `delegate_task` — the long synchronous Gateway path is rejected before model inference. Use supported native Hermes automation through a prepared background job instead of holding an MCP response open. Hermes' own native delegation remains unchanged.
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

## Backend budgets and request diagnostics

The 30-second worker limit is not a universal tool deadline. Hermes MCP reads
and LSP RPCs use their configured 30-second budgets; Graph has a separate
120-second call budget. Camofox owns its `browser.command_timeout` HTTP budget
(default 30 seconds); auxiliary vision uses native download/provider budgets
and bounded retries. Several backend steps may run sequentially, so these are
not promises that the entire browser/media call ends within 30 seconds.
Gateway cancellation does not forcibly kill thread-backed native handlers.
Those handlers remain counted until actual completion, including after an HTTP
disconnect. Long synchronous Gateway delegation is rejected before inference.

Request telemetry records generated request IDs, tool, elapsed time and result
category. A validated inbound request ID is correlated by a truncated SHA256,
not logged verbatim or used for authorization. Arguments, commands, recovery
credentials and raw headers are excluded. Compare these events with Tunnel
metrics; an empty-poll timeout is not itself proof of a failed tool request.

## Resumable background jobs

For builds, tests and other long commands use the two-phase deferred protocol:

1. Open an interactive capability with `session_open`.
2. `job_prepare(command, workdir, max_runtime?, idempotency_key?)` durably freezes
   the request and returns `job_id` plus a one-time `recovery_token`. **No command
   executes during preparation.**
3. Save both values outside Git in a private, owner-only recovery file before
   continuing. Do not put recovery tokens in logs, durable memory or chat prose.
4. `job_start(job_id, recovery_token)` with a valid interactive session starts
   only that frozen request. It no longer accepts a command or workdir.
5. After any missing start response, use the same credential with `job_status`
   from a new HTTP connection. Repeating an authorized start cannot relaunch it.

Preparation accepts an allowlisted workdir, an optional random idempotency key
(16–128 characters) and `max_runtime` of 1–7200 seconds (default: 1800).
Unused preparation is bounded and expires; it does not reserve active capacity
or lock a workspace. Active admission happens atomically at start. Work still
uses Hermes `terminal(background=true, persist_on_release=true)`; the Gateway is
**not** a second executor. The background deadline is enforced by bounded
maintenance/termination, not an exact wall-clock kill guarantee.

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

`job_status` and `job_cancel` authorize using the job credential alone; an
unrelated expired interactive session token cannot block recovery. This does
not relax session validation for prepare/start, browser, terminal or process
tools. A session's fixed 15-minute expiry does not kill an accepted job.
The ID alone grants no access. SQLite stores only the recovery token's hash,
the temporarily needed prepared command, and bounded redacted output in a
0600 database. The command payload is discarded when start is admitted.
States include `prepared`, `starting`, `running`, `succeeded`, `failed`,
`cancelled`, `timed_out`, `expired` and `unknown`. Completed receipts are retained
for seven days; recovery before finalization also depends on Hermes' native
process/result retention.

Never blindly retry push, deploy, migrations or other mutations. A duplicate
preparation returns the earlier ID/state, **not** its token. If the preparation
response is lost, no start has yet been authorized by this client: inspect the
duplicate's state before preparing again with a new key. Once start was issued,
use the previously saved credential; never substitute a new request/key merely
because the response is missing. Existing legacy receipts remain recoverable.
Idempotent launch admission does not promise exactly-once external effects.

Gateway restarts may terminate live processes with their systemd cgroup.
A newly started Gateway can recover durable receipts and query Hermes'
native process registry if the underlying process still exists, but it
must report `unknown` when actual completion cannot be established.
An unverifiable `unknown` remains a capacity/workspace quarantine even after its
deadline. Reconcile it against the job-owned native process and external target;
do not label it succeeded or free its workspace without evidence. Deployment
must abort rather than force-cancel these jobs. This mechanism cannot force a
paused ChatGPT client to resume, or guarantee live work survives a VPS outage.
Original terminal calls are not resumable jobs.

### Operator admission and deployment

The local-only `scripts/job_admission.py` supports `status`, `drain`, `resume`
and consistent SQLite `snapshot`. It is not exposed over MCP. Run it as the
database owner, with `--database /home/hermes/.hermes/gateway-jobs.sqlite3`.
`drain` and `resume` also require the same private `--owner-token` (32–128 safe
characters). The gate claim is atomic, stores only the token's hash, and refuses
to release another operator's pause; `status` never returns ownership tokens.
Keep database and snapshot paths under trusted-owner directories without
group/world write access. Unsafe parents and symlinks are rejected before
SQLite opens either path.
Draining pauses fresh starts atomically while recovery and authorized duplicate
starts remain available. It waits for starting/running/unknown jobs; on timeout
it fails closed and leaves admission paused for the deploy owner to restore.
Prepared but unstarted requests do not block drain. Never restore an old job
snapshot over live receipts as an ordinary rollback step.

The bootstrap repository owns the scoped deployment and host lock. It stops
Tunnel ingress, takes a consistent backup, drains jobs, then checks the cheap
`/healthz.inflight_requests` and `inflight_native_calls` counters before restarting
Gateway. Native threads remain counted after an HTTP waiter is cancelled; a zero
handler count alone does not prove the native work stopped. Legacy Gateways
without that counter require explicit operator acknowledgment rather than a
false claim that foreground work was measured idle. Health/readiness success
alone does not prove the actual ChatGPT control-plane route.

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
3. Guarded capability checks: real `session_search`, read-only `cronjob list`, and fail-fast synchronous Gateway delegation without model inference.
4. Honcho profile/context/search and controlled `memory_conclude` create/readback/delete.
5. Gateway MCP initialize + exact six-tool `tools/list`, lexical search + describe, denied direct hidden calls, and representative `tool_call` calls covering LSP, Camofox browser navigation/snapshot, and cwd-scoped Samchon Graph.
6. `healthz` and `readyz` readback, including Samchon Graph readiness.
7. systemd restart and enabled-state readback.
8. Keep Tunnel Client `main` pointed only at `3060/mcp`; its historical profile name is an identity label, not a dependency. Retire old components only after readiness and native execution gates pass.

## systemd

The repository ships `systemd/hermes-mcp-gateway.service`. It uses `Wants`/`After` rather than hard `Requires`, binds the application itself to loopback, imports the same Hermes runtime environment files, and keeps upstream failures from cascading through systemd dependency teardown.

## Native Git worktrees

Coding agents use Git's built-in `worktree` commands directly in their own terminal sessions. No task coordinator, workspace MCP tools, or separate database is required. Each agent works in a unique branch/worktree and must preserve uncommitted or unpushed changes. Worktrees isolate Git state, not OS-level filesystem access.
