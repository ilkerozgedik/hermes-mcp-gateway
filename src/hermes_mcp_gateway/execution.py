"""Per-request process boundary around upstream Hermes native execute_code.

This is lifecycle isolation, not an OS sandbox. Confirmed Python has the service
user's filesystem privileges; neither the tool allowlist nor env scrub is a jail.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
from uuid import uuid4

import anyio
import psutil
from agent.redact import redact_sensitive_text
from mcp import types

from .config import GatewayConfig

RPC_TOOLS = frozenset({'read_file', 'write_file', 'search_files', 'patch', 'terminal', 'web_search', 'web_extract'})


def execution_schema() -> types.Tool:
    from tools.code_execution_tool import build_execute_code_schema
    native = build_execute_code_schema(set(RPC_TOOLS))
    params = native['parameters']
    params['properties']['confirmed'] = {
        'type': 'boolean', 'const': True,
        'description': 'Must be true only after explicit user approval of this arbitrary Python script. This is client-attested consent, not a sandbox.',
    }
    params['required'] = ['code', 'confirmed']
    params['additionalProperties'] = False
    return types.Tool(
        name='execute_code', input_schema=params,
        description=(
            'Run upstream Hermes native Python/tool RPC in an isolated per-request worker. '
            'Requires confirmed=true after explicit user approval: Python can mutate files or spawn subprocesses as the service user. '
            'Interpreter state does not persist between Gateway requests, even for stateful MCP clients. '
            'Use absolute paths or terminal(workdir=...) for project work. No client-supplied owner is accepted. '
            'The native environment scrub, RPC allowlist, tool-call and stdout limits remain active. '
            'Gateway enforces a 300-second total deadline, 64KiB code cap, bounded concurrency and redacted output. '
            'Cancellation kills the worker process tree, but is not rollback. Never automatically retry mutations.\n\n'
            + '\n'.join(line for name, line in __import__('tools.code_execution_tool', fromlist=['_TOOL_DOC_LINES'])._TOOL_DOC_LINES if name in RPC_TOOLS)
        ),
    )


def _result(payload, *, limit, failed=False):
    text = redact_sensitive_text(json.dumps(payload, ensure_ascii=False, default=str), force=True)
    if len(text) > limit:
        text = text[:max(0, limit-48)] + '\n...[truncated by hermes-mcp-gateway]'
    return types.CallToolResult(content=[types.TextContent(text=text)], is_error=failed)


def _remember_tree(proc, known):
    try:
        root = psutil.Process(proc.pid)
        for child in [root, *root.children(recursive=True)]:
            known[(child.pid, child.create_time())] = child
    except psutil.Error:
        pass


async def _terminate(proc, known):
    _remember_tree(proc, known)
    # Signal both the worker's process group and tracked descendants: Hermes
    # kernels and user subprocesses may start their own sessions/process groups.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for child in reversed(list(known.values())):
            try:
                if child.is_running():
                    child.send_signal(sig)
            except psutil.Error:
                pass
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            await asyncio.sleep(.15)
    await proc.wait()


class NativeExecution:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self._slots = asyncio.Semaphore(config.execution_max_concurrent)
        self._active = {}
        self._calls = set()
        self._closed = False

    async def close(self):
        self._closed = True
        # Cancel/await whole calls, including queued work and shielded launches
        # not yet registered as active processes. Their finally blocks reap them.
        calls = [task for task in self._calls if task is not asyncio.current_task()]
        for task in calls:
            task.cancel()
        await asyncio.gather(*calls, return_exceptions=True)
        for proc, known in list(self._active.values()):
            await _terminate(proc, known)

    async def call(self, arguments):
        limit = self.config.max_text_chars
        def error(message):
            return _result({'status':'error','error':message}, limit=limit, failed=True)
        if not isinstance(arguments, dict) or arguments.get('confirmed') is not True:
            return error('execute_code requires confirmed=true after explicit user approval of this script')
        if set(arguments) - {'code','reset','confirmed'}:
            return error('unsupported execute_code argument')
        code = arguments.get('code')
        if not isinstance(code, str) or not code.strip() or len(code.encode('utf-8')) > self.config.execution_max_code_bytes:
            return error('code must be a nonempty Python string within the 64KiB request limit')
        if 'reset' in arguments and type(arguments['reset']) is not bool:
            return error('reset must be boolean')
        if self._closed:
            return error('native execution is shutting down')
        caller = asyncio.current_task()
        self._calls.add(caller)
        # Include queue wait in the deadline: bounded execution, not an unbounded
        # backlog whose timeout starts only when a worker becomes available.
        try:
            async with asyncio.timeout(self.config.execution_timeout_seconds):
                async with self._slots:
                    if self._closed:
                        return error('native execution is shutting down')
                    return await self._run(code)
        except TimeoutError:
            return error('native execution timeout; worker process tree terminated (no replay)')
        finally:
            self._calls.discard(caller)

    async def _run(self, code):
        owner = 'chatgpt:execution:' + uuid4().hex
        env = dict(os.environ)
        # Never inherit another chat's identity/approval or a blanket YOLO flag.
        for key in list(env):
            if key.startswith('HERMES_SESSION_') or key in {'HERMES_YOLO','HERMES_INTERACTIVE','HERMES_EXEC_ASK','HERMES_GATEWAY_SESSION','HERMES_CRON_SESSION','HERMES_SINGLE_QUERY_SESSION'}:
                env.pop(key, None)
        env.update(HERMES_HOME=self.config.hermes_home, HERMES_QUIET='1', HERMES_REDACT_SECRETS='true')
        # Do not depend on callers starting from the repo or on relative PYTHONPATH.
        from pathlib import Path
        import tools
        package_root = str(Path(__file__).resolve().parents[1])
        hermes_root = str(Path(tools.__file__).resolve().parent.parent)
        env['PYTHONPATH'] = os.pathsep.join([package_root, hermes_root])
        launch = asyncio.create_task(asyncio.create_subprocess_exec(
            self.config.hermes_python, '-m', 'hermes_mcp_gateway.execution_worker',
            cwd=self.config.hermes_cwd, env=env, start_new_session=True,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        ))
        proc = None
        known = {}
        tracker = None
        try:
            proc = await asyncio.shield(launch)
            self._active[owner] = (proc, known)
            if self._closed:
                raise RuntimeError("native execution shut down during worker launch")
            async def track():
                while proc.returncode is None:
                    await asyncio.to_thread(_remember_tree, proc, known)
                    await asyncio.sleep(.1)
            tracker = asyncio.create_task(track())
            async def bounded_read(stream):
                chunks, size = [], 0
                while chunk := await stream.read(16384):
                    size += len(chunk)
                    if size > self.config.execution_max_worker_bytes:
                        raise ValueError('native worker output exceeded transport budget')
                    chunks.append(chunk)
                return b''.join(chunks)
            payload = json.dumps({'code':code, 'owner':owner, 'confirmed':True}).encode()
            proc.stdin.write(payload)
            await proc.stdin.drain()
            proc.stdin.close()
            stdout, stderr = await asyncio.gather(bounded_read(proc.stdout), bounded_read(proc.stderr))
            await proc.wait()
            if proc.returncode != 0:
                raise RuntimeError('native worker failed: ' + stderr.decode(errors='replace')[:2000])
            result = json.loads(stdout)
            failed = not isinstance(result, dict) or result.get('status') != 'success'
            return _result(result, limit=self.config.max_text_chars, failed=failed)
        finally:
            # anyio level-cancellation must not cancel cleanup on every await.
            with anyio.CancelScope(shield=True):
                if proc is None:
                    proc = await asyncio.shield(launch)
                await _terminate(proc, known)
                if tracker is not None:
                    tracker.cancel()
                    await asyncio.gather(tracker, return_exceptions=True)
                self._active.pop(owner, None)
