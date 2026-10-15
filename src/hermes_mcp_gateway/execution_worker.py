"""Private one-shot bridge into unmodified Hermes native execute_code."""
from __future__ import annotations

import contextlib
import json
import sys


def _adopt_orphans():
    # Linux PR_SET_CHILD_SUBREAPER: double-forked children stay owned by this
    # worker rather than escaping to PID 1 before the Gateway can observe them.
    import ctypes
    if sys.platform != 'linux':
        raise RuntimeError('native Gateway execution requires Linux child-subreaper support')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot establish worker child subreaper')


def _reap_children():
    import psutil
    children = psutil.Process().children(recursive=True)
    for child in reversed(children):
        try:
            child.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(children, timeout=2)


def main():
    _adopt_orphans()
    # stdout is a protocol channel; all import/tool diagnostics go to stderr.
    response_channel = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        raw_input = sys.stdin.buffer.read(512 * 1024 + 1)
        if len(raw_input) > 512 * 1024:
            raise ValueError("worker input exceeds transport budget")
        payload = json.loads(raw_input)
        if payload.get('confirmed') is not True:
            raise ValueError('unconfirmed execution refused')
        owner, code = payload['owner'], payload['code']
        if not isinstance(owner, str) or not owner.startswith('chatgpt:execution:'):
            raise ValueError('server execution identity required')
        from gateway.session_context import set_session_vars, clear_session_vars
        from tools.approval_context import set_hermes_interactive_context, reset_hermes_interactive_context
        from tools.terminal_tool import set_approval_callback
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS
        from tools.code_kernel import shutdown_all_kernels
        from model_tools import handle_function_call
        from agent.redact import redact_sensitive_text
        from .execution import RPC_TOOLS
        # A nonempty explicit grant is critical: upstream empty intersections
        # intentionally fall back to the full native sandbox allowlist.
        allowed = sorted(RPC_TOOLS & SANDBOX_ALLOWED_TOOLS)
        if not allowed or set(allowed) != RPC_TOOLS:
            raise RuntimeError('native sandbox tool contract changed; refusing execution')
        tokens = set_session_vars(platform='mcp', source='tool', session_key=owner,
                                  session_id=owner, async_delivery=False,
                                  cron_session='', session_history_delivery='')
        interactive = set_hermes_interactive_context(True)
        expected = redact_sensitive_text("execute_code <<'PY'\n" + code + '\nPY')
        consumed = False
        def approve(command, description, **kwargs):
            nonlocal consumed
            # Client-attested confirmation grants only this already-described
            # script once, never a session/always grant or another nested action.
            if not consumed and command == expected and description.startswith('execute_code script execution.'):
                consumed = True
                return 'once'
            return 'deny'
        set_approval_callback(approve)
        try:
            raw = handle_function_call(
                'execute_code', {'code':code, 'reset':True}, task_id=owner,
                session_id=owner, tool_call_id=owner.rsplit(':',1)[-1],
                enabled_tools=allowed,
                enabled_toolsets=['file','terminal','web','code_execution'],
            )
            result = json.loads(raw)
        finally:
            shutdown_all_kernels()
            _reap_children()
            set_approval_callback(None)
            reset_hermes_interactive_context(interactive)
            clear_session_vars(tokens)
    response_channel.write(json.dumps(result, ensure_ascii=False))
    response_channel.flush()


if __name__ == '__main__':
    main()
