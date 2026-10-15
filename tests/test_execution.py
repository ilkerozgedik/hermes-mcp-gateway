import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import psutil

from hermes_mcp_gateway.config import GatewayConfig
from hermes_mcp_gateway.execution import NativeExecution, _result, execution_schema


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = NativeExecution(GatewayConfig())

    async def asyncTearDown(self):
        await self.executor.close()

    async def run_code(self, code, **kwargs):
        result = await self.executor.call({'code': code, **kwargs})
        return result, json.loads(result.content[0].text)

    def test_schema_is_truthful_without_redundant_confirmation(self):
        schema = execution_schema()
        self.assertIn('code', schema.input_schema['required'])
        self.assertNotIn('confirmed', schema.input_schema['properties'])
        self.assertIn('not persist', schema.description)
        self.assertFalse(schema.input_schema['additionalProperties'])

    async def test_legacy_confirmation_argument_is_rejected(self):
        for confirmed in [None, False, True]:
            result = await self.executor.call({'code': "raise RuntimeError('not run')", 'confirmed': confirmed})
            self.assertTrue(result.is_error)
        self.assertEqual(len(self.executor._active), 0)

    async def test_invalid_arguments_fail_closed(self):
        for args in [{}, {'code': ''}, {'code': 4}, {'code': 'print(1)', 'task_id': 'forged'}]:
            self.assertTrue((await self.executor.call({**args})).is_error)

    async def test_real_native_execution_rpc_and_env_scrubbing(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = Path(root) / 'fixture.txt'
            fixture.write_text('unique-native-fixture')
            code = f'''from hermes_tools import read_file, search_files, terminal
import json, os
r = read_file({str(fixture)!r})
s = search_files('unique-native-fixture', path={root!r})
t = terminal('printf native-rpc-ok')
print(json.dumps({{'r':r,'s':s,'t':t,'secret':os.getenv('MIGRATION_SECRET_TOKEN')}}))'''
            from unittest.mock import patch
            with patch.dict(os.environ, {'MIGRATION_SECRET_TOKEN': 'never-leak-this-test-value'}):
                result, payload = await self.run_code(code)
            self.assertFalse(result.is_error, payload)
            self.assertEqual(payload['status'], 'success', payload)
            self.assertEqual(payload['tool_calls_made'], 3)
            data = json.loads(payload['output'])
            self.assertIn('unique-native-fixture', data['r']['content'])
            self.assertEqual(data['t']['exit_code'], 0)
            self.assertIn('native-rpc-ok', data['t']['output'])
            self.assertIsNone(data['secret'])

    async def test_json_escaped_code_within_utf8_budget(self):
        result, payload = await self.run_code('#' + ('\\' * 40000) + "\nprint('escaped-ok')")
        self.assertFalse(result.is_error, payload)
        self.assertEqual(payload['output'].strip(), 'escaped-ok')

    async def test_code_byte_budget_rejects_before_spawn(self):
        result = await self.executor.call({'code': '#' + 'ş' * 40000})
        self.assertTrue(result.is_error)
        self.assertFalse(self.executor._active)

    async def test_calls_do_not_share_interpreter_state(self):
        first, _ = await self.run_code("private_value=42; print('set')")
        self.assertFalse(first.is_error)
        second, payload = await self.run_code("print('private_value' in globals())")
        self.assertFalse(second.is_error, payload)
        self.assertEqual(payload['output'].strip(), 'False')

    async def test_nested_disallowed_tool_is_unavailable(self):
        result, payload = await self.run_code('from hermes_tools import delegate_task')
        self.assertTrue(result.is_error, payload)

    async def test_output_is_bounded(self):
        result = await self.executor.call({'code': "print('x'*200000)"})
        self.assertLessEqual(sum(len(x.text) for x in result.content), GatewayConfig().max_text_chars)
        json.loads(result.content[0].text)
        bounded = _result({'output': 'x' * 10000}, limit=220)
        self.assertLessEqual(len(bounded.content[0].text), 220)
        payload = json.loads(bounded.content[0].text)
        self.assertTrue(payload['truncated'])
        self.assertIn('preview', payload)

    async def test_timeout_kills_native_tree(self):
        executor = NativeExecution(replace(GatewayConfig(), execution_timeout_seconds=1.0))
        try:
            result = await executor.call({'code': 'import time; time.sleep(60)'})
            self.assertTrue(result.is_error)
            self.assertIn('timeout', result.content[0].text.lower())
            self.assertEqual(len(executor._active), 0)
        finally:
            await executor.close()

    async def test_shutdown_waits_for_pending_launch_and_prevents_execution(self):
        from unittest.mock import patch
        started, release = asyncio.Event(), asyncio.Event()
        original = asyncio.create_subprocess_exec
        async def delayed(*args, **kwargs):
            started.set()
            await release.wait()
            return await original(*args, **kwargs)
        with patch('hermes_mcp_gateway.execution.asyncio.create_subprocess_exec', delayed):
            request = asyncio.create_task(self.executor.call({'code': "print('must not execute')"}))
            closing = None
            try:
                await asyncio.wait_for(started.wait(), 3)
                closing = asyncio.create_task(self.executor.close())
                await asyncio.sleep(.05)
                self.assertFalse(closing.done(), 'shutdown returned while worker launch was still pending')
                release.set()
                await asyncio.wait_for(closing, 10)
                self.assertTrue(request.cancelled())
                self.assertFalse(self.executor._active)
            finally:
                release.set()
                await asyncio.gather(request, *([closing] if closing else []), return_exceptions=True)

    async def test_normal_completion_reaps_orphaned_grandchild(self):
        with tempfile.TemporaryDirectory() as root:
            pidfile = Path(root) / 'orphan.pid'
            child = "import subprocess; from pathlib import Path; p=subprocess.Popen(['sleep','60'],start_new_session=True); Path(" + repr(str(pidfile)) + ").write_text(str(p.pid))"
            code = f"import subprocess,sys; subprocess.run([sys.executable, '-c', {child!r}], check=True); print('done')"
            try:
                result, payload = await self.run_code(code)
                self.assertFalse(result.is_error, payload)
                pid = int(pidfile.read_text())
                self.assertFalse(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE, pid)
            finally:
                if pidfile.exists():
                    try:
                        psutil.Process(int(pidfile.read_text())).kill()
                    except psutil.NoSuchProcess:
                        pass

    async def test_cancel_kills_detached_descendant(self):
        with tempfile.TemporaryDirectory() as root:
            pidfile = Path(root) / 'pids.json'
            code = f'''import os,subprocess,time,json
from pathlib import Path
p=subprocess.Popen(['sleep','60'],start_new_session=True)
Path({str(pidfile)!r}).write_text(json.dumps([os.getpid(),p.pid]))
time.sleep(60)'''
            task = asyncio.create_task(self.executor.call({'code': code}))
            try:
                for _ in range(300):
                    if pidfile.exists() or task.done():
                        break
                    await asyncio.sleep(.05)
                self.assertTrue(pidfile.exists(), task.result() if task.done() else 'worker did not start')
                pids = json.loads(pidfile.read_text())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                for pid in pids:
                    self.assertFalse(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE, pid)
                self.assertEqual(len(self.executor._active), 0)
            finally:
                task.cancel()
                if pidfile.exists():
                    for pid in json.loads(pidfile.read_text()):
                        try:
                            psutil.Process(pid).kill()
                        except psutil.NoSuchProcess:
                            pass
