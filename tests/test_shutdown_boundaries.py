"""Graceful shutdown must not abandon claimed owner cleanup on cancellation."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from hermes_mcp_gateway.server import Gateway


class ShutdownBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_close_keeps_all_owner_cleanup_tracked_until_finished(self):
        gateway = Gateway()
        gateway.execution.close = AsyncMock()
        gateway.hermes.close = AsyncMock()
        gateway.graph.close = AsyncMock()
        gateway.lsp.close = Mock()
        records = [gateway.session_capabilities.issue()[1] for _ in range(2)]
        started = []
        finished = []
        all_started, release = asyncio.Event(), asyncio.Event()

        async def cleanup(task_id: str):
            started.append(task_id)
            if len(started) == len(records):
                all_started.set()
            await release.wait()
            finished.append(task_id)

        gateway._cleanup_session_owner = cleanup
        closing = asyncio.create_task(gateway.close())
        try:
            await asyncio.wait_for(all_started.wait(), timeout=1)
            closing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            self.assertEqual(finished, [])
            self.assertEqual(len(gateway._session_cleanup_tasks), 2)
            for record in records:
                self.assertIn(record.task_id, gateway.session_capabilities._by_owner)
                self.assertTrue(record.cleanup_started)
            release.set()
            await asyncio.wait_for(gateway.close(), timeout=2)
            self.assertCountEqual(started, finished)
            self.assertEqual(len(started), 2)
            self.assertEqual(gateway.session_capabilities._by_owner, {})
            self.assertEqual(gateway._session_cleanup_tasks, set())
        finally:
            release.set()
            await asyncio.gather(closing, return_exceptions=True)
            await gateway.close()
