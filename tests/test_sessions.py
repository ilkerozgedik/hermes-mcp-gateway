import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from hermes_mcp_gateway.server import Gateway
from hermes_mcp_gateway.sessions import SessionCapabilityStore


class SessionCapabilityStoreTests(unittest.TestCase):
    def test_capabilities_are_random_owner_scoped_and_bounded(self):
        store = SessionCapabilityStore(max_active=1)
        first_token, first = store.issue()
        self.assertEqual(len(first_token), 43)
        self.assertIsNone(store.acquire("not-a-token"))
        lease = store.acquire(first_token)
        self.assertIsNotNone(lease)
        assert lease is not None
        self.assertIs(lease, first)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            store.issue()

        self.assertIsNone(store.revoke(first))
        self.assertIsNone(store.acquire(first_token))
        cleanup = store.release(lease)
        self.assertIs(cleanup, first)
        assert cleanup is not None
        self.assertIn(first.task_id, store._by_owner)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            store.issue()
        store.cleanup_complete(cleanup)
        second_token, second = store.issue()
        self.assertNotEqual(first_token, second_token)
        self.assertNotEqual(first.task_id, second.task_id)

    def test_invalid_token_types_alphabet_and_unicode_fail_closed(self):
        store = SessionCapabilityStore()
        for token in (None, 7, {}, "", "x" * 10000, "\ud800", "=" * 43):
            with self.subTest(token_type=type(token).__name__):
                self.assertIsNone(store.acquire(token))

    def test_expiration_uses_monotonic_clock_and_defers_cleanup_for_leases(self):
        now = [10.0]
        store = SessionCapabilityStore(ttl_seconds=5, clock=lambda: now[0])
        token, record = store.issue()
        lease = store.acquire(token)
        assert lease is not None
        now[0] = 14.999
        self.assertEqual(store.expire_due(), [])
        now[0] = 15.0
        self.assertEqual(store.expire_due(), [])
        self.assertIsNone(store.acquire(token))
        cleanup = store.expire_due()
        self.assertEqual(cleanup, [])  # already revoked while its lease is active
        cleanup_record = store.release(lease)
        self.assertIs(cleanup_record, record)
        assert cleanup_record is not None
        store.cleanup_complete(cleanup_record)
        self.assertEqual(store.expire_due(), [])

    def test_retained_native_reference_keeps_revoked_owner_until_cleanup(self):
        store = SessionCapabilityStore(max_active=1)
        token, record = store.issue()
        request_lease = store.acquire(token)
        self.assertIs(request_lease, record)
        assert request_lease is not None
        store.retain(record)

        self.assertIsNone(store.revoke(record))
        self.assertIsNone(store.release(request_lease))
        cleanup = store.release(record)
        self.assertIs(cleanup, record)
        self.assertEqual(record.active_calls, 0)
        self.assertIn(record.task_id, store._by_owner)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            store.issue()

        store.cleanup_complete(record)
        self.assertNotIn(record.task_id, store._by_owner)
        store.issue()


class NativeSessionCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_before_acquire_does_not_leak_a_lease(self):
        gateway = Gateway()
        token, record = gateway.session_capabilities.issue()
        started = asyncio.Event()

        async def pending_cleanup(_records):
            started.set()
            await asyncio.Future()

        gateway._cleanup_session_records = AsyncMock(side_effect=pending_cleanup)
        task = asyncio.create_task(gateway.acquire_session_token(token))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(record.active_calls, 0)
        await gateway.execution.close()

    async def test_cleanup_scopes_process_kill_and_drops_only_local_browser_binding(self):
        gateway = Gateway()
        owner = "gateway-session-test-owner"
        with (
            patch("tools.process_registry.process_registry.kill_all") as kill_all,
            patch("tools.terminal_tool_lifecycle.cleanup_vm") as cleanup_terminal,
            patch("tools.browser_camofox._drop_session") as drop_session,
            patch("tools.browser_camofox._delete") as delete_remote_session,
        ):
            await gateway._cleanup_session_owner(owner)

        kill_all.assert_called_once_with(
            task_id=owner,
            source="gateway_session_close",
            consume_output=True,
        )
        cleanup_terminal.assert_called_once_with(owner)
        drop_session.assert_called_once_with(owner)
        delete_remote_session.assert_not_called()
        await gateway.execution.close()
