"""Track existing native worker threads without pretending cancellation kills them."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


class NativeCalls:
    """Use asyncio's existing executor; retain actual work after its waiter leaves.

    No new executor, scheduler or command runner is created. Shielding keeps a
    queued/native call visible to deployment drain until its thread really ends.
    """
    def __init__(self):
        self._tasks: set[asyncio.Task] = set()
        self._owner_lease: ContextVar[
            tuple[Callable[[], None], Callable[[], None]] | None
        ] = ContextVar(f"native_owner_lease_{id(self)}", default=None)

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    async def wait_idle(self) -> None:
        """Wait for tracked worker tasks to finish; never cancel native work."""
        while self._tasks:
            tasks = tuple(self._tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    def _finished(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            # A disconnected waiter may no longer retrieve the exception. Never
            # log native payloads or diagnostic text from this bookkeeping hook.
            task.exception()

    @contextmanager
    def owner_lease(
        self, retain: Callable[[], None], release: Callable[[], None]
    ) -> Iterator[None]:
        """Keep a session owner alive for native work started in this context."""
        token = self._owner_lease.set((retain, release))
        try:
            yield
        finally:
            self._owner_lease.reset(token)

    async def run(self, function: Callable[..., Any], *args, **kwargs) -> Any:
        owner_lease = self._owner_lease.get()
        if owner_lease is not None:
            owner_lease[0]()
        try:
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        except BaseException:
            if owner_lease is not None:
                owner_lease[1]()
            raise
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        if owner_lease is not None:
            task.add_done_callback(lambda _task: owner_lease[1]())
        return await asyncio.shield(task)
