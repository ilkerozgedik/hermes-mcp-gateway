"""Track existing native worker threads without pretending cancellation kills them."""
from __future__ import annotations

import asyncio
from typing import Any, Callable


class NativeCalls:
    """Use asyncio's existing executor; retain actual work after its waiter leaves.

    No new executor, scheduler or command runner is created. Shielding keeps a
    queued/native call visible to deployment drain until its thread really ends.
    """
    def __init__(self):
        self._tasks: set[asyncio.Task] = set()

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    def _finished(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            # A disconnected waiter may no longer retrieve the exception. Never
            # log native payloads or diagnostic text from this bookkeeping hook.
            task.exception()

    async def run(self, function: Callable[..., Any], *args, **kwargs) -> Any:
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return await asyncio.shield(task)
