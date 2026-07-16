"""A thin, typed wrapper around `asyncio.Queue` for incoming jobs.

Kept as its own module so the queueing policy (max size, backpressure
behaviour) is defined in exactly one place.
"""

from __future__ import annotations

import asyncio
import logging

from models import IncomingJob

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, max_size: int) -> None:
        self._queue: asyncio.Queue[IncomingJob] = asyncio.Queue(maxsize=max_size)

    async def put(self, job: IncomingJob) -> None:
        """Enqueue a job. If the queue is full, this will wait rather than
        drop work -- the polling loop stays responsive because this await
        yields control back to the event loop, it does not block other
        coroutines."""
        if self._queue.full():
            logger.warning(
                "Queue is full (size=%d) -- backpressure engaged, polling loop will wait",
                self._queue.qsize(),
            )
        await self._queue.put(job)

    async def get(self) -> IncomingJob:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()
