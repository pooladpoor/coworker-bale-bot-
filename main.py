"""Entry point.

Architecture:

    polling loop (1 coroutine)  --enqueue-->  asyncio.Queue  --consumed by-->  N workers

The polling loop's only job is to call `getUpdates` and enqueue work.
It never calls an AI model or uploads a file itself, so a slow AI
response or a slow upload can never delay the next poll -- new updates
keep being picked up immediately, and many users are served
concurrently by the worker pool.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from bale_client import BaleClient
from comet_client import CometAPIClient
from config import Settings
from models import IncomingJob
from queue_manager import JobQueue
from utils import setup_logging
from worker import Worker

logger = logging.getLogger(__name__)


async def polling_loop(bale: BaleClient, queue: JobQueue, settings: Settings) -> None:
    """Non-blocking long-poll loop. Only extracts photo messages and
    enqueues them; text messages get an immediate reply inline since
    that's cheap and doesn't need a worker."""
    last_update_id = 0
    logger.info("Polling loop started")

    while True:
        try:
            updates = await bale.get_updates(
                offset=last_update_id + 1, timeout=settings.polling_timeout
            )
        except Exception:  # noqa: BLE001
            logger.exception("Polling failed, backing off")
            await asyncio.sleep(settings.polling_error_backoff)
            continue

        if not updates:
            continue

        logger.debug("Received %d update(s)", len(updates))

        for update in updates:
            last_update_id = update["update_id"]
            message = update.get("message")
            if message is None:
                continue

            chat_id = message.get("chat", {}).get("id")
            if not chat_id:
                continue

            if "photo" in message:
                file_id = message["photo"][-1]["file_id"]
                job = IncomingJob(chat_id=chat_id, file_id=file_id)
                await queue.put(job)
                logger.info("Enqueued job for chat %s (queue size=%d)", chat_id, queue.qsize())

            elif "text" in message:
                # Cheap, immediate reply -- fired off without blocking the
                # loop on its completion.
                asyncio.create_task(
                    bale.send_message(
                        chat_id, "سلام! 👋 لطفاً از سوال امتحانی خود یک عکس واضح بفرستید."
                    )
                )


async def run() -> None:
    settings = Settings.load()
    setup_logging(settings.log_level)

    logger.info("Starting bot | workers=%d queue_max=%d", settings.worker_count,
                settings.queue_max_size)

    connector = aiohttp.TCPConnector(
        limit=settings.connector_limit,
        limit_per_host=settings.connector_limit_per_host,
        keepalive_timeout=30,
    )
    timeout = aiohttp.ClientTimeout(
        total=settings.http_timeout_total, connect=settings.http_timeout_connect
    )

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        bale = BaleClient(session, settings)
        comet = CometAPIClient(session, settings)
        queue = JobQueue(settings.queue_max_size)

        workers = [
            Worker(worker_id=i, queue=queue, bale=bale, comet=comet, settings=settings)
            for i in range(settings.worker_count)
        ]

        async with asyncio.TaskGroup() as tg:
            tg.create_task(polling_loop(bale, queue, settings))
            for worker in workers:
                tg.create_task(worker.run())


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")


if __name__ == "__main__":
    main()
