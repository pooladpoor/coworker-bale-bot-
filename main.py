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
from ai_client import AIClient
from config import ACCESS_DENIED_MESSAGE, Settings
from models import IncomingJob
from queue_manager import JobQueue
from utils import message_mentions_bot, setup_logging
from worker import Worker

logger = logging.getLogger(__name__)


def _is_authorized(user_id: int | None, settings: Settings) -> bool:
    """Only the numeric Bale user IDs listed in ALLOWED_USER_IDS may use
    the bot -- checked against the message SENDER's id, not the chat id,
    so this works correctly in both private chats and groups."""
    if not settings.allowed_user_ids:
        # No allowlist configured -> bot is open to everyone (unchanged
        # default behaviour).
        return True
    return user_id is not None and user_id in settings.allowed_user_ids


async def polling_loop(
    bale: BaleClient, queue: JobQueue, settings: Settings, bot_username: str | None
) -> None:
    """Non-blocking long-poll loop. Only extracts photo messages and
    enqueues them; text messages get an immediate reply inline since
    that's cheap and doesn't need a worker.

    In group/supergroup chats, a photo is only enqueued if the bot is
    @-mentioned in its caption (when `GROUP_REQUIRE_MENTION` is on).
    Private chats always respond, exactly like before.

    If `ALLOWED_USER_IDS` is configured, any message directed at the bot
    from a sender NOT on that list gets a polite rejection instead of
    being processed -- in both private chats and groups.
    """
    last_update_id = 0
    logger.info("Polling loop started (bot_username=%s)", bot_username)

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

            chat = message.get("chat", {})
            chat_id = chat.get("id")
            if not chat_id:
                continue

            chat_type = chat.get("type", "private")
            is_group = chat_type in ("group", "supergroup")
            sender_id = message.get("from", {}).get("id")

            if "photo" in message:
                caption = message.get("caption")
                caption_entities = message.get("caption_entities")

                if is_group and settings.group_require_mention:
                    if not message_mentions_bot(caption, caption_entities, bot_username):
                        logger.debug(
                            "Chat %s (%s): photo without bot mention, ignoring", chat_id, chat_type
                        )
                        continue

                if not _is_authorized(sender_id, settings):
                    logger.info(
                        "Rejected photo from unauthorized user %s in chat %s (%s)",
                        sender_id, chat_id, chat_type,
                    )
                    asyncio.create_task(
                        bale.send_message(
                            chat_id,
                            ACCESS_DENIED_MESSAGE,
                            reply_to_message_id=message.get("message_id") if is_group else None,
                        )
                    )
                    continue

                file_id = message["photo"][-1]["file_id"]
                job = IncomingJob(
                    chat_id=chat_id,
                    file_id=file_id,
                    message_id=message.get("message_id"),
                    chat_type=chat_type,
                )
                await queue.put(job)
                logger.info(
                    "Enqueued job for chat %s (%s, queue size=%d)",
                    chat_id, chat_type, queue.qsize(),
                )

            elif "text" in message:
                text = message.get("text")
                entities = message.get("entities")

                if is_group:
                    # Avoid spamming groups with the greeting on every
                    # unrelated message -- only reply if directly mentioned.
                    if not message_mentions_bot(text, entities, bot_username):
                        continue

                if not _is_authorized(sender_id, settings):
                    logger.info(
                        "Rejected text from unauthorized user %s in chat %s (%s)",
                        sender_id, chat_id, chat_type,
                    )
                    asyncio.create_task(
                        bale.send_message(
                            chat_id,
                            ACCESS_DENIED_MESSAGE,
                            reply_to_message_id=message.get("message_id") if is_group else None,
                        )
                    )
                    continue

                # Cheap, immediate reply -- fired off without blocking the
                # loop on its completion.
                asyncio.create_task(
                    bale.send_message(
                        chat_id,
                        "سلام! 👋 لطفاً از سوال امتحانی خود یک عکس واضح بفرستید"
                        + (" و من رو تگ کنید." if is_group else "."),
                        reply_to_message_id=message.get("message_id") if is_group else None,
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
        ai_client = AIClient(session, settings)
        queue = JobQueue(settings.queue_max_size)

        bot_username = settings.bot_username
        if bot_username is None:
            try:
                me = await bale.get_me()
                bot_username = me.get("username")
                logger.info("Resolved bot identity: @%s (id=%s)", bot_username, me.get("id"))
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Could not resolve bot username via getMe(); group mention "
                    "detection will be disabled until BOT_USERNAME is set in .env"
                )

        workers = [
            Worker(worker_id=i, queue=queue, bale=bale, ai_client=ai_client, settings=settings)
            for i in range(settings.worker_count)
        ]

        async with asyncio.TaskGroup() as tg:
            tg.create_task(polling_loop(bale, queue, settings, bot_username))
            for worker in workers:
                tg.create_task(worker.run())


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")


if __name__ == "__main__":
    main()
