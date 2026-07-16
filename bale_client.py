"""Async client for the Bale Bot API.

All calls share the process-wide `aiohttp.ClientSession` for connection
pooling / keep-alive, and retry transient failures with exponential
backoff.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time

import aiohttp

from config import Settings
from utils import retry_async

logger = logging.getLogger(__name__)

_RETRIABLE_EXCEPTIONS = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    asyncio.TimeoutError,
)


class BaleAPIError(Exception):
    """Raised when Bale responds with ok=False or an unexpected shape."""


class BaleClient:
    def __init__(self, session: aiohttp.ClientSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings

    async def get_updates(self, offset: int, timeout: int) -> list[dict]:
        """Long-poll for new updates. Non-blocking with respect to the
        rest of the event loop -- other coroutines (workers) keep running
        while we await this."""
        params = {"offset": offset, "timeout": timeout}
        req_timeout = aiohttp.ClientTimeout(total=timeout + self._settings.http_timeout_connect)

        async def _do_request() -> list[dict]:
            async with self._session.get(
                f"{self._settings.bale_api_url}getUpdates",
                params=params,
                timeout=req_timeout,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if not data.get("ok"):
                    raise BaleAPIError(f"getUpdates returned ok=False: {data}")
                return data["result"]

        return await retry_async(
            _do_request,
            max_retries=self._settings.max_retries,
            backoff_base=self._settings.retry_backoff_base,
            retriable_exceptions=_RETRIABLE_EXCEPTIONS,
            logger=logger,
            op_name="getUpdates",
        )

    async def send_message(self, chat_id: int, text: str) -> None:
        payload = {"chat_id": chat_id, "text": text}

        async def _do_request() -> None:
            async with self._session.post(
                f"{self._settings.bale_api_url}sendMessage",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._settings.http_timeout_total),
            ) as resp:
                body = await resp.text()
                logger.debug("sendMessage status=%s body=%s", resp.status, body[:300])
                resp.raise_for_status()

        await retry_async(
            _do_request,
            max_retries=self._settings.max_retries,
            backoff_base=self._settings.retry_backoff_base,
            retriable_exceptions=_RETRIABLE_EXCEPTIONS,
            logger=logger,
            op_name="sendMessage",
        )

    async def get_file_path(self, file_id: str) -> str:
        async def _do_request() -> str:
            async with self._session.get(
                f"{self._settings.bale_api_url}getFile",
                params={"file_id": file_id},
                timeout=aiohttp.ClientTimeout(total=self._settings.http_timeout_total),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if not data.get("ok"):
                    raise BaleAPIError(f"getFile returned ok=False: {data}")
                return data["result"]["file_path"]

        return await retry_async(
            _do_request,
            max_retries=self._settings.max_retries,
            backoff_base=self._settings.retry_backoff_base,
            retriable_exceptions=_RETRIABLE_EXCEPTIONS,
            logger=logger,
            op_name="getFile",
        )

    async def download_file(self, file_path: str) -> bytes:
        url = f"{self._settings.bale_file_url}{file_path}"

        async def _do_request() -> bytes:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=self._settings.http_timeout_total)
            ) as resp:
                resp.raise_for_status()
                return await resp.read()

        return await retry_async(
            _do_request,
            max_retries=self._settings.max_retries,
            backoff_base=self._settings.retry_backoff_base,
            retriable_exceptions=_RETRIABLE_EXCEPTIONS,
            logger=logger,
            op_name="downloadFile",
        )

    async def send_document(
        self,
        chat_id: int,
        file_buffer: io.BytesIO,
        filename: str,
        caption: str,
    ) -> None:
        """Send a file directly from an in-memory buffer -- no temp file
        touches disk."""
        file_buffer.seek(0)

        async def _do_request() -> None:
            file_buffer.seek(0)
            form = aiohttp.FormData()
            form.add_field("chat_id", str(chat_id))
            form.add_field("caption", caption)
            form.add_field(
                "document",
                file_buffer,
                filename=filename,
                content_type="text/html",
            )
            async with self._session.post(
                f"{self._settings.bale_api_url}sendDocument",
                data=form,
                timeout=aiohttp.ClientTimeout(total=self._settings.http_timeout_total),
            ) as resp:
                body = await resp.text()
                logger.debug("sendDocument status=%s body=%s", resp.status, body[:300])
                resp.raise_for_status()

        await retry_async(
            _do_request,
            max_retries=self._settings.max_retries,
            backoff_base=self._settings.retry_backoff_base,
            retriable_exceptions=_RETRIABLE_EXCEPTIONS,
            logger=logger,
            op_name="sendDocument",
        )
