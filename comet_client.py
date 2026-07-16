"""Async client for CometAPI.

Both Gemini (via the OpenAI-compatible `chat/completions` endpoint) and
GPT (via the `responses` endpoint) are called through a single shared
`aiohttp.ClientSession`, giving us connection pooling + keep-alive instead
of opening a new connection per request.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import aiohttp

from config import SYSTEM_PROMPT, Settings
from models import AIResult
from utils import retry_async

logger = logging.getLogger(__name__)

# Errors worth retrying: network hiccups, timeouts, and 5xx from the server.
_RETRIABLE_EXCEPTIONS = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    asyncio.TimeoutError,
)


class CometAPIClient:
    """Thin async wrapper around the two CometAPI endpoints we use."""

    def __init__(self, session: aiohttp.ClientSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._headers = {
            "Authorization": f"Bearer {settings.comet_api_key}",
            "Content-Type": "application/json",
        }
        self._ai_timeout = aiohttp.ClientTimeout(total=settings.ai_request_timeout)

    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        """Base64-encode an image exactly once; callers should cache and
        reuse the resulting string for both AI requests."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self._settings.comet_base_url}{path}"

        async def _do_request() -> dict:
            async with self._session.post(
                url, json=payload, headers=self._headers, timeout=self._ai_timeout
            ) as resp:
                if resp.status >= 500:
                    # Treat server errors as retriable.
                    text = await resp.text()
                    raise aiohttp.ClientConnectionError(
                        f"server error {resp.status}: {text[:200]}"
                    )
                resp.raise_for_status()
                return await resp.json()

        return await retry_async(
            _do_request,
            max_retries=self._settings.max_retries,
            backoff_base=self._settings.retry_backoff_base,
            retriable_exceptions=_RETRIABLE_EXCEPTIONS,
            logger=logger,
            op_name=f"POST {path}",
        )

    async def ask_gemini(self, base64_image: str) -> AIResult:
        """Call Gemini through CometAPI's chat/completions endpoint."""
        start = time.monotonic()
        payload = {
            "model": self._settings.gemini_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        }
        try:
            data = await self._post_json("/chat/completions", payload)
            text = data["choices"][0]["message"]["content"]
            return AIResult(
                model_name="gemini",
                text=text,
                elapsed_seconds=time.monotonic() - start,
            )
        except Exception as exc:  # noqa: BLE001 - we want to isolate failures per model
            logger.exception("Gemini request failed")
            return AIResult(
                model_name="gemini",
                text=f"خطا در دریافت پاسخ از جمنای (CometAPI): {exc}",
                elapsed_seconds=time.monotonic() - start,
                error=True,
            )

    async def ask_chatgpt(self, base64_image: str) -> AIResult:
        """Call GPT through CometAPI's OpenAI-compatible responses endpoint."""
        start = time.monotonic()
        payload = {
            "model": self._settings.gpt_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        }
        try:
            data = await self._post_json("/responses", payload)
            text = self._extract_output_text(data)
            return AIResult(
                model_name="gpt",
                text=text,
                elapsed_seconds=time.monotonic() - start,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("GPT request failed")
            return AIResult(
                model_name="gpt",
                text=f"خطا در دریافت پاسخ از چت‌جی‌پتی (CometAPI): {exc}",
                elapsed_seconds=time.monotonic() - start,
                error=True,
            )

    @staticmethod
    def _extract_output_text(data: dict) -> str:
        """Mirror the convenience `response.output_text` behaviour of the
        official SDK, but from a raw JSON payload."""
        if "output_text" in data:
            return data["output_text"]

        chunks: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    chunks.append(content.get("text", ""))
        return "".join(chunks)

    async def ask_both(self, image_bytes: bytes) -> tuple[AIResult, AIResult]:
        """Encode the image once and fire both requests concurrently."""
        base64_image = self.encode_image(image_bytes)
        async with asyncio.TaskGroup() as tg:
            gemini_task = tg.create_task(self.ask_gemini(base64_image))
            gpt_task = tg.create_task(self.ask_chatgpt(base64_image))
        return gemini_task.result(), gpt_task.result()
