"""Async client for CometAPI.

Gemini and Claude (via the OpenAI-compatible `chat/completions` endpoint)
and GPT (via the `responses` endpoint) are called through a single shared
`aiohttp.ClientSession`, giving us connection pooling + keep-alive instead
of opening a new connection per request.

Each model uses its OWN CometAPI key. This means usage/cost for Gemini,
Claude, and GPT show up as three separate entries on the CometAPI
dashboard instead of one combined total, making it easy to monitor
consumption per model independently.
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
    """Thin async wrapper around the CometAPI endpoints we use, with a
    distinct API key per model."""

    def __init__(self, session: aiohttp.ClientSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._ai_timeout = aiohttp.ClientTimeout(total=settings.ai_request_timeout)

        # One header set per model, each carrying its own API key.
        self._gemini_headers = self._build_headers(settings.comet_gemini_key)
        self._claude_headers = self._build_headers(settings.comet_claude_key)
        self._gpt_headers = self._build_headers(settings.comet_gpt_key)

    @staticmethod
    def _build_headers(api_key: str) -> dict:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        """Base64-encode an image exactly once; callers should cache and
        reuse the resulting string for all three AI requests."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def _post_json(self, path: str, payload: dict, headers: dict) -> dict:
        url = f"{self._settings.comet_base_url}{path}"

        async def _do_request() -> dict:
            async with self._session.post(
                url, json=payload, headers=headers, timeout=self._ai_timeout
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
        """Call Gemini through CometAPI's chat/completions endpoint,
        using Gemini's own API key."""
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
            data = await self._post_json(
                "/chat/completions", payload, self._gemini_headers
            )
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

    async def ask_claude(self, base64_image: str) -> AIResult:
        """Call Claude through CometAPI's chat/completions endpoint (same
        OpenAI-compatible shape used for Gemini), using Claude's own API
        key."""
        start = time.monotonic()
        payload = {
            "model": self._settings.claude_model,
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
            data = await self._post_json(
                "/chat/completions", payload, self._claude_headers
            )
            text = data["choices"][0]["message"]["content"]
            return AIResult(
                model_name="claude",
                text=text,
                elapsed_seconds=time.monotonic() - start,
            )
        except Exception as exc:  # noqa: BLE001 - we want to isolate failures per model
            logger.exception("Claude request failed")
            return AIResult(
                model_name="claude",
                text=f"خطا در دریافت پاسخ از کلود (CometAPI): {exc}",
                elapsed_seconds=time.monotonic() - start,
                error=True,
            )

    async def ask_chatgpt(self, base64_image: str) -> AIResult:
        """Call GPT through CometAPI's OpenAI-compatible responses
        endpoint, using GPT's own API key."""
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
            data = await self._post_json("/responses", payload, self._gpt_headers)
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

    async def ask_all(self, image_bytes: bytes) -> tuple[AIResult, AIResult, AIResult]:
        """Encode the image once and fire all three requests concurrently,
        each authenticated with its own API key.
        Returns (gemini, claude, gpt)."""
        base64_image = self.encode_image(image_bytes)
        async with asyncio.TaskGroup() as tg:
            gemini_task = tg.create_task(self.ask_gemini(base64_image))
            claude_task = tg.create_task(self.ask_claude(base64_image))
            gpt_task = tg.create_task(self.ask_chatgpt(base64_image))
        return gemini_task.result(), claude_task.result(), gpt_task.result()
