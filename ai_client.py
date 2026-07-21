"""Async client for the three AI models.

Each model (Gemini, Claude, GPT) is called against its OWN configured
base URL, using its OWN API key -- so each one can live on a completely
different platform (CometAPI, OpenRouter, a direct provider API, a
self-hosted OpenAI-compatible gateway, ...). Nothing here is tied to any
single vendor.

All requests share a single `aiohttp.ClientSession`, giving us
connection pooling + keep-alive regardless of how many different hosts
the three models end up pointing at.

Endpoint conventions assumed (both widely supported as the "OpenAI
compatible" standard across most platforms):
  - Gemini & Claude: POST {base_url}/chat/completions
  - GPT:              POST {base_url}/responses
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import aiohttp

from config import SYSTEM_PROMPT, ProviderConfig, Settings
from models import AIResult
from utils import retry_async

logger = logging.getLogger(__name__)

# Errors worth retrying: network hiccups, timeouts, and 5xx from the server.
_RETRIABLE_EXCEPTIONS = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    asyncio.TimeoutError,
)


class AIClient:
    """Thin async wrapper around the per-model provider endpoints, each
    independently configured with its own base URL and API key."""

    def __init__(self, session: aiohttp.ClientSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._ai_timeout = aiohttp.ClientTimeout(total=settings.ai_request_timeout)

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

    async def _post_json(self, provider: ProviderConfig, path: str, payload: dict) -> dict:
        url = f"{provider.base_url}{path}"
        headers = self._build_headers(provider.api_key)

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
            op_name=f"POST {url}",
        )

    async def ask_gemini(self, base64_image: str) -> AIResult:
        """Call Gemini against its configured platform/base URL, using
        Gemini's own API key."""
        start = time.monotonic()
        provider = self._settings.gemini
        payload = {
            "model": provider.model,
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
            data = await self._post_json(provider, "/chat/completions", payload)
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
                text=f"خطا در دریافت پاسخ از جمنای: {exc}",
                elapsed_seconds=time.monotonic() - start,
                error=True,
            )

    async def ask_claude(self, base64_image: str) -> AIResult:
        """Call Claude against its configured platform/base URL (same
        OpenAI-compatible request shape used for Gemini), using Claude's
        own API key."""
        start = time.monotonic()
        provider = self._settings.claude
        payload = {
            "model": provider.model,
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
            data = await self._post_json(provider, "/chat/completions", payload)
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
                text=f"خطا در دریافت پاسخ از کلود: {exc}",
                elapsed_seconds=time.monotonic() - start,
                error=True,
            )

    async def ask_chatgpt(self, base64_image: str) -> AIResult:
        """Call GPT against its configured platform/base URL (OpenAI-
        compatible `responses` endpoint), using GPT's own API key."""
        start = time.monotonic()
        provider = self._settings.gpt
        payload = {
            "model": provider.model,
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
            data = await self._post_json(provider, "/responses", payload)
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
                text=f"خطا در دریافت پاسخ از چت‌جی‌پتی: {exc}",
                elapsed_seconds=time.monotonic() - start,
                error=True,
            )

    @staticmethod
    def _extract_output_text(data: dict) -> str:
        """Mirror the convenience `response.output_text` behaviour of the
        official OpenAI SDK, but from a raw JSON payload."""
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
        each against its own platform with its own API key.
        Returns (gemini, claude, gpt)."""
        base64_image = self.encode_image(image_bytes)
        async with asyncio.TaskGroup() as tg:
            gemini_task = tg.create_task(self.ask_gemini(base64_image))
            claude_task = tg.create_task(self.ask_claude(base64_image))
            gpt_task = tg.create_task(self.ask_chatgpt(base64_image))
        return gemini_task.result(), claude_task.result(), gpt_task.result()
