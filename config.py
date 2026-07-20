"""
Centralized configuration for the bot.

All tunables are read from environment variables (via `.env`) so the
deployment can be adjusted without touching code. Sensible defaults are
provided for anything non-critical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True, slots=True)
class Settings:
    # --- Secrets / endpoints -------------------------------------------------
    bale_token: str
    comet_base_url: str = "https://api.cometapi.com/v1"

    # Separate API keys per model. Using three distinct CometAPI keys/apps
    # lets you track usage and cost independently for each model on the
    # CometAPI dashboard, instead of one combined total.
    comet_gemini_key: str = ""
    comet_claude_key: str = ""
    comet_gpt_key: str = ""

    # --- Models ---------------------------------------------------------------
    gpt_model: str = "gpt-5.6-sol"
    gemini_model: str = "gemini-3-flash"
    claude_model: str = "claude-sonnet-4-6"

    # --- Concurrency / queue ----------------------------------------------
    worker_count: int = 8
    queue_max_size: int = 1000

    # --- HTTP behaviour ---------------------------------------------------
    http_timeout_total: float = 60.0
    http_timeout_connect: float = 10.0
    ai_request_timeout: float = 90.0
    max_retries: int = 3
    retry_backoff_base: float = 1.5  # seconds, exponential base
    connector_limit: int = 100
    connector_limit_per_host: int = 30

    # --- Polling ------------------------------------------------------------
    polling_timeout: int = 10
    polling_error_backoff: float = 3.0

    # --- Misc -----------------------------------------------------------------
    log_level: str = "INFO"
    min_valid_image_bytes: int = 1000

    # --- Group behaviour ---------------------------------------------------
    # In group/supergroup chats, only respond to a photo if the bot is
    # @-mentioned in its caption. Private chats always respond.
    group_require_mention: bool = True
    # Optional override; if unset, the username is fetched once at startup
    # via getMe() and cached for the process lifetime.
    bot_username: str | None = None

    @property
    def bale_api_url(self) -> str:
        return f"https://tapi.bale.ai/bot{self.bale_token}/"

    @property
    def bale_file_url(self) -> str:
        return f"https://tapi.bale.ai/file/bot{self.bale_token}/"

    @staticmethod
    def load() -> "Settings":
        bale_token = os.getenv("BALE_TOKEN")

        gemini_key = os.getenv("COMETAPI_GEMINI_KEY")
        claude_key = os.getenv("COMETAPI_CLAUDE_KEY")
        gpt_key = os.getenv("COMETAPI_GPT_KEY")

        missing = [
            name
            for name, value in (
                ("BALE_TOKEN", bale_token),
                ("COMETAPI_GEMINI_KEY", gemini_key),
                ("COMETAPI_CLAUDE_KEY", claude_key),
                ("COMETAPI_GPT_KEY", gpt_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )

        return Settings(
            bale_token=bale_token,
            comet_base_url=os.getenv("COMETAPI_BASE_URL", "https://api.cometapi.com/v1"),
            comet_gemini_key=gemini_key,
            comet_claude_key=claude_key,
            comet_gpt_key=gpt_key,
            gpt_model=os.getenv("COMETAPI_GPT_MODEL", "gpt-5.6-sol"),
            gemini_model=os.getenv("COMETAPI_GEMINI_MODEL", "gemini-3-flash"),
            claude_model=os.getenv("COMETAPI_CLAUDE_MODEL", "claude-sonnet-4-6"),
            worker_count=_get_int("WORKER_COUNT", 8),
            queue_max_size=_get_int("QUEUE_MAX_SIZE", 1000),
            http_timeout_total=_get_float("HTTP_TIMEOUT_TOTAL", 60.0),
            http_timeout_connect=_get_float("HTTP_TIMEOUT_CONNECT", 10.0),
            ai_request_timeout=_get_float("AI_REQUEST_TIMEOUT", 90.0),
            max_retries=_get_int("MAX_RETRIES", 3),
            retry_backoff_base=_get_float("RETRY_BACKOFF_BASE", 1.5),
            connector_limit=_get_int("CONNECTOR_LIMIT", 100),
            connector_limit_per_host=_get_int("CONNECTOR_LIMIT_PER_HOST", 30),
            polling_timeout=_get_int("POLLING_TIMEOUT", 10),
            polling_error_backoff=_get_float("POLLING_ERROR_BACKOFF", 3.0),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            min_valid_image_bytes=_get_int("MIN_VALID_IMAGE_BYTES", 1000),
            group_require_mention=os.getenv("GROUP_REQUIRE_MENTION", "true").strip().lower()
            not in ("false", "0", "no"),
            bot_username=(os.getenv("BOT_USERNAME") or None),
        )


SYSTEM_PROMPT = (
    "تو یک استاد دانشگاه و متخصص علوم و مهندسی هستی. تصویر ارسالی یک سوال امتحانی است."
    "سوال را به دقت تحلیل کن و پاسخ را کاملاً گام‌به‌گام، تشریحی و با فرمول‌های دقیق بنویس"
    "مهم: بدون هیچ توضیح اضافه در خروجی فقط جواب سوال رو بنویس"
)
