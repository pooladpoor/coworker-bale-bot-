"""
Centralized configuration for the bot.

All tunables are read from environment variables (via `.env`) so the
deployment can be adjusted without touching code. Sensible defaults are
provided for anything non-critical.

Each of the three models (Gemini, Claude, GPT) is configured completely
independently: its own base URL, its own API key, and its own model
name. This means each one can point at a *different* provider/platform
(CometAPI, a direct provider API, a self-hosted gateway, etc.) -- there
is no hardcoded dependency on any single platform.
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


def _get_id_set(name: str) -> frozenset[int]:
    raw = os.getenv(name, "")
    ids: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError:
            raise ValueError(
                f"{name} contains a non-numeric value: {chunk!r}. "
                "Use a comma-separated list of numeric Bale user IDs."
            )
    return frozenset(ids)


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Everything needed to call one model's API: which platform
    (base_url), which credential, and which model name."""

    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True, slots=True)
class Settings:
    # --- Bale -----------------------------------------------------------------
    bale_token: str

    # --- Per-model provider configuration -------------------------------------
    # Each model is fully independent -- point any of them at any
    # OpenAI-compatible platform (CometAPI, OpenRouter, a direct provider
    # API, a self-hosted gateway, ...) by changing its base URL.
    gemini: ProviderConfig = field(
        default_factory=lambda: ProviderConfig("", "", "gemini-3-flash")
    )
    claude: ProviderConfig = field(
        default_factory=lambda: ProviderConfig("", "", "claude-sonnet-4-6")
    )
    gpt: ProviderConfig = field(
        default_factory=lambda: ProviderConfig("", "", "gpt-5.6-sol")
    )

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

    # --- Access control ------------------------------------------------------
    # If non-empty, only these numeric Bale user IDs (the sender's own id,
    # not the chat id) may use the bot -- in both private chats and groups.
    # Everyone else gets a polite rejection message instead of an answer.
    allowed_user_ids: frozenset[int] = frozenset()

    @property
    def bale_api_url(self) -> str:
        return f"https://tapi.bale.ai/bot{self.bale_token}/"

    @property
    def bale_file_url(self) -> str:
        return f"https://tapi.bale.ai/file/bot{self.bale_token}/"

    @staticmethod
    def load() -> "Settings":
        bale_token = os.getenv("BALE_TOKEN")

        gemini_key = os.getenv("GEMINI_API_KEY")
        claude_key = os.getenv("CLAUDE_API_KEY")
        gpt_key = os.getenv("GPT_API_KEY")

        gemini_base_url = os.getenv("GEMINI_BASE_URL")
        claude_base_url = os.getenv("CLAUDE_BASE_URL")
        gpt_base_url = os.getenv("GPT_BASE_URL")

        missing = [
            name
            for name, value in (
                ("BALE_TOKEN", bale_token),
                ("GEMINI_API_KEY", gemini_key),
                ("CLAUDE_API_KEY", claude_key),
                ("GPT_API_KEY", gpt_key),
                ("GEMINI_BASE_URL", gemini_base_url),
                ("CLAUDE_BASE_URL", claude_base_url),
                ("GPT_BASE_URL", gpt_base_url),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required environment variable(s): " + ", ".join(missing)
            )

        return Settings(
            bale_token=bale_token,
            gemini=ProviderConfig(
                base_url=gemini_base_url.rstrip("/"),
                api_key=gemini_key,
                model=os.getenv("GEMINI_MODEL", "gemini-3-flash"),
            ),
            claude=ProviderConfig(
                base_url=claude_base_url.rstrip("/"),
                api_key=claude_key,
                model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            ),
            gpt=ProviderConfig(
                base_url=gpt_base_url.rstrip("/"),
                api_key=gpt_key,
                model=os.getenv("GPT_MODEL", "gpt-5.6-sol"),
            ),
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
            allowed_user_ids=_get_id_set("ALLOWED_USER_IDS"),
        )


SYSTEM_PROMPT = (
    "تو یک استاد دانشگاه و متخصص علوم و مهندسی هستی. تصویر ارسالی یک سوال امتحانی است."
    "سوال را به دقت تحلیل کن و پاسخ را کاملاً گام‌به‌گام، تشریحی و با فرمول‌های دقیق بنویس"
    "مهم: بدون هیچ توضیح اضافه در خروجی فقط جواب سوال رو بنویس"
)

ACCESS_DENIED_MESSAGE = "شرمنده! باید امین برات اشتراک فعال کنه..."
