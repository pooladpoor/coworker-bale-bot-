"""Shared data structures used across the bot."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class IncomingJob:
    """A single unit of work produced by the polling loop and consumed
    by a worker."""

    chat_id: int
    file_id: str
    message_id: int | None = None
    chat_type: str = "private"
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class AIResult:
    """Result of a single AI model call."""

    model_name: str
    text: str
    elapsed_seconds: float
    error: bool = False


@dataclass(slots=True)
class JobTimings:
    """Per-stage timing breakdown, used for benchmark logging."""

    download_seconds: float = 0.0
    gemini_seconds: float = 0.0
    gpt_seconds: float = 0.0
    html_seconds: float = 0.0
    upload_seconds: float = 0.0
    total_seconds: float = 0.0
