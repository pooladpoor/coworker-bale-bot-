"""Small cross-cutting utilities: logging setup and a generic async retry
helper with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once for the whole process."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep noisy third-party libraries quieter unless we're debugging.
    if level.upper() != "DEBUG":
        logging.getLogger("aiohttp").setLevel(logging.WARNING)


class Stopwatch:
    """Tiny context manager to measure elapsed wall-clock time."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed = time.monotonic() - self._start


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    backoff_base: float,
    retriable_exceptions: tuple[type[BaseException], ...],
    logger: logging.Logger,
    op_name: str,
) -> T:
    """Run `func` with exponential backoff retries.

    Raises the last exception if all attempts fail. `asyncio.CancelledError`
    is never retried and always propagates immediately.
    """
    attempt = 0
    while True:
        try:
            return await func()
        except asyncio.CancelledError:
            raise
        except retriable_exceptions as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error("%s failed after %d attempts: %s", op_name, attempt, exc)
                raise
            delay = backoff_base ** attempt
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                op_name,
                attempt,
                max_retries,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
