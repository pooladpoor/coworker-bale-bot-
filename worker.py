"""Worker coroutines that pull jobs off the queue and process them
end-to-end: download image -> call both AIs in parallel -> render HTML
-> upload result. Workers run independently of the polling loop and of
each other, so many users are served concurrently.
"""

from __future__ import annotations

import logging
import time

from bale_client import BaleClient
from comet_client import CometAPIClient
from config import Settings
from html_renderer import render_report_buffer
from models import IncomingJob, JobTimings
from queue_manager import JobQueue
from utils import Stopwatch

logger = logging.getLogger(__name__)


class Worker:
    """One logical worker. Multiple instances run concurrently as asyncio
    tasks, each pulling from the same shared queue."""

    def __init__(
        self,
        worker_id: int,
        queue: JobQueue,
        bale: BaleClient,
        comet: CometAPIClient,
        settings: Settings,
    ) -> None:
        self.worker_id = worker_id
        self._queue = queue
        self._bale = bale
        self._comet = comet
        self._settings = settings

    async def run(self) -> None:
        logger.info("Worker %d started", self.worker_id)
        while True:
            job = await self._queue.get()
            try:
                await self._process_job(job)
            except Exception:  # noqa: BLE001 - a single bad job must not kill the worker
                logger.exception("Worker %d: unhandled error processing job for chat %s",
                                  self.worker_id, job.chat_id)
                await self._safe_notify_failure(job.chat_id)
            finally:
                self._queue.task_done()

    async def _safe_notify_failure(self, chat_id: int) -> None:
        try:
            await self._bale.send_message(
                chat_id, "❌ خطایی در پردازش تصویر رخ داد. لطفاً دوباره امتحان کنید."
            )
        except Exception:  # noqa: BLE001
            logger.exception("Worker %d: also failed to notify user %s of the error",
                              self.worker_id, chat_id)

    async def _process_job(self, job: IncomingJob) -> None:
        timings = JobTimings()
        total_sw = Stopwatch()

        with total_sw:
            await self._bale.send_message(
                job.chat_id, "⏳ تصویر دریافت شد. در حال پردازش... لطفاً صبور باشید."
            )

            # 1. Resolve + download the image.
            with Stopwatch() as sw:
                file_path = await self._bale.get_file_path(job.file_id)
                image_bytes = await self._bale.download_file(file_path)
            timings.download_seconds = sw.elapsed

            if len(image_bytes) < self._settings.min_valid_image_bytes:
                logger.warning(
                    "Job for chat %s: downloaded content looks too small (%d bytes) "
                    "to be a real image",
                    job.chat_id,
                    len(image_bytes),
                )

            logger.info(
                "Chat %s: image downloaded in %.2fs (%d bytes)",
                job.chat_id, timings.download_seconds, len(image_bytes),
            )

            # 2. Gemini + GPT in parallel, image base64-encoded exactly once.
            gemini_result, gpt_result = await self._comet.ask_both(image_bytes)
            timings.gemini_seconds = gemini_result.elapsed_seconds
            timings.gpt_seconds = gpt_result.elapsed_seconds

            logger.info(
                "Chat %s: gemini=%.2fs (error=%s) gpt=%.2fs (error=%s)",
                job.chat_id,
                gemini_result.elapsed_seconds, gemini_result.error,
                gpt_result.elapsed_seconds, gpt_result.error,
            )

            # 3. Render HTML report in memory (no disk I/O).
            with Stopwatch() as sw:
                buffer = render_report_buffer(gemini_result.text, gpt_result.text)
            timings.html_seconds = sw.elapsed

            # 4. Upload result to the user.
            with Stopwatch() as sw:
                await self._bale.send_document(
                    job.chat_id,
                    buffer,
                    filename=f"answer_{job.chat_id}.html",
                    caption="🌐 پاسخ سوال شما آماده شد! فایل HTML بالا را باز کنید.",
                )
            timings.upload_seconds = sw.elapsed

        timings.total_seconds = total_sw.elapsed
        logger.info(
            "Chat %s DONE | download=%.2fs gemini=%.2fs gpt=%.2fs html=%.2fs "
            "upload=%.2fs total=%.2fs | queue_wait=%.2fs",
            job.chat_id,
            timings.download_seconds,
            timings.gemini_seconds,
            timings.gpt_seconds,
            timings.html_seconds,
            timings.upload_seconds,
            timings.total_seconds,
            time.monotonic() - job.enqueued_at - timings.total_seconds,
        )
