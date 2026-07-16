# Bale AI Exam-Answer Bot — Async Architecture

Full async rewrite of the original synchronous `requests`-based bot.
All existing functionality is preserved: photo intake, parallel
Gemini + GPT calls via CometAPI, the two-column MathJax/Markdown HTML
report, and delivery back through Bale.

## Project layout

```
project/
  main.py            entry point: polling loop + worker pool
  config.py          Settings dataclass, loaded from .env
  models.py          IncomingJob / AIResult / JobTimings dataclasses
  bale_client.py      async Bale Bot API client
  comet_client.py     async CometAPI client (Gemini + GPT)
  html_renderer.py    markdown+MathJax protection, HTML template (in-memory)
  worker.py           per-job pipeline: download -> AI calls -> render -> upload
  queue_manager.py    typed asyncio.Queue wrapper
  utils.py            logging setup, Stopwatch, retry_async
  requirements.txt
  .env.example
```

## How it works

```
polling_loop()  --enqueue IncomingJob-->  asyncio.Queue  --consumed by-->  Worker(0..N)
```

The polling loop's *only* responsibility is calling `getUpdates` and
putting `IncomingJob` objects on the queue (or, for plain text
messages, firing off a cheap reply via `asyncio.create_task` without
awaiting it). It never waits on an AI call or a file upload, so a slow
Gemini/GPT response for user A never delays picking up user B's
message. `N` independent `Worker` coroutines pull from the same queue
and each run the full pipeline (download → AI calls → render → upload)
concurrently with each other.

## Setup

## Group support

The bot can be added to a Bale group. Behaviour:

- **Private chat**: unchanged -- any photo is processed, any text gets the greeting.
- **Group / supergroup**: to avoid answering every photo posted by anyone
  in a busy group, a photo is only processed if its **caption tags the
  bot** (e.g. `@your_bot_username این سوالو حل کن`). Plain text messages
  in a group are also ignored unless they mention the bot.
- The bot's own `@username` is resolved automatically at startup via
  `getMe()`; set `BOT_USERNAME` in `.env` to skip that call or override it.
- All replies in a group (the "در حال پردازش" notice, error messages, and
  the final HTML report) are sent as a **reply to the original photo
  message** (`reply_to_message_id`), so it's clear which question each
  answer belongs to even in a fast-moving group.
- Set `GROUP_REQUIRE_MENTION=false` in `.env` to make the bot answer
  every photo posted in a group, with no tag required.


```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in BALE_TOKEN and COMETAPI_KEY
python main.py
```

## Optimizations and their impact

| Change | Why it helps | Rough impact |
|---|---|---|
| **Single shared `aiohttp.ClientSession` / `TCPConnector`** for all Bale + CometAPI calls | Reuses TCP + TLS connections (keep-alive) instead of a fresh handshake per `requests.post()` call | Saves ~50–300ms per call (TLS handshake avoided), scales with request volume |
| **Non-blocking polling loop that only enqueues work** | The original loop processed a photo fully (download, two AI calls, HTML write, upload) before polling again — every other user was blocked for the whole duration | Turns "1 user at a time" into "N users at a time"; end-to-end latency for user #2 goes from *wait for user #1 to finish* to *near-zero* |
| **Worker pool (`WORKER_COUNT`, default 8) draining `asyncio.Queue`** | Bounds concurrency so we don't overwhelm CometAPI/Bale, while still processing many jobs in parallel | Throughput scales roughly linearly with worker count until an external API becomes the bottleneck |
| **`asyncio.TaskGroup` for Gemini + GPT calls** | Same as the original `ThreadPoolExecutor` intent, but native async — no thread-pool overhead, and either call's exception is isolated (a Gemini failure doesn't kill the GPT result) | Total AI latency = `max(gemini, gpt)` instead of `gemini + gpt`; typically halves this stage vs. any sequential approach |
| **Base64 image encoding done once, string reused** | The original design risked re-encoding per model; here it's encoded once in `ask_both()` and passed to both calls | Saves one redundant encode of a multi-hundred-KB image (a few ms, but also less peak memory) |
| **HTML rendered directly to a string / `io.BytesIO`, never touching disk** | Original wrote `answer_<chat_id>.html` to disk, then read it back for `sendDocument`, then deleted it | Removes 2 blocking disk I/O calls + 1 delete per job; also removes a whole class of bugs (leftover files on crash, filename collisions under concurrency) |
| **Exponential-backoff retries (`retry_async`) on all network calls** | Original had no retry logic at all — a single transient network blip failed the whole job | Meaningfully improves success rate under flaky network conditions (common over degraded connections) without needing manual re-sends |
| **Structured `logging` with per-stage timings** | Original used `print()`, hard to filter/aggregate in production | Enables real dashboards/alerting; timing breakdown (download/gemini/gpt/html/upload/total) shows exactly where time goes per request |
| **Per-model failure isolation** | If Gemini or GPT fails, the error is captured as a normal `AIResult` (marked `error=True`) rather than raising — the other model's answer still reaches the user | User always gets at least one answer instead of a total failure |
| **`dataclasses(slots=True)` for hot-path objects** | Reduces per-instance memory overhead and attribute-access cost for `IncomingJob`/`AIResult`/`JobTimings`, created once per photo | Minor, but adds up under high message volume |

### Net effect

- **Per-request latency**: AI stage time drops from `gemini_time + gpt_time` (sequential) to `max(gemini_time, gpt_time)` (parallel) — typically ~2x on that stage; disk I/O removal shaves a further few tens of ms.
- **Multi-user throughput**: the biggest win. The original bot was effectively single-user-at-a-time (blocking polling loop); this version serves up to `WORKER_COUNT` users truly concurrently, and the polling loop keeps ingesting new messages the entire time.
- **Reliability**: retries + per-model error isolation mean transient network issues or a single model outage no longer fail the whole job.

## Notes on preserved behavior

- The Markdown/MathJax protection regexes, the HTML template (including all CSS), RTL support, and the exact CometAPI request shapes (`chat/completions` for Gemini, `responses` for GPT) are unchanged from the original — only *how* they run (async, in-memory, pooled connections) changed.
- `SYSTEM_PROMPT` text is unchanged.
- Text messages still get the same greeting reply; photo messages still get the same "⏳ در حال پردازش" and final caption text.
