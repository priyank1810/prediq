"""
ML Worker Process — polls SQLite job queue and processes heavy tasks.

Usage:
    python worker.py            # run continuously
    python worker.py --once     # process one batch, then exit (useful for testing)

Environment variables:
    WORKER_CONCURRENCY  — max parallel jobs (default: 2)
    LOW_RESOURCE_MODE   — throttle TensorFlow threads when "true" or "1"
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure TensorFlow before any import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if os.getenv("LOW_RESOURCE_MODE", "").lower() in ("true", "1"):
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("worker")

# --- Ensure tables exist ---
from app.database import engine, Base
from app.models import JobQueue  # noqa: F401 — registers the model
Base.metadata.create_all(bind=engine)

from app.services.job_service import job_service
from app.database import SessionLocal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "2"))
HEALTH_LOG_INTERVAL = 300  # seconds (5 minutes)


class Worker:
    """Polls the job queue and dispatches to the appropriate handler."""

    def __init__(self, concurrency: int = WORKER_CONCURRENCY):
        self._shutdown = False
        self._concurrency = max(1, concurrency)
        self._worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self._signal_service = None
        # Health counters (thread-safe via lock)
        self._lock = threading.Lock()
        self._jobs_processed = 0
        self._jobs_failed = 0
        self._start_time = time.monotonic()

    # --- Lazy-loaded services (defer heavy imports) ---

    @property
    def signal_service(self):
        if self._signal_service is None:
            log.info("Lazy-loading signal_service...")
            from app.services.signal_service import signal_service
            self._signal_service = signal_service
        return self._signal_service

    # --- Signal handlers ---

    def handle_shutdown(self, signum, frame):
        log.info("Received shutdown signal (%s), finishing current jobs...", signum)
        self._shutdown = True

    # --- Job dispatchers ---


    def handle_watchlist_trade_scan(self, params: dict) -> dict:
        """Compute MTF signals for all watchlist stocks and log trade predictions.
        scan_type: "full" (all timeframes) or "short" (intraday + short_term only)."""
        from app.models import WatchlistItem
        from app.services.trade_tracker import trade_tracker
        import time as _time

        scan_type = params.get("scan_type", "full")

        db = SessionLocal()
        try:
            symbols = [row.symbol for row in db.query(WatchlistItem.symbol).distinct().all()]
        finally:
            db.close()

        if not symbols:
            return {"symbols_processed": 0}

        logged = 0
        for sym in symbols:
            try:
                result = self.signal_service.get_multi_timeframe_signals(sym)
                current_price = result.get("current_price", 0)

                if scan_type == "short":
                    # Only log intraday and short_term, skip long_term
                    for tf_key in ("intraday", "short_term"):
                        sig = result.get(tf_key)
                        if sig:
                            trade_tracker.log_signal(sym, tf_key, sig, current_price)
                # "full" scan logs all via the internal call in get_multi_timeframe_signals

                logged += 1
            except Exception as e:
                log.debug("Trade scan failed for %s: %s", sym, e)

            # 30s pause between stocks to stay gentle on resources
            _time.sleep(30)

        return {"symbols_processed": logged, "total": len(symbols), "scan_type": scan_type}

    def handle_trade_validate(self, params: dict) -> dict:
        """Validate open trade predictions and learn from results."""
        from app.services.trade_tracker import trade_tracker
        result = trade_tracker.validate_open_signals()
        if result.get("resolved", 0) > 0:
            trade_tracker.learn_from_trades()
        return result

    # --- Health logging ---

    def _log_health(self):
        """Log basic health stats: jobs processed, failed, uptime."""
        uptime_secs = time.monotonic() - self._start_time
        hours, remainder = divmod(int(uptime_secs), 3600)
        minutes, seconds = divmod(remainder, 60)
        with self._lock:
            processed = self._jobs_processed
            failed = self._jobs_failed
        log.info(
            "Health: %s | processed=%d failed=%d concurrency=%d uptime=%dh%02dm%02ds",
            self._worker_id, processed, failed, self._concurrency,
            hours, minutes, seconds,
        )

    # --- Single job processing (used by thread pool) ---

    def _process_one_job(self, job: dict) -> bool:
        """Process a single claimed job. Returns True on success, False on failure."""
        job_id = job["id"]
        job_type = job["job_type"]
        params = job["params"]

        handler_name = self.DISPATCH.get(job_type)
        if not handler_name:
            job_service.fail(job_id, f"Unknown job type: {job_type}")
            log.warning("Unknown job type: %s (job %d)", job_type, job_id)
            return False

        log.info("Processing job %d: %s %s", job_id, job_type,
                 params.get("symbol", "") if isinstance(params, dict) else "")

        try:
            handler = getattr(self, handler_name)
            result = handler(params)
            job_service.complete(job_id, result)
            log.info("Completed job %d: %s", job_id, job_type)
            with self._lock:
                self._jobs_processed += 1
            return True
        except Exception as e:
            job_service.fail(job_id, str(e))
            log.error("Job %d failed: %s", job_id, e)
            with self._lock:
                self._jobs_failed += 1
            return False

    # --- Main loop ---

    DISPATCH = {
        "watchlist_trade_scan": "handle_watchlist_trade_scan",
        "trade_validate": "handle_trade_validate",
    }

    def run(self, once: bool = False):
        """Start the worker loop.

        Args:
            once: If True, process one batch of up to *concurrency* jobs, then exit.
                  Useful for testing and one-shot cron invocations.
        """
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

        log.info(
            "Worker %s started (concurrency=%d, once=%s), polling for jobs...",
            self._worker_id, self._concurrency, once,
        )

        last_stale_check = time.time()
        last_cleanup = time.time()
        last_health_log = time.time()

        executor = ThreadPoolExecutor(max_workers=self._concurrency)

        try:
            while not self._shutdown:
                try:
                    # --- Periodic maintenance ---
                    now_ts = time.time()

                    if now_ts - last_stale_check > 60:
                        # Use the new release_stale_jobs (10-min timeout) for
                        # multi-worker safety; fall back to the legacy reset_stale
                        # which is compatible with the old single-worker setup.
                        released = job_service.release_stale_jobs()
                        if released:
                            log.info("Released %d stale jobs (>10 min)", released)
                        last_stale_check = now_ts

                    if now_ts - last_cleanup > 3600:
                        deleted = job_service.cleanup_old()
                        if deleted:
                            log.info("Cleaned up %d old jobs", deleted)
                        last_cleanup = now_ts

                    if now_ts - last_health_log > HEALTH_LOG_INTERVAL:
                        self._log_health()
                        last_health_log = now_ts

                    # --- Claim a batch of jobs (up to concurrency) ---
                    jobs = []
                    for _ in range(self._concurrency):
                        job = job_service.claim_job(worker_id=self._worker_id)
                        if job is None:
                            break
                        jobs.append(job)

                    if not jobs:
                        if once:
                            log.info("--once mode: no pending jobs, exiting.")
                            break
                        time.sleep(1)
                        continue

                    # --- Submit jobs to thread pool ---
                    futures = {
                        executor.submit(self._process_one_job, job): job
                        for job in jobs
                    }

                    # Wait for the batch to complete (with periodic shutdown check)
                    for future in as_completed(futures):
                        if self._shutdown:
                            break
                        # Propagate any unexpected executor errors
                        try:
                            future.result()
                        except Exception as e:
                            failed_job = futures[future]
                            log.error("Unexpected executor error for job %d: %s",
                                      failed_job["id"], e)

                    if once:
                        log.info("--once mode: batch complete, exiting.")
                        break

                except Exception as e:
                    log.error("Worker loop error: %s", e)
                    if once:
                        break
                    time.sleep(1)
        finally:
            # Graceful shutdown: let running threads finish, then tear down
            log.info("Shutting down thread pool (waiting for in-flight jobs)...")
            executor.shutdown(wait=True)
            self._log_health()
            log.info("Worker %s shut down gracefully.", self._worker_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Stock Tracker ML Worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single batch of jobs, then exit (useful for testing).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=WORKER_CONCURRENCY,
        help=f"Max parallel jobs (default: {WORKER_CONCURRENCY}, from WORKER_CONCURRENCY env).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    worker = Worker(concurrency=args.concurrency)
    worker.run(once=args.once)
