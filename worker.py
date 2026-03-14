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
from datetime import timedelta

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
from app.models import SignalLog
from app.utils.helpers import now_ist

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
        self._prediction_service = None
        self._signal_service = None
        self._oi_service = None
        # Health counters (thread-safe via lock)
        self._lock = threading.Lock()
        self._jobs_processed = 0
        self._jobs_failed = 0
        self._start_time = time.monotonic()

    # --- Lazy-loaded services (defer heavy imports) ---

    @property
    def prediction_service(self):
        if self._prediction_service is None:
            log.info("Lazy-loading prediction_service...")
            from app.services.prediction_service import prediction_service
            self._prediction_service = prediction_service
        return self._prediction_service

    @property
    def signal_service(self):
        if self._signal_service is None:
            log.info("Lazy-loading signal_service...")
            from app.services.signal_service import signal_service
            self._signal_service = signal_service
        return self._signal_service

    @property
    def oi_service(self):
        if self._oi_service is None:
            log.info("Lazy-loading oi_service...")
            from app.services.oi_service import oi_service
            self._oi_service = oi_service
        return self._oi_service

    # --- Signal handlers ---

    def handle_shutdown(self, signum, frame):
        log.info("Received shutdown signal (%s), finishing current jobs...", signum)
        self._shutdown = True

    # --- Job dispatchers ---

    def handle_prediction(self, params: dict) -> dict:
        result = self.prediction_service.predict(
            symbol=params["symbol"],
            horizon=params.get("horizon", "1d"),
            models=params.get("models"),
        )
        return result

    def handle_signal(self, params: dict) -> dict:
        symbol = params["symbol"]
        signal_data = self.signal_service.get_signal(symbol)
        if not signal_data:
            raise ValueError(f"No signal data for {symbol}")

        # Log to SignalLog (1-min throttle, sector detection)
        self._log_signal(symbol, signal_data)

        return signal_data

    def handle_watchlist_signals(self, params: dict) -> dict:
        """Generate signals for all watchlist stocks."""
        from app.models import WatchlistItem

        db = SessionLocal()
        try:
            symbols = [
                row.symbol
                for row in db.query(WatchlistItem.symbol).distinct().all()
            ]
        finally:
            db.close()

        results = {}
        for sym in symbols:
            try:
                signal_data = self.signal_service.get_signal(sym)
                if signal_data:
                    self._log_signal(sym, signal_data)
                    results[sym] = signal_data
            except Exception as e:
                log.warning("Signal failed for %s: %s", sym, e)

        return {"symbols_processed": len(results), "signals": results}

    def handle_mtf_stream(self, params: dict) -> dict:
        """Compute MTF confluence for subscribed symbols."""
        from app.services.data_fetcher import data_fetcher
        from app.services.indicator_service import indicator_service

        symbols = params.get("symbols", [])
        results = {}
        for sym in symbols:
            try:
                intraday_df = data_fetcher.get_intraday_data(sym, "5d", "15m")
                tech_score = 0
                if intraday_df is not None and not intraday_df.empty:
                    tech_result = indicator_service.compute_intraday_indicators(intraday_df)
                    tech_score = tech_result["score"]
                mtf = self.signal_service._compute_mtf_confluence(sym, intraday_df, tech_score)
                mtf["symbol"] = sym
                results[sym] = mtf
            except Exception as e:
                log.warning("MTF failed for %s: %s", sym, e)

        return {"symbols_processed": len(results), "mtf_data": results}

    def handle_oi_stream(self, params: dict) -> dict:
        """Compute OI analysis for subscribed symbols."""
        symbols = params.get("symbols", [])
        results = {}
        for sym in symbols:
            try:
                result = self.oi_service.get_oi_analysis(sym)
                if result.get("available"):
                    result["symbol"] = sym
                    results[sym] = result
            except Exception as e:
                log.warning("OI failed for %s: %s", sym, e)

        return {"symbols_processed": len(results), "oi_data": results}

    # --- Signal logging helper ---

    def _log_signal(self, symbol: str, signal_data: dict):
        """Log signal to DB with 1-min throttle and sector detection."""
        try:
            db = SessionLocal()
            try:
                last_log = (
                    db.query(SignalLog)
                    .filter(SignalLog.symbol == symbol)
                    .order_by(SignalLog.created_at.desc())
                    .first()
                )
                should_log = (
                    last_log is None
                    or (now_ist().replace(tzinfo=None) - last_log.created_at) > timedelta(minutes=1)
                )
                if not should_log:
                    return

                price = None
                candles = signal_data.get("intraday_candles", [])
                if candles:
                    price = candles[-1].get("close")

                from app.config import SECTOR_MAP
                sector = None
                for s, syms in SECTOR_MAP.items():
                    if symbol in syms:
                        sector = s
                        break

                oi_s = None
                oi_data = signal_data.get("oi_analysis", {})
                if oi_data.get("available"):
                    oi_s = oi_data.get("score")

                log_entry = SignalLog(
                    symbol=symbol,
                    direction=signal_data["direction"],
                    confidence=signal_data["confidence"],
                    composite_score=signal_data["composite_score"],
                    technical_score=signal_data["technical"]["score"],
                    sentiment_score=signal_data["sentiment"]["score"],
                    global_score=signal_data["global_market"]["score"],
                    oi_score=oi_s,
                    price_at_signal=price,
                    sector=sector,
                )
                db.add(log_entry)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            log.warning("Signal logging failed for %s: %s", symbol, e)

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
        "prediction": "handle_prediction",
        "signal": "handle_signal",
        "watchlist_signals": "handle_watchlist_signals",
        "mtf_stream": "handle_mtf_stream",
        "oi_stream": "handle_oi_stream",
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
