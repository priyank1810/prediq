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
import asyncio
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


def _mins_to_candle_close(timeframe: str) -> "int | None":
    """Return minutes remaining until the current candle closes (IST).

    NSE market: 9:15–15:30 IST.
    1h candles anchor to 9:15 → close at 10:15, 11:15, 12:15, 13:15, 14:15, 15:15.
    4h candles: 9:15–13:15 and 13:15–15:30 (partial).
    """
    from app.utils.helpers import now_ist
    now = now_ist()
    minute_of_day = (now.hour * 60 + now.minute) - (9 * 60 + 15)  # mins since market open
    if minute_of_day < 0:
        return None

    if "1h" in timeframe:
        mins_into_candle = minute_of_day % 60
        return 60 - mins_into_candle
    if "4h" in timeframe:
        # First 4h candle: 0–240 min (9:15–13:15), second: 240–375 min (13:15–15:30)
        if minute_of_day < 240:
            return 240 - minute_of_day
        elif minute_of_day < 375:
            return 375 - minute_of_day
    return None


def _fire_telegram_signal(signal_data: dict) -> None:
    """Fire-and-forget: broadcast a signal alert to Telegram subscribers.

    Runs in a daemon thread because the worker is synchronous — there is no
    running event loop to schedule coroutines onto.
    """
    from app.services.telegram_service import broadcast_to_subscribers, send_signal_alert

    def _run():
        asyncio.run(broadcast_to_subscribers("signals", send_signal_alert, signal_data))

    threading.Thread(target=_run, daemon=True).start()


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
        """Compute MTF signals for watchlist + popular stocks.
        Watchlist: log all non-neutral signals.
        Popular stocks (not in watchlist): only log high-confidence (>=60%) signals."""
        from app.models import WatchlistItem
        from app.services.trade_tracker import trade_tracker
        from app.services.data_fetcher import data_fetcher
        from app.config import POPULAR_STOCKS
        import time as _time

        scan_type = params.get("scan_type", "full")

        db = SessionLocal()
        try:
            watchlist = set(row.symbol for row in db.query(WatchlistItem.symbol).distinct().all())
        finally:
            db.close()

        # Combine: watchlist + popular stocks not already in watchlist
        popular_extra = [s for s in POPULAR_STOCKS if s not in watchlist]

        # Filter popular stocks by liquidity — skip stocks with avg daily volume < 500k
        # (watchlist stocks are always scanned regardless of volume)
        MIN_AVG_VOLUME = 500_000
        if popular_extra:
            try:
                vol_quotes = data_fetcher.get_bulk_quotes(popular_extra)
                vol_map = {}
                for q in (vol_quotes if isinstance(vol_quotes, list) else []):
                    if q.get("symbol") and q.get("avg_volume"):
                        vol_map[q["symbol"]] = q["avg_volume"]
                liquid = [s for s in popular_extra if vol_map.get(s, MIN_AVG_VOLUME) >= MIN_AVG_VOLUME]
                skipped = len(popular_extra) - len(liquid)
                if skipped:
                    log.info("Volume filter: skipped %d illiquid stocks (avg_vol < %d)", skipped, MIN_AVG_VOLUME)
                popular_extra = liquid
            except Exception as e:
                log.warning("Volume filter failed, scanning all popular stocks: %s", e)

        all_symbols = list(watchlist) + popular_extra

        if not all_symbols:
            return {"symbols_processed": 0}

        logged = 0
        popular_logged = 0
        watchlist_bullish = 0
        near_bullish = []  # Popular stocks close to turning bullish
        scan_4h_bullish: set[str] = set()  # symbols with bullish 4h signal this scan
        pending_alerts: list[dict] = []    # candidates for Telegram (sector filter applied after scan)

        # Phase 1: Scan watchlist first to detect if all bearish
        for sym in watchlist:
            try:
                result = self.signal_service.get_multi_timeframe_signals(sym)
                current_price = result.get("current_price", 0)
                intraday = result.get("intraday", {})
                short_term = result.get("short_term", {})

                # Use live LTP for entry price (candle close can lag)
                live_price = current_price
                try:
                    quotes = data_fetcher.get_bulk_quotes([sym])
                    if quotes:
                        q = quotes[0] if isinstance(quotes, list) else quotes.get(sym, {})
                        if q.get("ltp"):
                            live_price = round(float(q["ltp"]), 2)
                except Exception:
                    pass

                all_sigs = list(intraday.values()) + list(short_term.values())
                bullish_count = sum(1 for sig in all_sigs if sig and sig.get("direction") == "BULLISH")
                watchlist_bullish += bullish_count

                sig_4h = short_term.get("4h") or {}
                if sig_4h.get("direction") == "BULLISH":
                    scan_4h_bullish.add(sym)

                # Multi-timeframe agreement: require 2+ bullish timeframes
                if bullish_count >= 2:
                    if scan_type in ("intraday", "full"):
                        for tf_key, sig in intraday.items():
                            if sig and sig.get("direction") == "BULLISH":
                                trade_tracker.log_signal(sym, f"intraday_{tf_key}", sig, live_price)
                    if scan_type in ("short", "full"):
                        for tf_key, sig in short_term.items():
                            if sig and sig.get("direction") == "BULLISH":
                                trade_tracker.log_signal(sym, f"short_{tf_key}", sig, live_price)
                        # Queue Telegram candidate — sector filter applied after full scan
                        sig_1h = short_term.get("1h") or {}
                        if (sig_1h.get("direction") == "BULLISH" and (sig_1h.get("confidence") or 0) >= 45
                                and sig_4h.get("direction") == "BULLISH" and (sig_4h.get("confidence") or 0) >= 45):
                            pending_alerts.append({"symbol": sym, "sig_1h": sig_1h, "sig_4h": sig_4h})

                logged += 1
            except Exception as e:
                log.debug("Trade scan failed for %s: %s", sym, e)
            _time.sleep(1 if scan_type == "intraday" else 3)

        # Adaptive threshold: if watchlist is all bearish, lower popular stock threshold
        if watchlist_bullish == 0 and len(watchlist) > 0:
            popular_threshold = 45  # Lower threshold — hunt harder for opportunities
            log.info("All watchlist bearish — lowering popular stock threshold to 45%")
        else:
            popular_threshold = 60  # Normal threshold

        # Phase 2: Scan popular stocks
        for sym in popular_extra:
            try:
                result = self.signal_service.get_multi_timeframe_signals(sym)
                current_price = result.get("current_price", 0)
                intraday = result.get("intraday", {})
                short_term = result.get("short_term", {})

                # Use live LTP for entry price
                live_price = current_price
                try:
                    quotes = data_fetcher.get_bulk_quotes([sym])
                    if quotes:
                        q = quotes[0] if isinstance(quotes, list) else quotes.get(sym, {})
                        if q.get("ltp"):
                            live_price = round(float(q["ltp"]), 2)
                except Exception:
                    pass

                all_sigs = list(intraday.values()) + list(short_term.values())

                # Track near-bullish stocks (neutral with positive composite)
                for sig in all_sigs:
                    if sig and sig.get("direction") == "NEUTRAL" and (sig.get("score") or 0) > 0:
                        near_bullish.append({
                            "symbol": sym,
                            "score": sig.get("score", 0),
                            "confidence": sig.get("confidence", 0),
                            "label": sig.get("label", ""),
                        })

                # Multi-timeframe agreement: require 2+ bullish timeframes (long-only)
                bullish_count = sum(1 for sig in all_sigs if sig and sig.get("direction") == "BULLISH")

                sig_4h = short_term.get("4h") or {}
                if sig_4h.get("direction") == "BULLISH":
                    scan_4h_bullish.add(sym)

                if bullish_count >= 2:
                    if scan_type in ("intraday", "full"):
                        for tf_key, sig in intraday.items():
                            if sig and sig.get("direction") == "BULLISH" and (sig.get("confidence") or 0) >= popular_threshold:
                                trade_tracker.log_signal(sym, f"intraday_{tf_key}", sig, live_price)
                                popular_logged += 1
                    if scan_type in ("short", "full"):
                        for tf_key, sig in short_term.items():
                            if sig and sig.get("direction") == "BULLISH" and (sig.get("confidence") or 0) >= popular_threshold:
                                trade_tracker.log_signal(sym, f"short_{tf_key}", sig, live_price)
                                popular_logged += 1
                        # Queue Telegram candidate — sector filter applied after full scan
                        sig_1h = short_term.get("1h") or {}
                        if (sig_1h.get("direction") == "BULLISH" and (sig_1h.get("confidence") or 0) >= 45
                                and sig_4h.get("direction") == "BULLISH" and (sig_4h.get("confidence") or 0) >= 45):
                            pending_alerts.append({"symbol": sym, "sig_1h": sig_1h, "sig_4h": sig_4h})

                logged += 1
            except Exception as e:
                log.debug("Trade scan failed for %s: %s", sym, e)
            _time.sleep(1 if scan_type == "intraday" else 3)

        # Save near-bullish stocks to cache for the UI
        if near_bullish:
            from app.utils.cache import cache
            near_bullish.sort(key=lambda x: -x["score"])
            cache.set("near_bullish_stocks", near_bullish[:10], 3600)

        if popular_logged:
            log.info(f"Popular stocks: {popular_logged} signals logged (threshold: {popular_threshold}%)")

        # ── Sector momentum filter + fire Telegram ──
        # Only alert if ≥30% of scanned sector peers are also bullish on 4h.
        # Unmapped stocks (no sector) are always allowed through.
        if pending_alerts and scan_type in ("short", "full"):
            from app.config import get_sector_peers
            alerts_fired = 0
            for alert in pending_alerts:
                sym = alert["symbol"]
                peers = get_sector_peers(sym)
                if peers:
                    bullish_peers = sum(1 for p in peers if p in scan_4h_bullish)
                    sector_ok = bullish_peers / len(peers) >= 0.30
                    if not sector_ok:
                        log.info("Sector filter blocked %s: %d/%d peers bullish", sym, bullish_peers, len(peers))
                        continue
                m4h = _mins_to_candle_close("4h")
                m1h = _mins_to_candle_close("1h")
                _fire_telegram_signal({**alert["sig_4h"], "symbol": sym, "timeframe": "short_4h",
                                       "mins_to_close": m4h})
                _fire_telegram_signal({**alert["sig_1h"], "symbol": sym, "timeframe": "short_1h",
                                       "mins_to_close": m1h})
                alerts_fired += 1
            log.info("Telegram: %d/%d candidates passed sector filter", alerts_fired, len(pending_alerts))

        result = {
            "symbols_processed": logged,
            "total": len(all_symbols),
            "watchlist": len(watchlist),
            "watchlist_bullish": watchlist_bullish,
            "popular_scanned": len(popular_extra),
            "popular_logged": popular_logged,
            "popular_threshold": popular_threshold,
            "near_bullish": len(near_bullish),
            "scan_type": scan_type,
        }

        # Save scan status for the UI
        from app.utils.cache import cache
        from app.utils.helpers import now_ist
        cache.set("last_scan_status", {
            **result,
            "timestamp": now_ist().isoformat(),
            "stocks_list": [s for s in all_symbols],
        }, 7200)

        return result

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
