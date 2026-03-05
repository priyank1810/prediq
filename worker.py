"""
ML Worker Process — polls SQLite job queue and processes heavy tasks.

Usage: python worker.py
"""

import os
import sys
import json
import time
import signal
import logging
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


class Worker:
    """Polls the job queue and dispatches to the appropriate handler."""

    def __init__(self):
        self._shutdown = False
        self._prediction_service = None
        self._signal_service = None
        self._oi_service = None

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
        log.info("Received shutdown signal, finishing current job...")
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

    # --- Main loop ---

    DISPATCH = {
        "prediction": "handle_prediction",
        "signal": "handle_signal",
        "watchlist_signals": "handle_watchlist_signals",
        "mtf_stream": "handle_mtf_stream",
        "oi_stream": "handle_oi_stream",
    }

    def run(self):
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

        log.info("Worker started, polling for jobs...")

        last_stale_check = time.time()
        last_cleanup = time.time()

        while not self._shutdown:
            try:
                # Periodic maintenance
                now = time.time()
                if now - last_stale_check > 60:
                    reset = job_service.reset_stale()
                    if reset:
                        log.info("Reset %d stale jobs", reset)
                    last_stale_check = now

                if now - last_cleanup > 3600:
                    deleted = job_service.cleanup_old()
                    if deleted:
                        log.info("Cleaned up %d old jobs", deleted)
                    last_cleanup = now

                # Poll for next job
                job = job_service.claim_next()
                if not job:
                    time.sleep(1)
                    continue

                job_id = job["id"]
                job_type = job["job_type"]
                params = job["params"]

                handler_name = self.DISPATCH.get(job_type)
                if not handler_name:
                    job_service.fail(job_id, f"Unknown job type: {job_type}")
                    log.warning("Unknown job type: %s (job %d)", job_type, job_id)
                    continue

                log.info("Processing job %d: %s %s", job_id, job_type,
                         params.get("symbol", "") if isinstance(params, dict) else "")

                handler = getattr(self, handler_name)
                result = handler(params)
                job_service.complete(job_id, result)

                log.info("Completed job %d: %s", job_id, job_type)

            except Exception as e:
                if 'job_id' in locals():
                    job_service.fail(job_id, str(e))
                    log.error("Job %d failed: %s", job_id, e)
                else:
                    log.error("Worker error: %s", e)

        log.info("Worker shut down gracefully.")


if __name__ == "__main__":
    worker = Worker()
    worker.run()
