import os
import logging

# Configure TensorFlow before any TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress TF info/warning logs
if os.getenv("LOW_RESOURCE_MODE", "").lower() in ("true", "1"):
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.responses import Response
from app.database import engine, Base
from app.routers import stocks, predictions, portfolio, alerts, indicators, signals, watchlist, screener, options
from app.routers.mtf_dashboard import router as mtf_dashboard_router
from app.routers.fii_dii import router as fii_dii_router
from app.routers.sectors import router as sectors_router
from app.routers.websocket import router as ws_router, price_streamer, alert_checker, signal_accuracy_validator, signal_accuracy_validator_30min, signal_accuracy_validator_1hr
from app.routers.jobs import router as jobs_router
from app.routers.trade_journal import router as trade_journal_router
from app.routers.strategies import router as strategies_router
from app.routers.broker import router as broker_router
from app.routers.telegram import router as telegram_router
from app.routers.sms import router as sms_router
from app.routers.auth import router as auth_router
from app.utils.rate_limiter import RateLimiter


async def smart_alert_checker():
    """Background task to check smart alerts. Only runs when active alerts exist."""
    from app.services.alert_service import alert_service
    from app.database import SessionLocal
    from app.models import SmartAlert

    await asyncio.sleep(30)
    while True:
        try:
            from app.utils.helpers import is_market_open
            from app.routers.websocket import manager
            if is_market_open() and manager.active_connections:
                def _check_smart():
                    db = SessionLocal()
                    try:
                        count = db.query(SmartAlert).filter(SmartAlert.is_triggered == False).count()
                        if count > 0:
                            return alert_service.check_smart_alerts(db)
                    finally:
                        db.close()
                    return None

                triggered = await asyncio.to_thread(_check_smart)
                if triggered:
                    for alert_data in triggered:
                        await manager.broadcast_to_all("smart_alert_triggered", alert_data)
                        # Fire-and-forget Telegram broadcast for price alerts
                        from app.services.telegram_service import broadcast_to_subscribers, send_price_alert as _tg_price
                        asyncio.create_task(broadcast_to_subscribers("price_alerts", _tg_price, alert_data))
                        # Fire-and-forget SMS broadcast for price alerts
                        from app.services.sms_service import broadcast_to_subscribers as sms_broadcast, send_price_alert as _sms_price
                        asyncio.create_task(sms_broadcast("price_alerts", _sms_price, alert_data))
        except Exception:
            pass
        await asyncio.sleep(300)  # 5 min (was 60s) — smart alerts don't need second-level checks


async def market_mood_broadcaster():
    """Broadcast Market Mood Score every 5 minutes during market hours."""
    from app.config import MARKET_MOOD_REFRESH_INTERVAL

    await asyncio.sleep(60)
    while True:
        try:
            from app.utils.helpers import is_market_open
            from app.routers.websocket import manager
            if is_market_open() and manager.active_connections:
                from app.services.market_mood_service import market_mood_service
                mood = await asyncio.to_thread(market_mood_service.get_mood)
                await manager.broadcast_to_all("market_mood_update", mood)
        except Exception:
            pass
        await asyncio.sleep(MARKET_MOOD_REFRESH_INTERVAL)


def _migrate_db():
    """Add missing columns to existing tables.

    For SQLite: uses sqlite3 directly (SQLite doesn't support IF NOT EXISTS for columns).
    For PostgreSQL: uses SQLAlchemy inspect to check columns before ALTER TABLE,
    and relies on Base.metadata.create_all for initial schema (called before this function).
    """
    from app.config import DATABASE_URL
    from app.database import _is_sqlite

    if _is_sqlite:
        _migrate_db_sqlite(DATABASE_URL)
    else:
        _migrate_db_postgres()


def _migrate_db_sqlite(database_url: str):
    """SQLite-specific migration using sqlite3 module."""
    import sqlite3

    db_path = database_url.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    migrations = [
        ("prediction_logs", "sector", "TEXT"),
        ("prediction_logs", "regime", "TEXT"),
        ("signal_logs", "sector", "TEXT"),
        ("signal_logs", "regime", "TEXT"),
        ("signal_logs", "oi_score", "REAL"),
        ("signal_logs", "price_after_30min", "REAL"),
        ("signal_logs", "price_after_1hr", "REAL"),
        ("signal_logs", "was_correct_30min", "BOOLEAN"),
        ("signal_logs", "was_correct_1hr", "BOOLEAN"),
        ("users", "telegram_chat_id", "TEXT"),
        ("users", "sms_phone", "TEXT"),
        ("users", "google_id", "TEXT"),
        ("users", "avatar_url", "TEXT"),
        ("users", "auth_provider", "TEXT DEFAULT 'local'"),
    ]

    for table, column, col_type in migrations:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Add indexes for performance (idempotent — IF NOT EXISTS)
    indexes = [
        "CREATE INDEX IF NOT EXISTS ix_portfolio_holdings_user_id ON portfolio_holdings(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_price_alerts_user_id ON price_alerts(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_price_alerts_is_triggered ON price_alerts(is_triggered)",
        "CREATE INDEX IF NOT EXISTS ix_prediction_logs_symbol ON prediction_logs(symbol)",
        "CREATE INDEX IF NOT EXISTS ix_signal_logs_created_at ON signal_logs(created_at)",
        "CREATE INDEX IF NOT EXISTS ix_signal_logs_symbol_created ON signal_logs(symbol, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_job_queue_poll ON job_queue(status, priority, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_job_queue_job_type ON job_queue(job_type)",
        "CREATE INDEX IF NOT EXISTS ix_job_queue_status ON job_queue(status)",
        "CREATE INDEX IF NOT EXISTS ix_telegram_subscriptions_chat_id ON telegram_subscriptions(chat_id)",
        "CREATE INDEX IF NOT EXISTS ix_sms_subscriptions_phone_number ON sms_subscriptions(phone_number)",
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_id ON users(google_id)",
        "CREATE INDEX IF NOT EXISTS ix_smart_alerts_is_triggered ON smart_alerts(is_triggered)",
        "CREATE INDEX IF NOT EXISTS ix_smart_alerts_user_id ON smart_alerts(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_prediction_logs_backfill ON prediction_logs(actual_price, target_date)",
        "CREATE INDEX IF NOT EXISTS ix_prediction_logs_target_date ON prediction_logs(target_date)",
        "CREATE INDEX IF NOT EXISTS ix_watchlist_items_user_id ON watchlist_items(user_id)",
    ]
    for idx_sql in indexes:
        try:
            cursor.execute(idx_sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()


def _migrate_db_postgres():
    """PostgreSQL migration using SQLAlchemy inspect to safely add missing columns."""
    from sqlalchemy import inspect, text
    from app.database import engine

    inspector = inspect(engine)

    migrations = [
        ("prediction_logs", "sector", "TEXT"),
        ("prediction_logs", "regime", "TEXT"),
        ("signal_logs", "sector", "TEXT"),
        ("signal_logs", "regime", "TEXT"),
        ("signal_logs", "oi_score", "DOUBLE PRECISION"),
        ("signal_logs", "price_after_30min", "DOUBLE PRECISION"),
        ("signal_logs", "price_after_1hr", "DOUBLE PRECISION"),
        ("signal_logs", "was_correct_30min", "BOOLEAN"),
        ("signal_logs", "was_correct_1hr", "BOOLEAN"),
        ("users", "telegram_chat_id", "TEXT"),
        ("users", "sms_phone", "TEXT"),
        ("users", "google_id", "TEXT"),
        ("users", "avatar_url", "TEXT"),
        ("users", "auth_provider", "TEXT DEFAULT 'local'"),
    ]

    with engine.begin() as conn:
        for table, column, col_type in migrations:
            if not inspector.has_table(table):
                continue
            existing_cols = {c["name"] for c in inspector.get_columns(table)}
            if column not in existing_cols:
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {column} {col_type}'))

        # Add indexes (IF NOT EXISTS is supported in PostgreSQL 9.5+)
        indexes = [
            "CREATE INDEX IF NOT EXISTS ix_portfolio_holdings_user_id ON portfolio_holdings(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_price_alerts_user_id ON price_alerts(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_price_alerts_is_triggered ON price_alerts(is_triggered)",
            "CREATE INDEX IF NOT EXISTS ix_prediction_logs_symbol ON prediction_logs(symbol)",
            "CREATE INDEX IF NOT EXISTS ix_signal_logs_created_at ON signal_logs(created_at)",
            "CREATE INDEX IF NOT EXISTS ix_signal_logs_symbol_created ON signal_logs(symbol, created_at)",
            "CREATE INDEX IF NOT EXISTS ix_job_queue_poll ON job_queue(status, priority, created_at)",
            "CREATE INDEX IF NOT EXISTS ix_job_queue_job_type ON job_queue(job_type)",
            "CREATE INDEX IF NOT EXISTS ix_job_queue_status ON job_queue(status)",
            "CREATE INDEX IF NOT EXISTS ix_telegram_subscriptions_chat_id ON telegram_subscriptions(chat_id)",
            "CREATE INDEX IF NOT EXISTS ix_sms_subscriptions_phone_number ON sms_subscriptions(phone_number)",
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_id ON users(google_id)",
            "CREATE INDEX IF NOT EXISTS ix_smart_alerts_is_triggered ON smart_alerts(is_triggered)",
            "CREATE INDEX IF NOT EXISTS ix_smart_alerts_user_id ON smart_alerts(user_id)",
            "CREATE INDEX IF NOT EXISTS ix_prediction_logs_backfill ON prediction_logs(actual_price, target_date)",
            "CREATE INDEX IF NOT EXISTS ix_prediction_logs_target_date ON prediction_logs(target_date)",
            "CREATE INDEX IF NOT EXISTS ix_watchlist_items_user_id ON watchlist_items(user_id)",
        ]
        for idx_sql in indexes:
            try:
                conn.execute(text(idx_sql))
            except Exception:
                pass  # Table may not exist yet


async def prediction_accuracy_backfiller():
    """Background task: fill in actual_price for predictions whose target_date has passed."""
    from datetime import date
    from app.database import SessionLocal
    from app.models import PredictionLog

    await asyncio.sleep(180)  # Let services warm up
    while True:
        try:
            def _backfill():
                db = SessionLocal()
                try:
                    today = date.today()
                    # Find predictions where target_date <= today and actual_price is NULL
                    pending = (
                        db.query(PredictionLog)
                        .filter(
                            PredictionLog.actual_price.is_(None),
                            PredictionLog.target_date <= today,
                        )
                        .limit(50)
                        .all()
                    )
                    if not pending:
                        return 0

                    # Group by symbol to batch quote fetches
                    symbols = list({p.symbol for p in pending})
                    from app.services.data_fetcher import data_fetcher
                    quotes = {}
                    for sym in symbols:
                        try:
                            q = data_fetcher.get_live_quote(sym)
                            if q and q.get("ltp"):
                                quotes[sym] = float(q["ltp"])
                        except Exception:
                            pass

                    updated = 0
                    for p in pending:
                        if p.symbol in quotes:
                            p.actual_price = quotes[p.symbol]
                            updated += 1

                    if updated:
                        db.commit()
                    return updated
                except Exception as e:
                    db.rollback()
                    logging.getLogger(__name__).debug(f"Prediction backfill error: {e}")
                    return 0
                finally:
                    db.close()

            count = await asyncio.to_thread(_backfill)
            if count > 0:
                logging.getLogger(__name__).info(f"Backfilled {count} prediction actual prices")
        except Exception:
            pass
        await asyncio.sleep(3600)  # Run every hour


async def periodic_job_enqueuer():
    """Every 5 min, enqueue background jobs (watchlist_signals, mtf_stream, oi_stream)
    if market is open and no duplicate pending jobs exist."""
    from app.services.job_service import job_service
    from app.utils.helpers import is_market_open
    from app.routers.websocket import manager

    await asyncio.sleep(120)  # Let services warm up
    while True:
        try:
            if is_market_open():
                def _enqueue_jobs():
                    if not job_service.has_pending("watchlist_signals"):
                        job_service.enqueue("watchlist_signals", {}, priority=0)
                    if manager.active_connections:
                        symbols = list(manager.get_all_subscribed_symbols())
                        if symbols:
                            if not job_service.has_pending("mtf_stream"):
                                job_service.enqueue("mtf_stream", {"symbols": symbols}, priority=0)
                            if not job_service.has_pending("oi_stream"):
                                job_service.enqueue("oi_stream", {"symbols": symbols}, priority=0)

                await asyncio.to_thread(_enqueue_jobs)
        except Exception:
            pass
        await asyncio.sleep(300)


async def worker_result_broadcaster():
    """Every 3s, read completed background jobs and broadcast results via WebSocket."""
    from app.services.job_service import job_service
    from app.routers.websocket import manager

    await asyncio.sleep(10)
    while True:
        try:
            if manager.active_connections:
                jobs = await asyncio.to_thread(job_service.get_completed_background_jobs)
                for job in jobs:
                    job_type = job["job_type"]
                    result = job["result"]

                    if job_type == "watchlist_signals":
                        signals_data = result.get("signals", {})
                        for sym, signal_data in signals_data.items():
                            await manager.broadcast_signal(sym, signal_data)
                            # Telegram: broadcast high-confidence signals
                            confidence = signal_data.get("confidence", 0)
                            if confidence >= 60:
                                from app.services.telegram_service import broadcast_to_subscribers, send_signal_alert as _tg_signal
                                tg_data = {**signal_data, "symbol": sym}
                                asyncio.create_task(broadcast_to_subscribers("signals", _tg_signal, tg_data))
                                # SMS: broadcast high-confidence signals
                                from app.services.sms_service import broadcast_to_subscribers as sms_broadcast, send_signal_alert as _sms_signal
                                asyncio.create_task(sms_broadcast("signals", _sms_signal, tg_data))

                    elif job_type == "mtf_stream":
                        mtf_data = result.get("mtf_data", {})
                        for sym, data in mtf_data.items():
                            await manager.broadcast_mtf_update(sym, data)

                    elif job_type == "oi_stream":
                        oi_data = result.get("oi_data", {})
                        for sym, data in oi_data.items():
                            await manager.broadcast_oi_update(sym, data)

                    await asyncio.to_thread(job_service.mark_broadcast, job["id"])
        except Exception:
            pass
        await asyncio.sleep(3)


async def news_alert_scanner():
    """Monitor watchlist stocks for significant sentiment changes and broadcast alerts."""
    from app.database import SessionLocal
    from app.models import WatchlistItem

    await asyncio.sleep(300)  # Let services warm up
    _prev_scores = {}  # symbol -> last known score

    while True:
        try:
            from app.utils.helpers import is_market_open
            from app.routers.websocket import manager

            if is_market_open() and manager.active_connections:
                def _check_news():
                    db = SessionLocal()
                    try:
                        # Get all watchlist symbols
                        items = db.query(WatchlistItem.symbol).distinct().all()
                        return [i.symbol for i in items]
                    finally:
                        db.close()

                symbols = await asyncio.to_thread(_check_news)
                if symbols:
                    from app.services.sentiment_service import sentiment_service

                    for symbol in symbols[:15]:  # Limit to avoid rate limiting
                        try:
                            sentiment = await asyncio.to_thread(sentiment_service.get_sentiment, symbol)
                            if not sentiment:
                                continue

                            score = sentiment.get("score", 0)
                            prev = _prev_scores.get(symbol, 0)
                            _prev_scores[symbol] = score

                            # Alert if score changed significantly (>30 points) or is extreme (>60)
                            change = abs(score - prev)
                            if prev != 0 and change >= 30:
                                alert_data = {
                                    "symbol": symbol,
                                    "type": "news_sentiment_change",
                                    "score": round(score, 1),
                                    "previous_score": round(prev, 1),
                                    "change": round(score - prev, 1),
                                    "headline_count": sentiment.get("headline_count", 0),
                                    "top_headline": sentiment["headlines"][0]["title"] if sentiment.get("headlines") else None,
                                }
                                await manager.broadcast_to_all("news_alert", alert_data)
                                # Fire-and-forget Telegram broadcast
                                from app.services.telegram_service import broadcast_to_subscribers, send_news_alert as _tg_news
                                asyncio.create_task(broadcast_to_subscribers("news", _tg_news, alert_data))
                                # Fire-and-forget SMS broadcast
                                from app.services.sms_service import broadcast_to_subscribers as sms_broadcast, send_news_alert as _sms_news
                                asyncio.create_task(sms_broadcast("news", _sms_news, alert_data))
                            elif abs(score) >= 60 and prev == 0:
                                # First check found extreme sentiment
                                alert_data = {
                                    "symbol": symbol,
                                    "type": "news_extreme_sentiment",
                                    "score": round(score, 1),
                                    "headline_count": sentiment.get("headline_count", 0),
                                    "top_headline": sentiment["headlines"][0]["title"] if sentiment.get("headlines") else None,
                                }
                                await manager.broadcast_to_all("news_alert", alert_data)
                                from app.services.telegram_service import broadcast_to_subscribers, send_news_alert as _tg_news2
                                asyncio.create_task(broadcast_to_subscribers("news", _tg_news2, alert_data))
                                from app.services.sms_service import broadcast_to_subscribers as sms_broadcast2, send_news_alert as _sms_news2
                                asyncio.create_task(sms_broadcast2("news", _sms_news2, alert_data))
                        except Exception:
                            pass
                        await asyncio.sleep(2)  # Pace requests
        except Exception:
            pass
        await asyncio.sleep(600)  # Run every 10 minutes


async def watchlist_trade_scanner():
    """Auto-compute MTF signals for watchlist stocks and log trade predictions.
    Runs twice daily (10:30 AM, 2:30 PM IST) to avoid overloading the server.
    Processes stocks sequentially with 10s delays between each."""
    await asyncio.sleep(300)  # Let services warm up

    _ran_today = set()  # Track which windows we've run today

    while True:
        try:
            from app.utils.helpers import is_market_open, now_ist
            current = now_ist()
            hour, minute = current.hour, current.minute

            # Run at 10:30 and 14:30 — two windows per day
            window = None
            if hour == 10 and 28 <= minute <= 35:
                window = f"{current.date()}_morning"
            elif hour == 14 and 28 <= minute <= 35:
                window = f"{current.date()}_afternoon"

            if window and window not in _ran_today and is_market_open():
                _ran_today.add(window)
                # Clean old entries (keep last 2 days)
                _ran_today = {w for w in _ran_today if current.date().isoformat() in w}

                # Get watchlist symbols
                from app.database import SessionLocal
                from app.models import WatchlistItem
                db = SessionLocal()
                try:
                    symbols = [item.symbol for item in db.query(WatchlistItem.symbol).distinct().all()]
                finally:
                    db.close()

                if symbols:
                    from app.services.signal_service import signal_service
                    log = logging.getLogger(__name__)
                    log.info(f"Trade scanner: processing {len(symbols)} watchlist stocks")

                    for i, symbol in enumerate(symbols):
                        try:
                            # Sequential + delay to avoid CPU/memory spike
                            await asyncio.to_thread(
                                signal_service.get_multi_timeframe_signals, symbol
                            )
                            log.debug(f"Trade scanner: {symbol} ({i+1}/{len(symbols)})")
                        except Exception as e:
                            log.debug(f"Trade scanner: {symbol} failed: {e}")

                        # 10s delay between stocks — gentle on the server
                        await asyncio.sleep(10)

                    log.info(f"Trade scanner: completed {len(symbols)} stocks")

        except Exception as e:
            logging.getLogger(__name__).debug(f"Trade scanner error: {e}")

        await asyncio.sleep(120)  # Check every 2 minutes


async def trade_prediction_validator():
    """Background task: validate open trade predictions every 5 minutes during market hours."""
    await asyncio.sleep(120)  # Let services warm up
    while True:
        try:
            from app.utils.helpers import is_market_open
            if is_market_open():
                from app.services.trade_tracker import trade_tracker
                result = await asyncio.to_thread(trade_tracker.validate_open_signals)
                if result and result.get("resolved", 0) > 0:
                    logging.getLogger(__name__).info(
                        f"Trade validator: checked {result['checked']}, resolved {result['resolved']}"
                    )
                    # Learn from resolved trades
                    await asyncio.to_thread(trade_tracker.learn_from_trades)
        except Exception:
            pass
        await asyncio.sleep(300)  # Every 5 minutes


async def daily_stock_learner():
    """Daily learning task: rebuild per-stock profiles after market close.
    Analyzes signal accuracy for each stock and learns optimal weights/thresholds."""
    await asyncio.sleep(600)  # Let services warm up

    while True:
        try:
            from app.utils.helpers import is_market_open, now_ist
            current = now_ist()
            # Run at 4:15 PM IST (after market close at 3:30 PM)
            if current.hour == 16 and 15 <= current.minute <= 20:
                from app.services.stock_learner import stock_learner
                result = await asyncio.to_thread(stock_learner.rebuild_all_profiles)
                if result:
                    logging.getLogger(__name__).info(
                        f"Daily stock learning complete: {result['profiles_built']} profiles, "
                        f"{len(result['improvements'])} improved, "
                        f"{len(result['degradations'])} degraded"
                    )
                    # Broadcast learning summary via WebSocket
                    try:
                        from app.routers.websocket import manager
                        if manager.active_connections:
                            await manager.broadcast_to_all("learning_update", {
                                "profiles_built": result["profiles_built"],
                                "improvements": result["improvements"][:5],
                                "degradations": result["degradations"][:5],
                            })
                    except Exception:
                        pass
                # Sleep until next day (avoid running multiple times)
                await asyncio.sleep(3600)
            else:
                await asyncio.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logging.getLogger(__name__).debug(f"Daily learner error: {e}")
            await asyncio.sleep(600)


async def live_scanner():
    """Run screener filters every 5 minutes and broadcast new matches via WebSocket."""
    await asyncio.sleep(120)
    _prev_matches = set()

    while True:
        try:
            from app.utils.helpers import is_market_open
            from app.routers.websocket import manager

            if is_market_open() and manager.active_connections:
                def _scan():
                    from app.services.screener_service import screener_service
                    # Scan for high-confidence setups
                    filters = {
                        "rsi_oversold": True,
                        "volume_spike": True,
                    }
                    results1 = screener_service.scan(filters)

                    filters2 = {
                        "rsi_overbought": True,
                        "volume_spike": True,
                    }
                    results2 = screener_service.scan(filters2)

                    filters3 = {
                        "macd_bullish": True,
                        "volume_spike": True,
                    }
                    results3 = screener_service.scan(filters3)

                    # Merge and deduplicate
                    seen = set()
                    all_results = []
                    for r in results1 + results2 + results3:
                        if r["symbol"] not in seen:
                            seen.add(r["symbol"])
                            all_results.append(r)
                    return all_results

                results = await asyncio.to_thread(_scan)
                current_matches = {r["symbol"] for r in results}

                # Only broadcast NEW matches (not already seen)
                new_matches = current_matches - _prev_matches
                _prev_matches = current_matches

                for result in results:
                    if result["symbol"] in new_matches:
                        alert_data = {
                            "symbol": result["symbol"],
                            "ltp": result.get("ltp"),
                            "change_pct": result.get("change_pct"),
                            "rsi": result.get("rsi"),
                            "volume_ratio": result.get("volume_ratio"),
                            "matched_filters": result.get("matched_filters", []),
                        }
                        await manager.broadcast_to_all("scanner_alert", alert_data)
                        # Fire-and-forget Telegram broadcast
                        from app.services.telegram_service import broadcast_to_subscribers, send_scanner_alert as _tg_scanner
                        asyncio.create_task(broadcast_to_subscribers("scanner", _tg_scanner, alert_data))
                        # Fire-and-forget SMS broadcast
                        from app.services.sms_service import broadcast_to_subscribers as sms_broadcast, send_scanner_alert as _sms_scanner
                        asyncio.create_task(sms_broadcast("scanner", _sms_scanner, alert_data))

        except Exception:
            pass
        await asyncio.sleep(300)  # Every 5 minutes


def _setup_audit_logger():
    """Configure audit logger to write to logs/audit.log."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, "audit.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    audit = logging.getLogger("audit")
    audit.setLevel(logging.INFO)
    audit.addHandler(handler)
    audit.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    _setup_audit_logger()
    Base.metadata.create_all(bind=engine)
    _migrate_db()
    tasks = [
        asyncio.create_task(price_streamer()),
        asyncio.create_task(alert_checker()),
        asyncio.create_task(signal_accuracy_validator()),
        asyncio.create_task(signal_accuracy_validator_30min()),
        asyncio.create_task(signal_accuracy_validator_1hr()),
        asyncio.create_task(smart_alert_checker()),
        asyncio.create_task(market_mood_broadcaster()),
        asyncio.create_task(periodic_job_enqueuer()),
        asyncio.create_task(worker_result_broadcaster()),
        asyncio.create_task(prediction_accuracy_backfiller()),
        asyncio.create_task(news_alert_scanner()),
        asyncio.create_task(live_scanner()),
        asyncio.create_task(daily_stock_learner()),
        asyncio.create_task(trade_prediction_validator()),
        asyncio.create_task(watchlist_trade_scanner()),
    ]
    yield
    # Shutdown
    for t in tasks:
        t.cancel()


app = FastAPI(title="Indian Stock Market Tracker & AI Predictor", lifespan=lifespan)

# GZip compression for responses > 500 bytes (significant savings for JSON API & HTML)
app.add_middleware(GZipMiddleware, minimum_size=500)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_TIMEOUT = 120  # seconds — global safety net for all HTTP requests

# HTTP API rate limiting — configurable via RATE_LIMIT_RPM env var
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
_rate_limiter = RateLimiter(max_requests=RATE_LIMIT_RPM, window_seconds=60)

# Allowed origins for CSRF protection (state-changing requests must have valid Origin)
_ALLOWED_ORIGINS = set(CORS_ORIGINS) | {"null"}  # "null" for same-origin requests without Origin header
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


@app.middleware("http")
async def cache_control_middleware(request: Request, call_next):
    """Set Cache-Control headers: long cache for versioned static assets, no-cache for HTML."""
    response: Response = await call_next(request)
    path = request.url.path

    # Versioned static assets (contain ?v=) — cache for 1 year (immutable)
    if path.startswith("/static/") and "v=" in str(request.url.query):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    # Non-versioned static assets — cache for 1 day with revalidation
    elif path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=86400, stale-while-revalidate=3600"
    # HTML pages — always revalidate
    elif not path.startswith("/api/") and not path.startswith("/ws/") and path != "/health":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"

    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate-limit HTTP API requests per IP (skip WebSocket upgrades)."""
    from fastapi.responses import JSONResponse
    if (request.url.path.startswith("/api/")
            and request.headers.get("upgrade", "").lower() != "websocket"):
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.check_rate_limit(client_ip):
            retry_after = _rate_limiter.time_until_available(client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """Reject state-changing requests from unknown origins."""
    from fastapi.responses import JSONResponse
    if request.method not in _SAFE_METHODS and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin and origin not in _ALLOWED_ORIGINS:
            # Allow same-origin: compare Origin against the request's own Host
            host = request.headers.get("host", "")
            scheme = "https" if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https" else "http"
            self_origin = f"{scheme}://{host}"
            if origin != self_origin:
                return JSONResponse(status_code=403, content={"detail": "Origin not allowed"})
    return await call_next(request)


audit_logger = logging.getLogger("audit")
_AUDIT_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    """Log state-changing API requests with user identity and outcome."""
    start = time.monotonic()
    response = await call_next(request)

    if request.method in _AUDIT_METHODS and request.url.path.startswith("/api/"):
        duration_ms = round((time.monotonic() - start) * 1000)
        # Extract user from Authorization header (without full decode — just for logging)
        auth_header = request.headers.get("authorization", "")
        user_hint = "anonymous"
        if auth_header.startswith("Bearer "):
            try:
                from jose import jwt as jose_jwt
                from app.config import SECRET_KEY, ALGORITHM
                payload = jose_jwt.decode(auth_header[7:], SECRET_KEY, algorithms=[ALGORITHM])
                user_hint = payload.get("sub", "unknown")
            except Exception:
                user_hint = "invalid-token"
        elif request.headers.get("x-api-key"):
            user_hint = "api-key"

        client_ip = request.client.host if request.client else "unknown"
        audit_logger.info(
            f"{request.method} {request.url.path} | user={user_hint} ip={client_ip} "
            f"status={response.status_code} duration={duration_ms}ms"
        )

    return response


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=504, content={"detail": "Request timed out"})

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API Routers
app.include_router(stocks.router, prefix="/api/stocks", tags=["stocks"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(indicators.router, prefix="/api/indicators", tags=["indicators"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(watchlist.router, prefix="/api/watchlist", tags=["watchlist"])
app.include_router(fii_dii_router, prefix="/api/fii-dii", tags=["fii-dii"])
app.include_router(sectors_router, prefix="/api/sectors", tags=["sectors"])
app.include_router(jobs_router, prefix="/api/jobs", tags=["jobs"])
app.include_router(screener.router, prefix="/api/screener", tags=["screener"])
app.include_router(options.router, prefix="/api/options", tags=["options"])
app.include_router(mtf_dashboard_router, prefix="/api/mtf", tags=["mtf-dashboard"])
app.include_router(trade_journal_router, prefix="/api/journal", tags=["journal"])
app.include_router(strategies_router, prefix="/api/strategies", tags=["strategies"])
app.include_router(broker_router, prefix="/api/broker", tags=["broker"])
app.include_router(telegram_router, prefix="/api/telegram", tags=["telegram"])
app.include_router(sms_router, prefix="/api/sms", tags=["sms"])
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root(request: Request):
    from app.config import GOOGLE_CLIENT_ID
    return templates.TemplateResponse("index.html", {"request": request, "google_client_id": GOOGLE_CLIENT_ID})
