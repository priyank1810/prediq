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
                        # Telegram price alerts disabled — signals only
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
        # V1 vs V2 shadow tracking
        ("trade_signal_logs", "model_used", "TEXT"),
        ("trade_signal_logs", "v1_predicted_price", "REAL"),
        ("trade_signal_logs", "v1_confidence", "REAL"),
        ("trade_signal_logs", "v2_predicted_price", "REAL"),
        ("trade_signal_logs", "v2_confidence", "REAL"),
        ("trade_signal_logs", "v2_direction", "TEXT"),
        # Timeframe checkpoint
        ("trade_signal_logs", "check_at", "TIMESTAMP"),
        # Confidence trend tracking
        ("trade_signal_logs", "confidence_trend", "TEXT"),
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
        # V1 vs V2 shadow tracking
        ("trade_signal_logs", "model_used", "TEXT"),
        ("trade_signal_logs", "v1_predicted_price", "DOUBLE PRECISION"),
        ("trade_signal_logs", "v1_confidence", "DOUBLE PRECISION"),
        ("trade_signal_logs", "v2_predicted_price", "DOUBLE PRECISION"),
        ("trade_signal_logs", "v2_confidence", "DOUBLE PRECISION"),
        ("trade_signal_logs", "v2_direction", "TEXT"),
        # Timeframe checkpoint
        ("trade_signal_logs", "check_at", "TIMESTAMP"),
        # Confidence trend tracking
        ("trade_signal_logs", "confidence_trend", "TEXT"),
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
                                # Telegram news alerts disabled — signals only
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
                                # Telegram news alerts disabled — signals only
                                from app.services.sms_service import broadcast_to_subscribers as sms_broadcast2, send_news_alert as _sms_news2
                                asyncio.create_task(sms_broadcast2("news", _sms_news2, alert_data))
                        except Exception:
                            pass
                        await asyncio.sleep(2)  # Pace requests
        except Exception:
            pass
        await asyncio.sleep(600)  # Run every 10 minutes


async def trade_job_enqueuer():
    """Tiered trade scanning + validation.

    Intraday (15m, 30m):  every 10 min during market hours (~193 stocks)
    Short-term (1h, 4h):  every 20 min during market hours
    Validation:           every 5 min (real-time via ticks + worker fallback)
    """
    await asyncio.sleep(300)  # Let services warm up

    _last_intraday_scan = 0
    _last_shortterm_scan = 0

    while True:
        try:
            from app.utils.helpers import is_market_open, now_ist
            from app.services.job_service import job_service
            import time

            current = now_ist()
            now_ts = time.time()

            if is_market_open():
                # Release stuck jobs so scans don't get blocked forever
                try:
                    job_service.release_stale_jobs(timeout_minutes=10)
                except Exception:
                    pass

                # Validate open trades every 5 min (fallback for missed ticks)
                if not job_service.has_pending("trade_validate"):
                    job_service.enqueue("trade_validate", {}, priority=0)

                # Only enqueue one scan type per cycle — short-term takes priority
                # when both are due, since it runs less frequently
                if not job_service.has_pending("watchlist_trade_scan"):
                    if now_ts - _last_shortterm_scan >= 1200:  # 20 min
                        job_service.enqueue("watchlist_trade_scan", {"scan_type": "short"}, priority=0)
                        _last_shortterm_scan = now_ts
                    elif now_ts - _last_intraday_scan >= 600:  # 10 min
                        job_service.enqueue("watchlist_trade_scan", {"scan_type": "intraday"}, priority=0)
                        _last_intraday_scan = now_ts

        except Exception:
            pass

        await asyncio.sleep(180)  # Check every 3 minutes


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


async def eod_cleanup_unresolved_trades():
    """End-of-day cleanup: delete any open trades that didn't resolve before market close.
    Runs at 3:35 PM IST (5 min after market close at 3:30 PM).
    Trades shouldn't carry across days because intraday/short-term setups become stale."""
    await asyncio.sleep(120)

    while True:
        try:
            from app.utils.helpers import now_ist
            current = now_ist()
            # Run at 3:35 PM IST (after market close at 3:30 PM)
            if current.hour == 15 and 35 <= current.minute <= 40:
                from app.database import SessionLocal
                from app.models import TradeSignalLog

                def _cleanup():
                    db = SessionLocal()
                    try:
                        deleted = db.query(TradeSignalLog).filter(
                            TradeSignalLog.status == "open"
                        ).delete(synchronize_session=False)
                        db.commit()
                        return deleted
                    finally:
                        db.close()

                deleted = await asyncio.to_thread(_cleanup)
                if deleted > 0:
                    logging.getLogger(__name__).info(
                        f"EOD cleanup: deleted {deleted} unresolved open trades"
                    )
                    # Refresh trade tracker cache
                    try:
                        from app.services.trade_tracker import trade_tracker
                        trade_tracker._cache_loaded = False
                    except Exception:
                        pass
                # Sleep until next day to avoid duplicate runs
                await asyncio.sleep(3600)
            else:
                await asyncio.sleep(300)  # Check every 5 min
        except Exception as e:
            logging.getLogger(__name__).debug(f"EOD cleanup error: {e}")
            await asyncio.sleep(600)


async def weekly_analysis_report():
    """Friday 5 PM IST: automated weekly analysis with recommendations."""
    await asyncio.sleep(900)

    while True:
        try:
            from app.utils.helpers import now_ist
            current = now_ist()

            # Run on Friday at 5 PM IST
            if current.weekday() == 4 and current.hour == 17 and 0 <= current.minute <= 10:
                from app.database import SessionLocal
                from app.models import TradeSignalLog
                from collections import defaultdict

                db = SessionLocal()
                try:
                    resolved = db.query(TradeSignalLog).filter(
                        TradeSignalLog.status != "open",
                        TradeSignalLog.confidence >= 40,
                    ).all()

                    if len(resolved) < 20:
                        await asyncio.sleep(3600)
                        continue

                    total = len(resolved)
                    wins = sum(1 for r in resolved if r.status in ("target_hit", "correct"))
                    wr = round(wins / total * 100, 1)

                    # By timeframe
                    tf_stats = {}
                    for r in resolved:
                        tf = r.timeframe or "unknown"
                        if tf not in tf_stats:
                            tf_stats[tf] = {"wins": 0, "total": 0, "pnl": 0}
                        tf_stats[tf]["total"] += 1
                        if r.status in ("target_hit", "correct"):
                            tf_stats[tf]["wins"] += 1
                        if r.outcome_pct:
                            tf_stats[tf]["pnl"] += r.outcome_pct

                    # By confidence
                    conf_stats = {}
                    for lo, hi in [(40, 50), (50, 60), (60, 80), (80, 100)]:
                        bucket = [r for r in resolved if r.confidence and lo <= r.confidence < hi]
                        if bucket:
                            b_wins = sum(1 for r in bucket if r.status in ("target_hit", "correct"))
                            conf_stats[f"{lo}-{hi}%"] = {
                                "total": len(bucket),
                                "wr": round(b_wins / len(bucket) * 100, 1),
                            }

                    # By hour
                    hour_stats = {}
                    for h in range(9, 16):
                        h_trades = [r for r in resolved if r.created_at and r.created_at.hour == h]
                        if h_trades:
                            h_wins = sum(1 for r in h_trades if r.status in ("target_hit", "correct"))
                            hour_stats[h] = {
                                "total": len(h_trades),
                                "wr": round(h_wins / len(h_trades) * 100, 1),
                            }

                    # By direction
                    bull = [r for r in resolved if r.direction == "BULLISH"]
                    bear = [r for r in resolved if r.direction == "BEARISH"]
                    bull_wr = round(sum(1 for r in bull if r.status in ("target_hit", "correct")) / max(len(bull), 1) * 100, 1)
                    bear_wr = round(sum(1 for r in bear if r.status in ("target_hit", "correct")) / max(len(bear), 1) * 100, 1)

                    # Build recommendations
                    recs = []

                    # TF recommendations
                    best_tf = max(tf_stats.items(), key=lambda x: x[1]["wins"] / max(x[1]["total"], 1))
                    worst_tf = min(tf_stats.items(), key=lambda x: x[1]["wins"] / max(x[1]["total"], 1))
                    if worst_tf[1]["total"] >= 10:
                        worst_wr = worst_tf[1]["wins"] / worst_tf[1]["total"] * 100
                        if worst_wr < 45:
                            recs.append(f"⚠️ Consider dropping {worst_tf[0]} ({worst_wr:.0f}% WR)")
                    recs.append(f"✅ Best timeframe: {best_tf[0]} ({best_tf[1]['wins'] / best_tf[1]['total'] * 100:.0f}% WR)")

                    # Hour recommendations
                    if hour_stats:
                        worst_hour = min(hour_stats.items(), key=lambda x: x[1]["wr"])
                        best_hour = max(hour_stats.items(), key=lambda x: x[1]["wr"])
                        if worst_hour[1]["wr"] < 35 and worst_hour[1]["total"] >= 10:
                            recs.append(f"⚠️ Avoid trading at {worst_hour[0]}:00 ({worst_hour[1]['wr']}% WR)")
                        recs.append(f"✅ Best time: {best_hour[0]}:00 ({best_hour[1]['wr']}% WR)")

                    # Direction
                    if bull_wr < 40 and len(bull) >= 10:
                        recs.append(f"⚠️ Bullish signals weak ({bull_wr}% WR) — market may be bearish")
                    if bear_wr > 65 and len(bear) >= 10:
                        recs.append(f"✅ Bearish signals strong ({bear_wr}% WR)")

                    # Build report
                    report = f"""
📊 <b>Weekly AI Analysis — {current.strftime('%d %b %Y')}</b>

<b>Overall: {wr}% Win Rate ({total} trades)</b>
Bullish: {bull_wr}% ({len(bull)} trades)
Bearish: {bear_wr}% ({len(bear)} trades)

<b>By Timeframe:</b>"""
                    for tf, s in sorted(tf_stats.items(), key=lambda x: -x[1]["wins"] / max(x[1]["total"], 1)):
                        tf_wr = round(s["wins"] / s["total"] * 100, 1) if s["total"] > 0 else 0
                        emoji = "🟢" if tf_wr >= 60 else "🟡" if tf_wr >= 45 else "🔴"
                        report += f"\n{emoji} {tf}: {tf_wr}% WR ({s['total']} trades)"

                    report += "\n\n<b>By Confidence:</b>"
                    for conf, s in conf_stats.items():
                        emoji = "🟢" if s["wr"] >= 60 else "🟡" if s["wr"] >= 45 else "🔴"
                        report += f"\n{emoji} {conf}: {s['wr']}% WR ({s['total']} trades)"

                    report += "\n\n<b>Recommendations:</b>"
                    for rec in recs:
                        report += f"\n{rec}"

                    report += "\n\n🔗 prediq.duckdns.org/#analysis"

                    # Send via Telegram
                    from app.services.telegram_service import send_message
                    from app.models import User
                    users = db.query(User).filter(User.telegram_chat_id.isnot(None)).all()
                    for user in users:
                        await send_message(user.telegram_chat_id, report.strip())

                    logging.getLogger(__name__).info(f"Weekly analysis sent to {len(users)} users")
                finally:
                    db.close()

                await asyncio.sleep(86400)  # Sleep 1 day
            else:
                await asyncio.sleep(300)
        except Exception as e:
            logging.getLogger(__name__).debug(f"Weekly analysis error: {e}")
            await asyncio.sleep(600)


async def daily_telegram_report():
    """Send daily portfolio + track record summary via Telegram at 4:30 PM IST."""
    await asyncio.sleep(900)

    _sent_today = None

    while True:
        try:
            from app.utils.helpers import now_ist
            current = now_ist()

            if current.hour == 16 and 28 <= current.minute <= 35 and _sent_today != current.date():
                _sent_today = current.date()

                from app.services.virtual_portfolio import virtual_portfolio
                from app.services.trade_tracker import trade_tracker

                # Get portfolio data
                pf = virtual_portfolio.get_portfolio()
                stats = trade_tracker.get_accuracy_stats()

                # Build report
                pnl = pf.get("total_pnl", 0)
                pnl_pct = pf.get("total_pnl_pct", 0)
                wr = pf.get("win_rate", 0)
                trades = pf.get("total_trades", 0)
                open_pos = len(pf.get("open_positions", []))
                current_val = pf.get("current_value", 100000)

                # Track record
                total_signals = stats.get("total", 0)
                signal_wr = stats.get("win_rate", 0)
                direction_acc = stats.get("direction_accuracy", 0)
                pred_err = stats.get("avg_prediction_error", 0)

                sign = "+" if pnl >= 0 else ""
                pnl_emoji = "📈" if pnl >= 0 else "📉"

                report = f"""
{pnl_emoji} <b>Daily Report — {current.strftime('%d %b %Y')}</b>

<b>💰 Virtual Portfolio</b>
Value: ₹{current_val:,.0f} ({sign}{pnl_pct:.1f}%)
P&L: {sign}₹{abs(pnl):,.0f}
Win Rate: {wr:.0f}% | Trades: {trades}
Open Positions: {open_pos}

<b>🎯 Track Record</b>
Signals Tracked: {total_signals}
Signal Win Rate: {signal_wr:.0f}%
Direction Accuracy: {direction_acc:.0f}%
Avg Prediction Error: {pred_err:.2f}%

<b>📊 Top Performers</b>"""

                # Add top 3 stocks
                for s in (pf.get("stock_summary") or [])[:3]:
                    report += f"\n• {s['symbol']}: {s['win_rate']:.0f}% WR ({s['trades']} trades)"

                # Add best trade
                best = pf.get("best_trade")
                if best:
                    report += f"\n\n🏆 Best: {best['symbol']} +₹{abs(best['pnl']):.0f} (+{best['pnl_pct']}%)"

                report += "\n\n🔗 prediq.duckdns.org"

                # Send to all telegram subscribers
                from app.services.telegram_service import broadcast_to_subscribers, send_message
                from app.database import SessionLocal
                from app.models import User

                db = SessionLocal()
                try:
                    users = db.query(User).filter(User.telegram_chat_id.isnot(None)).all()
                    for user in users:
                        await send_message(user.telegram_chat_id, report.strip())
                    if users:
                        logging.getLogger(__name__).info(f"Daily report sent to {len(users)} users")
                finally:
                    db.close()

                await asyncio.sleep(3600)
            else:
                await asyncio.sleep(300)
        except Exception as e:
            logging.getLogger(__name__).debug(f"Daily report error: {e}")
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
                        # Telegram scanner alerts disabled — signals only
                        # Fire-and-forget SMS broadcast
                        from app.services.sms_service import broadcast_to_subscribers as sms_broadcast, send_scanner_alert as _sms_scanner
                        asyncio.create_task(sms_broadcast("scanner", _sms_scanner, alert_data))

        except Exception:
            pass
        await asyncio.sleep(300)  # Every 5 minutes


async def cache_cleanup():
    """Periodically purge expired cache entries to free memory."""
    from app.utils.cache import cache
    await asyncio.sleep(120)
    while True:
        try:
            if hasattr(cache, "purge_expired"):
                cache.purge_expired()
        except Exception:
            pass
        await asyncio.sleep(600)  # Every 10 minutes


def _setup_audit_logger():
    """Configure audit logger to write to logs/audit.log with rotation."""
    from logging.handlers import RotatingFileHandler
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = RotatingFileHandler(
        os.path.join(log_dir, "audit.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
    )
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
        asyncio.create_task(prediction_accuracy_backfiller()),
        asyncio.create_task(news_alert_scanner()),
        asyncio.create_task(live_scanner()),
        asyncio.create_task(daily_stock_learner()),
        asyncio.create_task(eod_cleanup_unresolved_trades()),
        asyncio.create_task(daily_telegram_report()),
        asyncio.create_task(weekly_analysis_report()),
        asyncio.create_task(trade_job_enqueuer()),
        asyncio.create_task(cache_cleanup()),
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
