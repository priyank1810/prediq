"""Trade prediction tracker — logs, validates, and learns from entry/target/SL predictions.

Flow:
1. Log: Every MTF signal computation logs a trade prediction
2. Validate: Background task checks if price hit target or stop-loss
3. Learn: Feeds win/loss data back to improve per-stock profiles
"""

import logging
from datetime import datetime, timedelta

from app.database import SessionLocal
from app.models import TradeSignalLog
from app.utils.helpers import now_ist

logger = logging.getLogger(__name__)

# Timeframe → window duration (how long before a signal "expires")
TIMEFRAME_WINDOWS = {
    "intraday_10m": timedelta(minutes=30),
    "intraday_15m": timedelta(minutes=45),
    "intraday_30m": timedelta(hours=2),
    "short_1h": timedelta(hours=4),
    "short_4h": timedelta(hours=8),
}

# Minimum minutes between logging same symbol+timeframe (prevent duplicates)
MIN_LOG_INTERVAL_MINUTES = {
    "intraday_10m": 15,
    "intraday_15m": 20,
    "intraday_30m": 30,
    "short_1h": 60,
    "short_4h": 240,
}


class TradeTracker:
    """Tracks trade signal predictions and validates outcomes."""

    def __init__(self):
        self._open_trades_cache = {}  # symbol -> [trade dicts]
        self._cache_loaded = False

    def _load_open_trades_cache(self):
        """Load all open trades into memory for fast tick checking."""
        db = SessionLocal()
        try:
            open_trades = (
                db.query(TradeSignalLog)
                .filter(TradeSignalLog.status == "open")
                .all()
            )
            self._open_trades_cache = {}
            for t in open_trades:
                if t.symbol not in self._open_trades_cache:
                    self._open_trades_cache[t.symbol] = []
                self._open_trades_cache[t.symbol].append({
                    "id": t.id, "direction": t.direction,
                    "entry": t.entry, "target": t.target,
                    "stop_loss": t.stop_loss, "timeframe": t.timeframe,
                    "expires_at": t.expires_at,
                })
            self._cache_loaded = True
        finally:
            db.close()

    def check_symbol_tick(self, symbol: str, ltp: float, high: float = 0, low: float = 0):
        """Called on every price tick — real-time trade validation.
        Uses in-memory cache, only hits DB when a trade resolves."""
        if not self._cache_loaded:
            self._load_open_trades_cache()

        trades = self._open_trades_cache.get(symbol)
        if not trades:
            return

        now = now_ist().replace(tzinfo=None)
        resolved_ids = []

        for trade in trades:
            status = None
            outcome_pct = 0

            if trade["direction"] == "BULLISH":
                if trade["target"] and ltp >= trade["target"]:
                    status = "target_hit"
                    outcome_pct = round((ltp - trade["entry"]) / trade["entry"] * 100, 2) if trade["entry"] else 0
                elif trade["stop_loss"] and ltp <= trade["stop_loss"]:
                    status = "sl_hit"
                    outcome_pct = round((ltp - trade["entry"]) / trade["entry"] * 100, 2) if trade["entry"] else 0
                elif trade["expires_at"] and now > trade["expires_at"]:
                    status = "expired"
                    outcome_pct = round((ltp - trade["entry"]) / trade["entry"] * 100, 2) if trade["entry"] else 0

            elif trade["direction"] == "BEARISH":
                if trade["target"] and ltp <= trade["target"]:
                    status = "target_hit"
                    outcome_pct = round((trade["entry"] - ltp) / trade["entry"] * 100, 2) if trade["entry"] else 0
                elif trade["stop_loss"] and ltp >= trade["stop_loss"]:
                    status = "sl_hit"
                    outcome_pct = round((trade["entry"] - ltp) / trade["entry"] * 100, 2) if trade["entry"] else 0
                elif trade["expires_at"] and now > trade["expires_at"]:
                    status = "expired"
                    outcome_pct = round((trade["entry"] - ltp) / trade["entry"] * 100, 2) if trade["entry"] else 0

            if status:
                resolved_ids.append(trade["id"])
                # Update DB
                try:
                    db = SessionLocal()
                    try:
                        record = db.query(TradeSignalLog).filter(TradeSignalLog.id == trade["id"]).first()
                        if record and record.status == "open":
                            record.status = status
                            record.outcome_price = ltp
                            record.outcome_pct = outcome_pct
                            record.resolved_at = now
                            if record.highest_price is None or ltp > record.highest_price:
                                record.highest_price = ltp
                            if record.lowest_price is None or ltp < record.lowest_price:
                                record.lowest_price = ltp
                            db.commit()
                            logger.info(f"Trade resolved via tick: {symbol} {trade['timeframe']} → {status} ({outcome_pct:+.2f}%)")
                    finally:
                        db.close()
                except Exception as e:
                    logger.debug(f"Tick trade resolve failed: {e}")

        # Remove resolved trades from cache
        if resolved_ids:
            self._open_trades_cache[symbol] = [
                t for t in trades if t["id"] not in resolved_ids
            ]

    def log_signal(self, symbol: str, timeframe: str, signal_data: dict,
                   current_price: float):
        """Log a trade prediction from an MTF signal computation."""
        if not signal_data or signal_data.get("direction") == "NEUTRAL":
            return

        entry = signal_data.get("entry")
        target = signal_data.get("target")
        stop_loss = signal_data.get("stop_loss")

        if not entry or not target:
            return

        db = SessionLocal()
        try:
            # Check for recent duplicate
            min_interval = MIN_LOG_INTERVAL_MINUTES.get(timeframe, 30)
            cutoff = now_ist().replace(tzinfo=None) - timedelta(minutes=min_interval)
            recent = (
                db.query(TradeSignalLog)
                .filter(
                    TradeSignalLog.symbol == symbol,
                    TradeSignalLog.timeframe == timeframe,
                    TradeSignalLog.created_at > cutoff,
                )
                .first()
            )
            if recent:
                return  # Already logged recently

            window = TIMEFRAME_WINDOWS.get(timeframe, timedelta(days=1))
            expires_at = now_ist().replace(tzinfo=None) + window

            log = TradeSignalLog(
                symbol=symbol,
                timeframe=timeframe,
                direction=signal_data["direction"],
                confidence=signal_data.get("confidence", 0),
                current_price=current_price,
                predicted_price=signal_data.get("predicted_price"),
                entry=entry,
                target=target,
                stop_loss=stop_loss,
                risk_reward=signal_data.get("risk_reward"),
                model_confidence=signal_data.get("model_confidence"),
                regime=signal_data.get("regime"),
                volume_conviction=signal_data.get("volume_conviction"),
                status="open",
                expires_at=expires_at,
            )
            db.add(log)
            db.commit()
            logger.debug(f"Logged trade signal: {symbol} {timeframe} {signal_data['direction']}")
            # Refresh cache so tick checker picks up new trade
            self._cache_loaded = False
        except Exception as e:
            db.rollback()
            logger.debug(f"Trade signal logging failed: {e}")
        finally:
            db.close()

    def validate_open_signals(self):
        """Check all open signals against current prices.
        Called by background task periodically."""
        from app.services.data_fetcher import data_fetcher

        db = SessionLocal()
        try:
            now = now_ist().replace(tzinfo=None)

            # Get all open signals
            open_signals = (
                db.query(TradeSignalLog)
                .filter(TradeSignalLog.status == "open")
                .all()
            )

            if not open_signals:
                return {"checked": 0, "resolved": 0}

            # Group by symbol to batch quote fetches
            symbols = list({s.symbol for s in open_signals})
            quotes = data_fetcher.get_bulk_quotes(symbols)
            quote_map = {q["symbol"]: q for q in quotes if q.get("symbol")}

            resolved = 0
            for sig in open_signals:
                quote = quote_map.get(sig.symbol, {})
                ltp = quote.get("ltp")
                high = quote.get("high")
                low = quote.get("low")

                if not ltp:
                    continue

                # Track high/low watermarks
                if sig.highest_price is None or ltp > sig.highest_price:
                    sig.highest_price = ltp
                if high and (sig.highest_price is None or high > sig.highest_price):
                    sig.highest_price = high
                if sig.lowest_price is None or ltp < sig.lowest_price:
                    sig.lowest_price = ltp
                if low and (sig.lowest_price is None or low < sig.lowest_price):
                    sig.lowest_price = low

                # Check outcomes
                if sig.direction == "BULLISH":
                    if sig.target and ltp >= sig.target:
                        sig.status = "target_hit"
                        sig.outcome_price = ltp
                        sig.outcome_pct = round((ltp - sig.entry) / sig.entry * 100, 2) if sig.entry else 0
                        sig.resolved_at = now
                        resolved += 1
                    elif sig.stop_loss and ltp <= sig.stop_loss:
                        sig.status = "sl_hit"
                        sig.outcome_price = ltp
                        sig.outcome_pct = round((ltp - sig.entry) / sig.entry * 100, 2) if sig.entry else 0
                        sig.resolved_at = now
                        resolved += 1
                    elif sig.expires_at and now > sig.expires_at:
                        sig.status = "expired"
                        sig.outcome_price = ltp
                        sig.outcome_pct = round((ltp - sig.entry) / sig.entry * 100, 2) if sig.entry else 0
                        sig.resolved_at = now
                        resolved += 1

                elif sig.direction == "BEARISH":
                    if sig.target and ltp <= sig.target:
                        sig.status = "target_hit"
                        sig.outcome_price = ltp
                        sig.outcome_pct = round((sig.entry - ltp) / sig.entry * 100, 2) if sig.entry else 0
                        sig.resolved_at = now
                        resolved += 1
                    elif sig.stop_loss and ltp >= sig.stop_loss:
                        sig.status = "sl_hit"
                        sig.outcome_price = ltp
                        sig.outcome_pct = round((sig.entry - ltp) / sig.entry * 100, 2) if sig.entry else 0
                        sig.resolved_at = now
                        resolved += 1
                    elif sig.expires_at and now > sig.expires_at:
                        sig.status = "expired"
                        sig.outcome_price = ltp
                        sig.outcome_pct = round((sig.entry - ltp) / sig.entry * 100, 2) if sig.entry else 0
                        sig.resolved_at = now
                        resolved += 1

            db.commit()
            return {"checked": len(open_signals), "resolved": resolved}
        except Exception as e:
            db.rollback()
            logger.warning(f"Trade validation failed: {e}")
            return {"checked": 0, "resolved": 0, "error": str(e)}
        finally:
            db.close()

    def get_accuracy_stats(self, symbol: str = None) -> dict:
        """Get trade prediction accuracy stats, optionally for a specific stock."""
        db = SessionLocal()
        try:
            from sqlalchemy import func, case

            query = db.query(TradeSignalLog).filter(
                TradeSignalLog.status != "open"
            )
            if symbol:
                query = query.filter(TradeSignalLog.symbol == symbol)

            resolved = query.all()
            if not resolved:
                return {"total": 0}

            # Overall stats
            total = len(resolved)
            target_hits = sum(1 for s in resolved if s.status == "target_hit")
            sl_hits = sum(1 for s in resolved if s.status == "sl_hit")
            expired = sum(1 for s in resolved if s.status == "expired")
            expired_profit = sum(1 for s in resolved if s.status == "expired" and (s.outcome_pct or 0) > 0)

            win_rate = round((target_hits + expired_profit) / total * 100, 1) if total > 0 else 0
            avg_win = 0
            avg_loss = 0
            wins = [s.outcome_pct for s in resolved if s.status == "target_hit" and s.outcome_pct]
            losses = [s.outcome_pct for s in resolved if s.status == "sl_hit" and s.outcome_pct]
            if wins:
                avg_win = round(sum(wins) / len(wins), 2)
            if losses:
                avg_loss = round(sum(losses) / len(losses), 2)

            # By timeframe
            by_timeframe = {}
            for tf in ["intraday", "short_term", "long_term"]:
                tf_signals = [s for s in resolved if s.timeframe == tf]
                if tf_signals:
                    tf_target = sum(1 for s in tf_signals if s.status == "target_hit")
                    tf_expired_profit = sum(1 for s in tf_signals if s.status == "expired" and (s.outcome_pct or 0) > 0)
                    by_timeframe[tf] = {
                        "total": len(tf_signals),
                        "target_hit": tf_target,
                        "sl_hit": sum(1 for s in tf_signals if s.status == "sl_hit"),
                        "expired": sum(1 for s in tf_signals if s.status == "expired"),
                        "win_rate": round((tf_target + tf_expired_profit) / len(tf_signals) * 100, 1),
                    }

            # By symbol (top 10)
            by_symbol = {}
            for s in resolved:
                if s.symbol not in by_symbol:
                    by_symbol[s.symbol] = {"total": 0, "wins": 0, "pnl_sum": 0}
                by_symbol[s.symbol]["total"] += 1
                if s.status == "target_hit" or (s.status == "expired" and (s.outcome_pct or 0) > 0):
                    by_symbol[s.symbol]["wins"] += 1
                by_symbol[s.symbol]["pnl_sum"] += s.outcome_pct or 0

            top_symbols = sorted(
                [{"symbol": sym, "total": d["total"],
                  "win_rate": round(d["wins"] / d["total"] * 100, 1) if d["total"] > 0 else 0,
                  "total_pnl": round(d["pnl_sum"], 2)}
                 for sym, d in by_symbol.items()],
                key=lambda x: x["total"], reverse=True
            )[:15]

            # Recent trades
            recent = (
                db.query(TradeSignalLog)
                .filter(TradeSignalLog.status != "open")
            )
            if symbol:
                recent = recent.filter(TradeSignalLog.symbol == symbol)
            recent = recent.order_by(TradeSignalLog.resolved_at.desc()).limit(20).all()

            recent_list = [{
                "symbol": s.symbol,
                "timeframe": s.timeframe,
                "direction": s.direction,
                "entry": s.entry,
                "target": s.target,
                "stop_loss": s.stop_loss,
                "status": s.status,
                "outcome_pct": s.outcome_pct,
                "outcome_price": s.outcome_price,
                "highest_price": s.highest_price,
                "lowest_price": s.lowest_price,
                "confidence": s.confidence,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "resolved_at": s.resolved_at.isoformat() if s.resolved_at else None,
            } for s in recent]

            # Open trades count
            open_count = db.query(TradeSignalLog).filter(TradeSignalLog.status == "open").count()

            return {
                "total": total,
                "target_hit": target_hits,
                "sl_hit": sl_hits,
                "expired": expired,
                "win_rate": win_rate,
                "avg_win_pct": avg_win,
                "avg_loss_pct": avg_loss,
                "open_trades": open_count,
                "by_timeframe": by_timeframe,
                "by_symbol": top_symbols,
                "recent_trades": recent_list,
            }
        finally:
            db.close()

    def learn_from_trades(self) -> dict:
        """Analyze resolved trades and update stock learning profiles with trade accuracy.
        Called after validate_open_signals."""
        try:
            from app.services.stock_learner import stock_learner

            db = SessionLocal()
            try:
                from sqlalchemy import func

                # Get symbols with enough resolved trades
                symbol_counts = (
                    db.query(TradeSignalLog.symbol, func.count(TradeSignalLog.id))
                    .filter(TradeSignalLog.status != "open")
                    .group_by(TradeSignalLog.symbol)
                    .having(func.count(TradeSignalLog.id) >= 5)
                    .all()
                )

                updated = 0
                for symbol, count in symbol_counts:
                    profile = stock_learner.get_profile(symbol)
                    if not profile:
                        continue

                    # Get trade accuracy for this symbol
                    trades = (
                        db.query(TradeSignalLog)
                        .filter(
                            TradeSignalLog.symbol == symbol,
                            TradeSignalLog.status != "open",
                        )
                        .order_by(TradeSignalLog.created_at.desc())
                        .limit(50)
                        .all()
                    )

                    if not trades:
                        continue

                    target_hits = sum(1 for t in trades if t.status == "target_hit")
                    expired_profit = sum(1 for t in trades if t.status == "expired" and (t.outcome_pct or 0) > 0)
                    trade_win_rate = (target_hits + expired_profit) / len(trades) * 100

                    # Update profile with trade accuracy
                    profile["trade_accuracy"] = {
                        "win_rate": round(trade_win_rate, 1),
                        "total_trades": len(trades),
                        "target_hits": target_hits,
                        "avg_pnl": round(sum(t.outcome_pct or 0 for t in trades) / len(trades), 2),
                    }

                    # Find best timeframe by trade win rate
                    tf_stats = {}
                    for t in trades:
                        if t.timeframe not in tf_stats:
                            tf_stats[t.timeframe] = {"wins": 0, "total": 0}
                        tf_stats[t.timeframe]["total"] += 1
                        if t.status == "target_hit" or (t.status == "expired" and (t.outcome_pct or 0) > 0):
                            tf_stats[t.timeframe]["wins"] += 1

                    profile["trade_accuracy"]["by_timeframe"] = {
                        tf: {"win_rate": round(s["wins"] / s["total"] * 100, 1), "trades": s["total"]}
                        for tf, s in tf_stats.items() if s["total"] >= 3
                    }

                    # Save updated profile
                    import json
                    from app.services.stock_learner import PROFILES_DIR
                    safe = symbol.replace(" ", "_").replace("^", "")
                    path = PROFILES_DIR / f"{safe}.json"
                    profile["updated_at"] = datetime.utcnow().isoformat()
                    path.write_text(json.dumps(profile, indent=2, default=str))
                    updated += 1

                return {"symbols_updated": updated}
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Trade learning failed: {e}")
            return {"error": str(e)}


trade_tracker = TradeTracker()
