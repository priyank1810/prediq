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
    "intraday_15m": timedelta(hours=2),
    "intraday_30m": timedelta(hours=3),
    "short_1h": timedelta(hours=6),
    "short_4h": timedelta(days=1),
}

# Timeframe → first checkpoint interval (check prediction at this exact time)
TIMEFRAME_CHECK_INTERVALS = {
    "intraday_15m": timedelta(minutes=15),
    "intraday_30m": timedelta(minutes=30),
    "short_1h": timedelta(hours=1),
    "short_4h": timedelta(hours=4),
}

# Minimum minutes between logging same symbol+timeframe (prevent duplicates)
MIN_LOG_INTERVAL_MINUTES = {
    "intraday_15m": 15,
    "intraday_30m": 30,
    "short_1h": 60,
    "short_4h": 240,
}

# Daily loss limit: stop taking new trades if daily P&L drops below this %
DAILY_LOSS_LIMIT_PCT = -2.0

# Cooldown after SL hit: don't re-enter same stock for this many minutes
SL_COOLDOWN_MINUTES = 120

# Market regime: suppress bullish signals if NIFTY is down more than this %
MARKET_DOWN_THRESHOLD_PCT = -1.0

# Auto-blacklist: stocks with <50% win rate over 30+ resolved trades
BLACKLIST_MIN_TRADES = 30
BLACKLIST_WIN_RATE_PCT = 50.0
BLACKLIST_REFRESH_SECONDS = 1800  # rebuild cache every 30 min


class TradeTracker:
    """Tracks trade signal predictions and validates outcomes."""

    def _effective_sl(self, sig) -> "float | None":
        """Return trailing SL for 4h signals; original SL for others.

        Uses persisted highest_price to reconstruct the trailing SL level
        from the same milestone thresholds used in check_symbol_tick.
        """
        if sig.timeframe != "short_4h" or not sig.entry or not sig.target or not sig.stop_loss:
            return sig.stop_loss
        highest = sig.highest_price or sig.entry
        target_dist = sig.target - sig.entry
        if target_dist <= 0:
            return sig.stop_loss
        profit = highest - sig.entry
        if profit >= target_dist:
            trailing = round(sig.entry + profit * 0.70, 2)
        elif profit >= target_dist * 0.75:
            trailing = round(sig.entry + profit * 0.50, 2)
        elif profit >= target_dist * 0.50:
            trailing = round(sig.entry, 2)
        else:
            return sig.stop_loss
        return max(trailing, sig.stop_loss)

    def _check_signal_still_valid(self, symbol: str, direction: str, timeframe: str = None) -> bool:
        """Re-evaluate AI signal at checkpoint time.
        Returns True if AI still agrees with the trade direction.

        If `timeframe` is provided, checks ONLY that specific timeframe (e.g.
        "intraday_15m"). Each trade's checkpoint must evaluate its own
        timeframe — otherwise a still-bullish 30m signal would wrongly keep a
        15m trade open (and vice versa).

        If `timeframe` is not provided, falls back to "any bullish" mode for
        backwards compatibility with the old trailing-SL smart exit path.
        """
        try:
            from app.services.signal_service import signal_service
            mtf = signal_service.get_multi_timeframe_signals(symbol)
            if not mtf:
                return False

            # Map TradeSignalLog timeframe (e.g. "intraday_15m") to MTF structure key
            # MTF returns: {"intraday": {"15m": {...}, "30m": {...}}, "short_term": {"1h": {...}, "4h": {...}}}
            if timeframe:
                tf_map = {
                    "intraday_15m": ("intraday", "15m"),
                    "intraday_30m": ("intraday", "30m"),
                    "short_1h": ("short_term", "1h"),
                    "short_4h": ("short_term", "4h"),
                }
                if timeframe in tf_map:
                    group_key, sub_key = tf_map[timeframe]
                    tf_sig = mtf.get(group_key, {}).get(sub_key)
                    if tf_sig and tf_sig.get("direction") == direction:
                        return True
                    return False

            # Fallback: any timeframe still bullish (for non-checkpoint callers)
            for group in (mtf.get("intraday", {}), mtf.get("short_term", {})):
                for tf_sig in group.values():
                    if tf_sig and tf_sig.get("direction") == direction:
                        return True
            return False
        except Exception:
            return True  # If signal check fails, assume still valid (don't exit on error)

    def __init__(self):
        self._open_trades_cache = {}  # symbol -> [trade dicts]
        self._cache_loaded = False
        self._daily_loss_breaker = False  # circuit breaker for the day
        self._breaker_date = None  # date when breaker was last set
        self._market_regime_cache = None  # (timestamp, nifty_change_pct)
        self._blacklist_cache = None  # (timestamp, set of blacklisted symbols)

    def _get_blacklisted_symbols(self, db) -> set:
        """Return set of symbols with <50% win rate over 30+ resolved trades.
        Cached for 30 min to avoid running this query on every signal log."""
        import time
        if self._blacklist_cache:
            ts, symbols = self._blacklist_cache
            if time.time() - ts < BLACKLIST_REFRESH_SECONDS:
                return symbols
        try:
            from collections import defaultdict
            stats = defaultdict(lambda: {"total": 0, "wins": 0})
            resolved = db.query(
                TradeSignalLog.symbol, TradeSignalLog.outcome_pct
            ).filter(TradeSignalLog.status != "open").all()
            for sym, pct in resolved:
                stats[sym]["total"] += 1
                if (pct or 0) > 0:
                    stats[sym]["wins"] += 1
            blacklist = set()
            for sym, s in stats.items():
                if s["total"] >= BLACKLIST_MIN_TRADES:
                    wr = s["wins"] / s["total"] * 100
                    if wr < BLACKLIST_WIN_RATE_PCT:
                        blacklist.add(sym)
            self._blacklist_cache = (time.time(), blacklist)
            if blacklist:
                logger.info(f"Auto-blacklist refreshed: {len(blacklist)} stocks excluded ({sorted(blacklist)[:5]}...)")
            return blacklist
        except Exception as e:
            logger.debug(f"Blacklist computation failed: {e}")
            return self._blacklist_cache[1] if self._blacklist_cache else set()


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
                    "id": t.id, "_symbol": t.symbol,
                    "direction": t.direction,
                    "entry": t.entry, "target": t.target,
                    "stop_loss": t.stop_loss, "timeframe": t.timeframe,
                    "check_at": t.check_at,
                    "expires_at": t.expires_at,
                })
            self._cache_loaded = True
        finally:
            db.close()

    def check_symbol_tick(self, symbol: str, ltp: float, high: float = 0, low: float = 0):
        """Called on every price tick — real-time trade validation.
        Uses in-memory cache, only hits DB when a trade resolves.
        Skips validation when market is closed."""
        from app.utils.helpers import is_market_open
        if not is_market_open():
            return

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
                entry = trade["entry"] or 0
                target = trade["target"] or 0
                sl = trade.get("trailing_sl") or trade["stop_loss"] or 0
                highest = trade.get("highest") or entry

                # Track highest price seen
                if ltp > highest:
                    trade["highest"] = ltp
                    highest = ltp

                # Smart trailing stop: move SL up as price rises
                if entry > 0 and target > entry:
                    target_dist = target - entry
                    profit = highest - entry

                    if profit >= target_dist:
                        # Past target — trail SL at 70% of max profit
                        trade["trailing_sl"] = round(entry + profit * 0.70, 2)
                    elif profit >= target_dist * 0.75:
                        # 75% to target — lock in 50% of profit
                        trade["trailing_sl"] = round(entry + profit * 0.50, 2)
                    elif profit >= target_dist * 0.50:
                        # 50% to target — move SL to breakeven
                        trade["trailing_sl"] = round(entry, 2)

                    sl = trade.get("trailing_sl") or sl

                # Check exits
                if sl and ltp <= sl:
                    if trade.get("trailing_sl") and ltp > entry:
                        status = "target_hit"
                    else:
                        status = "sl_hit"
                    outcome_pct = round((ltp - entry) / entry * 100, 2) if entry else 0
                elif ltp > entry and profit >= target_dist * 0.5:
                    # In profit and past 50% — check if AI still predicts up
                    # Only check every few ticks (not every 3s)
                    last_check = trade.get("_last_pred_check", 0)
                    import time as _time
                    if _time.time() - last_check > 300:  # Check every 5 min
                        trade["_last_pred_check"] = _time.time()
                        try:
                            from app.services.signal_service import signal_service
                            mtf = signal_service.get_multi_timeframe_signals(trade.get("_symbol") or symbol)
                            # Check if any timeframe still says bullish
                            still_bullish = False
                            for group in (mtf.get("intraday", {}), mtf.get("short_term", {})):
                                for tf_sig in group.values():
                                    if tf_sig and tf_sig.get("direction") == "BULLISH":
                                        still_bullish = True
                                        break
                            if not still_bullish:
                                # AI says sell — exit with profit
                                status = "target_hit"
                                outcome_pct = round((ltp - entry) / entry * 100, 2) if entry else 0
                                logger.info(f"Smart exit: {symbol} — AI flipped, selling at +{outcome_pct}%")
                        except Exception:
                            pass  # If check fails, hold
                elif trade.get("check_at") and now > trade["check_at"]:
                    # Timeframe checkpoint: re-evaluate AI signal to decide continue/book
                    outcome_pct = round((ltp - entry) / entry * 100, 2) if entry else 0
                    target_move = (target - entry) if target and entry else 0
                    achieved = (ltp - entry) if entry else 0
                    hit_half_target = target_move > 0 and achieved >= target_move * 0.5
                    still_valid = self._check_signal_still_valid(symbol, "BULLISH", trade.get("timeframe"))
                    if still_valid and hit_half_target:
                        # AI agrees but already captured 50%+ of target — book profit
                        status = "target_hit"
                        logger.info(f"Checkpoint {symbol} {trade.get('timeframe')}: AI BULLISH + {outcome_pct:+.2f}% (>=50% target) — booking profit")
                    elif still_valid:
                        # AI agrees, not yet at 50% — continue to target/SL/expiry
                        trade["check_at"] = None
                        trade["_check_at_cleared"] = True
                        logger.info(f"Checkpoint {symbol}: AI still BULLISH ({outcome_pct:+.2f}%) — continuing")
                    elif outcome_pct > 0:
                        # AI flipped but in profit — book profit
                        status = "target_hit"
                        logger.info(f"Checkpoint {symbol}: AI flipped, booking profit +{outcome_pct:.2f}%")
                    else:
                        # AI flipped and in loss — book loss
                        status = "wrong"
                        logger.info(f"Checkpoint {symbol}: AI flipped, booking loss {outcome_pct:.2f}%")
                elif trade["expires_at"] and now > trade["expires_at"]:
                    outcome_pct = round((ltp - entry) / entry * 100, 2) if entry else 0
                    status = "correct" if outcome_pct > 0 else "wrong"

            elif trade["direction"] == "BEARISH":
                if trade["target"] and ltp <= trade["target"]:
                    status = "target_hit"
                    outcome_pct = round((trade["entry"] - ltp) / trade["entry"] * 100, 2) if trade["entry"] else 0
                elif trade["stop_loss"] and ltp >= trade["stop_loss"]:
                    status = "sl_hit"
                    outcome_pct = round((trade["entry"] - ltp) / trade["entry"] * 100, 2) if trade["entry"] else 0
                elif trade.get("check_at") and now > trade["check_at"]:
                    bear_entry = trade["entry"] or 0
                    bear_target = trade["target"] or 0
                    outcome_pct = round((bear_entry - ltp) / bear_entry * 100, 2) if bear_entry else 0
                    target_move = (bear_entry - bear_target) if bear_entry and bear_target else 0
                    achieved = (bear_entry - ltp) if bear_entry else 0
                    hit_half_target = target_move > 0 and achieved >= target_move * 0.5
                    still_valid = self._check_signal_still_valid(symbol, "BEARISH", trade.get("timeframe"))
                    if still_valid and hit_half_target:
                        status = "target_hit"
                        logger.info(f"Checkpoint {symbol} {trade.get('timeframe')}: AI BEARISH + {outcome_pct:+.2f}% (>=50% target) — booking profit")
                    elif still_valid:
                        trade["check_at"] = None
                        trade["_check_at_cleared"] = True
                        logger.info(f"Checkpoint {symbol}: AI still BEARISH ({outcome_pct:+.2f}%) — continuing")
                    elif outcome_pct > 0:
                        status = "target_hit"
                        logger.info(f"Checkpoint {symbol}: AI flipped, booking profit +{outcome_pct:.2f}%")
                    else:
                        status = "wrong"
                        logger.info(f"Checkpoint {symbol}: AI flipped, booking loss {outcome_pct:.2f}%")
                elif trade["expires_at"] and now > trade["expires_at"]:
                    outcome_pct = round((trade["entry"] - ltp) / trade["entry"] * 100, 2) if trade["entry"] else 0
                    status = "correct" if outcome_pct > 0 else "wrong"

            # Persist check_at cleared to DB (checkpoint passed, AI still valid)
            if not status and trade.get("check_at") is None and trade.get("_check_at_cleared"):
                try:
                    db = SessionLocal()
                    try:
                        record = db.query(TradeSignalLog).filter(TradeSignalLog.id == trade["id"]).first()
                        if record:
                            record.check_at = None
                            db.commit()
                    finally:
                        db.close()
                except Exception:
                    pass
                trade.pop("_check_at_cleared", None)

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
            else:
                # Not resolved yet — track high/low watermarks in cache
                if "highest" not in trade or ltp > trade.get("highest", 0):
                    trade["highest"] = ltp
                if "lowest" not in trade or ltp < trade.get("lowest", float("inf")):
                    trade["lowest"] = ltp

        # Remove resolved trades from cache
        if resolved_ids:
            self._open_trades_cache[symbol] = [
                t for t in trades if t["id"] not in resolved_ids
            ]

    def _is_near_earnings(self, symbol: str) -> bool:
        """Check if stock has earnings within 2 days — risky to trade."""
        try:
            from app.utils.cache import cache as _cache
            key = f"earnings_check:{symbol}"
            cached = _cache.get(key)
            if cached is not None:
                return cached

            from app.services.fundamental_service import fundamental_service
            fund = fundamental_service.get_fundamentals(symbol)
            if fund and fund.get("earnings_quarterly"):
                from datetime import datetime, timedelta
                today = datetime.now().date()
                for q in fund["earnings_quarterly"][:2]:
                    date_str = q.get("date", "")
                    if date_str:
                        try:
                            earn_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                            if abs((earn_date - today).days) <= 2:
                                _cache.set(key, True, 3600)
                                return True
                        except Exception:
                            pass
            _cache.set(key, False, 3600)
            return False
        except Exception:
            return False

    def _check_daily_loss_limit(self, db) -> bool:
        """Return True if daily loss limit has been breached — stop new trades."""
        today = now_ist().replace(tzinfo=None).date()

        # Reset breaker on new day
        if self._breaker_date != today:
            self._daily_loss_breaker = False
            self._breaker_date = today

        if self._daily_loss_breaker:
            return True

        # Calculate today's realized P&L
        today_start = datetime.combine(today, datetime.min.time())
        today_resolved = (
            db.query(TradeSignalLog)
            .filter(
                TradeSignalLog.status != "open",
                TradeSignalLog.resolved_at >= today_start,
            )
            .all()
        )
        if not today_resolved:
            return False

        # Approximate P&L as % of capital (assume ₹5000 per trade avg)
        total_pnl_pct = sum(t.outcome_pct or 0 for t in today_resolved) / len(today_resolved)
        if total_pnl_pct <= DAILY_LOSS_LIMIT_PCT:
            self._daily_loss_breaker = True
            logger.warning(f"Daily loss limit hit: avg {total_pnl_pct:.2f}% across {len(today_resolved)} trades — pausing new entries")
            return True
        return False

    def _check_market_regime(self) -> bool:
        """Return True if market is in risk-off mode (NIFTY down >1%). Suppress bullish signals."""
        try:
            import time
            # Cache for 5 minutes
            if self._market_regime_cache:
                ts, change_pct = self._market_regime_cache
                if time.time() - ts < 300:
                    return change_pct <= MARKET_DOWN_THRESHOLD_PCT

            from app.services.data_fetcher import data_fetcher
            # Use yesterday's close from history + live LTP for accurate change%
            # (Angel One quote 'close' field is unreliable for indices)
            change_pct = 0
            hist = data_fetcher.get_historical_data("NIFTY 50", period="5d")
            if hist is not None and not hist.empty:
                prev_close = float(hist["close"].iloc[-1])  # Last completed day's close

                # Get live price for today's value
                quotes = data_fetcher.get_bulk_quotes(["NIFTY 50"])
                ltp = None
                if quotes:
                    q = quotes[0] if isinstance(quotes, list) else quotes.get("NIFTY 50", {})
                    ltp = q.get("ltp")

                if ltp and prev_close > 0:
                    change_pct = round((ltp - prev_close) / prev_close * 100, 2)
                    logger.debug(f"Market regime: NIFTY prev_close={prev_close} ltp={ltp} change={change_pct}%")

            if change_pct == 0:
                # Fallback: trust quote API pct_change
                quotes = data_fetcher.get_bulk_quotes(["NIFTY 50"])
                if quotes:
                    q = quotes[0] if isinstance(quotes, list) else quotes.get("NIFTY 50", {})
                    change_pct = q.get("change_pct") or q.get("pChange") or 0

            self._market_regime_cache = (time.time(), change_pct)
            if change_pct <= MARKET_DOWN_THRESHOLD_PCT:
                logger.info(f"Market risk-off: NIFTY {change_pct:+.2f}% — suppressing bullish signals")
                return True
            return False
        except Exception as e:
            logger.debug(f"Market regime check failed: {e}")
            return False

    def _check_sl_cooldown(self, symbol: str, db) -> bool:
        """Return True if this stock hit a stop-loss recently — needs cooldown."""
        cooldown_cutoff = now_ist().replace(tzinfo=None) - timedelta(minutes=SL_COOLDOWN_MINUTES)
        recent_sl = (
            db.query(TradeSignalLog)
            .filter(
                TradeSignalLog.symbol == symbol,
                TradeSignalLog.status == "sl_hit",
                TradeSignalLog.resolved_at >= cooldown_cutoff,
            )
            .first()
        )
        if recent_sl:
            logger.debug(f"Skipped {symbol}: SL cooldown ({SL_COOLDOWN_MINUTES}min)")
            return True
        return False

    def log_signal(self, symbol: str, timeframe: str, signal_data: dict,
                   current_price: float):
        """Log a trade prediction from an MTF signal computation."""
        if not signal_data or signal_data.get("direction") != "BULLISH":
            return  # Only track bullish signals (portfolio is long-only)

        # Only track signals with 45%+ confidence
        if (signal_data.get("confidence") or 0) < 45:
            return

        # Skip signals where AI confidence is falling — model is losing conviction
        if signal_data.get("confidence_trend") == "falling":
            logger.debug(f"Skipped {symbol} {timeframe}: confidence trend falling")
            return

        # Don't log signals near market close — not enough time to play out
        current = now_ist().replace(tzinfo=None)
        market_close_hour, market_close_min = 15, 10
        if timeframe.startswith("intraday"):
            # Intraday signals: stop 20 min before close (15:10)
            market_close_min = 10
        else:
            # Short-term: stop 5 min before close (15:25)
            market_close_min = 25
        if current.hour > market_close_hour or (current.hour == market_close_hour and current.minute >= market_close_min):
            return

        # Market regime filter: skip bullish signals when NIFTY is tanking
        if self._check_market_regime():
            return

        # Skip stocks near earnings announcements (too risky)
        if self._is_near_earnings(symbol):
            logger.debug(f"Skipped {symbol}: near earnings announcement")
            return

        target = signal_data.get("target")
        stop_loss = signal_data.get("stop_loss")

        if not current_price or not target:
            return

        # Use actual market price as entry (what you'd really pay)
        entry = round(current_price, 2)

        # Rebase target/SL relative to live entry price.
        # signal_service computes target/SL from candle close; if live LTP has
        # drifted, the levels would be misaligned. Shift by the same delta.
        candle_price = signal_data.get("entry")  # support level used as candle reference
        if candle_price and candle_price > 0 and abs(entry - candle_price) > 0.01:
            shift = entry - candle_price
            target = round(target + shift, 2)
            if stop_loss is not None:
                stop_loss = round(stop_loss + shift, 2)

        db = SessionLocal()
        try:
            # Auto-blacklist: skip stocks with poor historical win rate
            blacklist = self._get_blacklisted_symbols(db)
            if symbol in blacklist:
                logger.debug(f"Skipped {symbol}: auto-blacklisted (<{BLACKLIST_WIN_RATE_PCT}% win rate over {BLACKLIST_MIN_TRADES}+ trades)")
                return

            # Daily loss circuit breaker
            if self._check_daily_loss_limit(db):
                return

            # SL cooldown: don't re-enter a stock that just hit stop-loss
            if self._check_sl_cooldown(symbol, db):
                return

            # Check for recent duplicate (same symbol + timeframe)
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

            # Cross-timeframe dedup: max 2 open trades per symbol
            open_for_symbol = (
                db.query(TradeSignalLog)
                .filter(
                    TradeSignalLog.symbol == symbol,
                    TradeSignalLog.status == "open",
                )
                .count()
            )
            if open_for_symbol >= 2:
                return  # Already have 2 open positions on this stock

            now = now_ist().replace(tzinfo=None)
            window = TIMEFRAME_WINDOWS.get(timeframe, timedelta(days=1))
            expires_at = now + window
            check_interval = TIMEFRAME_CHECK_INTERVALS.get(timeframe)
            check_at = (now + check_interval) if check_interval else None

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
                model_used=signal_data.get("model_used", "v1"),
                regime=signal_data.get("regime"),
                volume_conviction=signal_data.get("volume_conviction"),
                confidence_trend=signal_data.get("confidence_trend"),
                status="open",
                check_at=check_at,
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
        Called by background task periodically. Skips when market closed."""
        from app.utils.helpers import is_market_open
        if not is_market_open():
            return {"checked": 0, "resolved": 0, "market_closed": True}
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
                    elif (effective_sl := self._effective_sl(sig)) and ltp <= effective_sl:
                        outcome_pct = round((ltp - sig.entry) / sig.entry * 100, 2) if sig.entry else 0
                        # Trailing SL triggered above entry = profitable exit
                        sig.status = "target_hit" if (sig.entry and ltp > sig.entry) else "sl_hit"
                        sig.outcome_price = ltp
                        sig.outcome_pct = outcome_pct
                        sig.resolved_at = now
                        resolved += 1
                        if sig.timeframe == "short_4h" and effective_sl != sig.stop_loss:
                            logger.info(f"Trailing SL exit {sig.symbol}: ltp={ltp} trailing_sl={effective_sl} outcome={outcome_pct:+.2f}%")
                    elif sig.check_at and now > sig.check_at:
                        # Timeframe checkpoint: re-evaluate AI signal
                        pct = round((ltp - sig.entry) / sig.entry * 100, 2) if sig.entry else 0
                        target_move = (sig.target - sig.entry) if sig.target and sig.entry else 0
                        achieved = (ltp - sig.entry) if sig.entry else 0
                        hit_half_target = target_move > 0 and achieved >= target_move * 0.5
                        still_valid = self._check_signal_still_valid(sig.symbol, "BULLISH", sig.timeframe)
                        if still_valid and hit_half_target:
                            # AI agrees + already captured 50%+ of target — book profit
                            sig.status = "target_hit"
                            sig.outcome_price = ltp
                            sig.outcome_pct = pct
                            sig.resolved_at = now
                            resolved += 1
                            logger.info(f"Checkpoint {sig.symbol} {sig.timeframe}: AI BULLISH + {pct:+.2f}% (>=50% target) — booking profit")
                        elif still_valid:
                            sig.check_at = None  # AI agrees, < 50% — continue
                            logger.info(f"Checkpoint {sig.symbol}: AI still BULLISH ({pct:+.2f}%) — continuing")
                        else:
                            sig.outcome_price = ltp
                            sig.outcome_pct = pct
                            sig.resolved_at = now
                            resolved += 1
                            if pct > 0:
                                sig.status = "target_hit"
                                logger.info(f"Checkpoint {sig.symbol}: AI flipped, booking profit +{pct:.2f}%")
                            else:
                                sig.status = "wrong"
                                logger.info(f"Checkpoint {sig.symbol}: AI flipped, booking loss {pct:.2f}%")
                    elif sig.expires_at and now > sig.expires_at:
                        sig.outcome_price = ltp
                        sig.outcome_pct = round((ltp - sig.entry) / sig.entry * 100, 2) if sig.entry else 0
                        sig.status = "correct" if sig.outcome_pct > 0 else "wrong"
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
                    elif sig.check_at and now > sig.check_at:
                        pct = round((sig.entry - ltp) / sig.entry * 100, 2) if sig.entry else 0
                        target_move = (sig.entry - sig.target) if sig.entry and sig.target else 0
                        achieved = (sig.entry - ltp) if sig.entry else 0
                        hit_half_target = target_move > 0 and achieved >= target_move * 0.5
                        still_valid = self._check_signal_still_valid(sig.symbol, "BEARISH", sig.timeframe)
                        if still_valid and hit_half_target:
                            sig.status = "target_hit"
                            sig.outcome_price = ltp
                            sig.outcome_pct = pct
                            sig.resolved_at = now
                            resolved += 1
                            logger.info(f"Checkpoint {sig.symbol} {sig.timeframe}: AI BEARISH + {pct:+.2f}% (>=50% target) — booking profit")
                        elif still_valid:
                            sig.check_at = None
                            logger.info(f"Checkpoint {sig.symbol}: AI still BEARISH ({pct:+.2f}%) — continuing")
                        else:
                            sig.outcome_price = ltp
                            sig.outcome_pct = pct
                            sig.resolved_at = now
                            resolved += 1
                            if pct > 0:
                                sig.status = "target_hit"
                                logger.info(f"Checkpoint {sig.symbol}: AI flipped, booking profit +{pct:.2f}%")
                            else:
                                sig.status = "wrong"
                                logger.info(f"Checkpoint {sig.symbol}: AI flipped, booking loss {pct:.2f}%")
                    elif sig.expires_at and now > sig.expires_at:
                        sig.outcome_price = ltp
                        sig.outcome_pct = round((sig.entry - ltp) / sig.entry * 100, 2) if sig.entry else 0
                        sig.status = "correct" if sig.outcome_pct > 0 else "wrong"
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
            correct = sum(1 for s in resolved if s.status == "correct")
            wrong = sum(1 for s in resolved if s.status == "wrong")

            # Win = any trade with positive P&L (consistent with stock_learner)
            total_wins = sum(1 for s in resolved if (s.outcome_pct or 0) > 0)
            win_rate = round(total_wins / total * 100, 1) if total > 0 else 0
            avg_win = 0
            avg_loss = 0
            # Count ALL profitable trades as wins, ALL losing trades as losses
            wins = [s.outcome_pct for s in resolved if s.outcome_pct and s.outcome_pct > 0]
            losses = [s.outcome_pct for s in resolved if s.outcome_pct and s.outcome_pct < 0]
            if wins:
                avg_win = round(sum(wins) / len(wins), 2)
            if losses:
                avg_loss = round(sum(losses) / len(losses), 2)

            # By timeframe group
            by_timeframe = {}
            for tf_group in ["intraday", "short_term"]:
                tf_signals = [s for s in resolved if s.timeframe and s.timeframe.startswith(tf_group.split("_")[0])]
                if tf_signals:
                    tf_target = sum(1 for s in tf_signals if s.status == "target_hit")
                    tf_correct = sum(1 for s in tf_signals if s.status == "correct")
                    tf_wrong = sum(1 for s in tf_signals if s.status == "wrong")
                    tf_wins = sum(1 for s in tf_signals if (s.outcome_pct or 0) > 0)
                    by_timeframe[tf_group] = {
                        "total": len(tf_signals),
                        "target_hit": tf_target,
                        "sl_hit": sum(1 for s in tf_signals if s.status == "sl_hit"),
                        "correct": tf_correct,
                        "wrong": tf_wrong,
                        "win_rate": round(tf_wins / len(tf_signals) * 100, 1),
                    }

            # By symbol (top 10)
            by_symbol = {}
            for s in resolved:
                if s.symbol not in by_symbol:
                    by_symbol[s.symbol] = {"total": 0, "wins": 0, "pnl_sum": 0}
                by_symbol[s.symbol]["total"] += 1
                if (s.outcome_pct or 0) > 0:
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

            def _target_progress(s):
                """How close price got to target (0-100%)."""
                if not s.entry or not s.target or s.entry == s.target:
                    return None
                if s.direction == "BULLISH":
                    best = s.highest_price or s.outcome_price or s.entry
                    move = best - s.entry
                    needed = s.target - s.entry
                else:
                    best = s.lowest_price or s.outcome_price or s.entry
                    move = s.entry - best
                    needed = s.entry - s.target
                if needed <= 0:
                    return None
                return min(100, round(move / needed * 100, 1))

            recent_list = [{
                "symbol": s.symbol,
                "timeframe": s.timeframe,
                "direction": s.direction,
                "entry": s.entry,
                "target": s.target,
                "stop_loss": s.stop_loss,
                "predicted_price": s.predicted_price,
                "status": s.status,
                "outcome_pct": s.outcome_pct,
                "outcome_price": s.outcome_price,
                "highest_price": s.highest_price,
                "lowest_price": s.lowest_price,
                "target_progress": _target_progress(s),
                "confidence": s.confidence,
                "prediction_error": round(abs(s.predicted_price - s.outcome_price) / s.outcome_price * 100, 2)
                    if s.predicted_price and s.outcome_price and s.outcome_price > 0 else None,
                "direction_correct": s.status in ("target_hit", "correct"),
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "resolved_at": s.resolved_at.isoformat() if s.resolved_at else None,
            } for s in recent]

            # Direction accuracy (separate from target hit rate)
            direction_correct = sum(1 for s in resolved if s.status in ("target_hit", "correct"))
            direction_accuracy = round(direction_correct / total * 100, 1) if total > 0 else 0

            # Prediction error stats
            pred_errors = [
                abs(s.predicted_price - s.outcome_price) / s.outcome_price * 100
                for s in resolved
                if s.predicted_price and s.outcome_price and s.outcome_price > 0
            ]
            avg_pred_error = round(sum(pred_errors) / len(pred_errors), 2) if pred_errors else None

            # Open trades count
            open_count = db.query(TradeSignalLog).filter(TradeSignalLog.status == "open").count()

            return {
                "total": total,
                "target_hit": target_hits,
                "sl_hit": sl_hits,
                "correct": correct,
                "wrong": wrong,
                "win_rate": win_rate,
                "direction_accuracy": direction_accuracy,
                "avg_prediction_error": avg_pred_error,
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
