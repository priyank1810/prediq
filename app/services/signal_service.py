import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.data_fetcher import data_fetcher
from app.services.indicator_service import indicator_service
from app.services.sentiment_service import sentiment_service
from app.services.global_market_service import global_market_service
from app.utils.helpers import now_ist, is_market_open
from app.utils.cache import cache
from app.config import (
    SIGNAL_WEIGHT_TECHNICAL, SIGNAL_WEIGHT_SENTIMENT, SIGNAL_WEIGHT_GLOBAL,
    SIGNAL_WEIGHT_FUNDAMENTAL, SIGNAL_DIRECTION_THRESHOLD,
    CACHE_TTL_MTF_DAILY, CACHE_TTL_MTF_1H,
)

logger = logging.getLogger(__name__)


class SignalService:
    def _compute_mtf_confluence(self, symbol, intraday_df, tech_score_15m):
        """Multi-timeframe confluence: 15m, 1h, Daily."""
        timeframes = []

        # 15m — already computed
        dir_15m = "BULLISH" if tech_score_15m > SIGNAL_DIRECTION_THRESHOLD else ("BEARISH" if tech_score_15m < -SIGNAL_DIRECTION_THRESHOLD else "NEUTRAL")
        timeframes.append({"label": "15m", "direction": dir_15m, "score": round(tech_score_15m, 1)})

        # 1h — resample 15m data to 1h (cached)
        score_1h = 0
        dir_1h = "NEUTRAL"
        cache_key_1h = f"mtf_1h_indicators:{symbol}"
        cached_1h = cache.get(cache_key_1h)
        if cached_1h:
            score_1h = cached_1h["score"]
            dir_1h = cached_1h["direction"]
        else:
            try:
                if intraday_df is not None and len(intraday_df) >= 20:
                    df_1h = intraday_df.copy()
                    if "datetime_str" in df_1h.columns:
                        df_1h["_dt"] = pd.to_datetime(df_1h["datetime_str"], format="%Y-%m-%d %H:%M")
                        df_1h = df_1h.set_index("_dt")
                        resampled = df_1h.resample("1h").agg({
                            "open": "first", "high": "max", "low": "min",
                            "close": "last", "volume": "sum"
                        }).dropna()
                        if "datetime_str" not in resampled.columns:
                            resampled["datetime_str"] = resampled.index.strftime("%Y-%m-%d %H:%M")
                        resampled = resampled.reset_index(drop=True)
                        if len(resampled) >= 20:
                            res_1h = indicator_service.compute_intraday_indicators(resampled)
                            score_1h = res_1h["score"]
                            dir_1h = "BULLISH" if score_1h > SIGNAL_DIRECTION_THRESHOLD else ("BEARISH" if score_1h < -SIGNAL_DIRECTION_THRESHOLD else "NEUTRAL")
                            cache.set(cache_key_1h, {"score": score_1h, "direction": dir_1h}, CACHE_TTL_MTF_1H)
            except Exception as e:
                logger.warning(f"MTF 1h failed: {e}")
        timeframes.append({"label": "1h", "direction": dir_1h, "score": round(score_1h, 1)})

        # Daily — fetch 3mo historical (cached)
        score_daily = 0
        dir_daily = "NEUTRAL"
        cache_key_daily = f"mtf_daily_indicators:{symbol}"
        cached_daily = cache.get(cache_key_daily)
        if cached_daily:
            score_daily = cached_daily["score"]
            dir_daily = cached_daily["direction"]
        else:
            try:
                daily_df = data_fetcher.get_historical_data(symbol, period="3mo")
                if daily_df is not None and len(daily_df) >= 20:
                    if "datetime_str" not in daily_df.columns and "date" in daily_df.columns:
                        daily_df["datetime_str"] = daily_df["date"].astype(str)
                    res_daily = indicator_service.compute_intraday_indicators(daily_df)
                    score_daily = res_daily["score"]
                    dir_daily = "BULLISH" if score_daily > SIGNAL_DIRECTION_THRESHOLD else ("BEARISH" if score_daily < -SIGNAL_DIRECTION_THRESHOLD else "NEUTRAL")
                    cache.set(cache_key_daily, {"score": score_daily, "direction": dir_daily}, CACHE_TTL_MTF_DAILY)
            except Exception as e:
                logger.warning(f"MTF daily failed: {e}")
        timeframes.append({"label": "Daily", "direction": dir_daily, "score": round(score_daily, 1)})

        # Confluence scoring
        directions = [tf["direction"] for tf in timeframes]
        bullish_count = directions.count("BULLISH")
        bearish_count = directions.count("BEARISH")
        agreement = max(bullish_count, bearish_count)

        if agreement == 3:
            level = "HIGH"
            boost = 10
        elif agreement == 2:
            level = "MEDIUM"
            boost = 5
        else:
            level = "LOW"
            boost = 0  # No directional agreement → no boost

        # Direction of boost based on majority
        if bearish_count > bullish_count:
            boost = -abs(boost)

        return {
            "level": level,
            "boost": boost,
            "timeframes": timeframes,
            "agreement_count": agreement,
        }

    def get_signal(self, symbol: str) -> dict:
        # Fetch intraday data, sentiment, global market, and OI in parallel
        intraday_df = None
        sent_result = {"score": 0, "headline_count": 0, "positive_count": 0,
                       "negative_count": 0, "neutral_count": 0, "headlines": []}
        global_result = {"score": 0, "markets": [], "news_magnitude": 0}
        oi_result = {"score": 0, "available": False}
        sector_result = {"available": False, "score": 0}
        fund_result = {"score": 0, "classification": "balanced", "details": {}}

        with ThreadPoolExecutor(max_workers=6) as executor:
            future_intraday = executor.submit(
                data_fetcher.get_intraday_data, symbol, "5d", "15m"
            )
            future_sentiment = executor.submit(
                sentiment_service.get_sentiment, symbol
            )
            future_global = executor.submit(
                global_market_service.get_global_signal
            )
            # OI analysis (may not be available)
            future_oi = executor.submit(self._fetch_oi, symbol)
            # Sector-relative strength
            future_sector = executor.submit(self._fetch_sector_strength, symbol)
            # Fundamental analysis
            future_fund = executor.submit(self._fetch_fundamental_score, symbol)

            try:
                intraday_df = future_intraday.result(timeout=15)
            except Exception as e:
                logger.error(f"Intraday data failed for {symbol}: {e}")

            try:
                sent_result = future_sentiment.result(timeout=15)
            except Exception as e:
                logger.warning(f"Sentiment failed for {symbol}: {e}")

            try:
                global_result = future_global.result(timeout=15)
            except Exception as e:
                logger.warning(f"Global market failed: {e}")

            try:
                oi_result = future_oi.result(timeout=15)
            except Exception as e:
                logger.warning(f"OI analysis failed for {symbol}: {e}")

            try:
                sector_result = future_sector.result(timeout=15)
            except Exception as e:
                logger.warning(f"Sector strength failed for {symbol}: {e}")

            try:
                fund_result = future_fund.result(timeout=15)
            except Exception as e:
                logger.warning(f"Fundamental analysis failed for {symbol}: {e}")

        # Technical score (depends on intraday data)
        tech_raw = {}
        if intraday_df is not None and not intraday_df.empty:
            tech_result = indicator_service.compute_intraday_indicators(intraday_df)
            technical_score = tech_result["score"]
            tech_details = tech_result["details"]
            tech_raw = tech_result.get("raw", {})
        else:
            technical_score = 0
            tech_details = {}

        sentiment_score = sent_result["score"]
        global_score = global_result["score"]

        # Sector-aware news impact adjustment
        sector_news_adjustment = None
        try:
            from app.services.sector_service import sector_service
            sector_adj = sector_service.get_sector_adjusted_scores(
                symbol, sentiment_score, global_score, global_result
            )
            sector_news_adjustment = {
                "original_sentiment": sentiment_score,
                "original_global": global_score,
                "adjusted_sentiment": sector_adj["sentiment_score"],
                "adjusted_global": sector_adj["global_score"],
                "sector": sector_adj["sector"],
                "modifier_applied": sector_adj["modifier_applied"],
                "active_events": sector_adj["active_events"],
            }
            sentiment_score = sector_adj["sentiment_score"]
            global_score = sector_adj["global_score"]
        except Exception as e:
            logger.debug(f"Sector news adjustment skipped: {e}")

        # 5. Dynamic weights — adaptive or static
        w_tech = SIGNAL_WEIGHT_TECHNICAL
        w_sent = SIGNAL_WEIGHT_SENTIMENT
        w_glob = SIGNAL_WEIGHT_GLOBAL
        w_fund = SIGNAL_WEIGHT_FUNDAMENTAL
        adaptive_info = {"adapted": False, "sample_size": 0, "component_accuracies": {}}

        try:
            from app.services.adaptive_weights import adaptive_weight_service
            adaptive = adaptive_weight_service.get_weights(symbol)
            if adaptive["adapted"]:
                w_tech = adaptive["weights"]["technical"]
                w_sent = adaptive["weights"]["sentiment"]
                w_glob = adaptive["weights"]["global"]
                w_fund = adaptive["weights"].get("fundamental", SIGNAL_WEIGHT_FUNDAMENTAL)
                adaptive_info = {
                    "adapted": True,
                    "sample_size": adaptive["sample_size"],
                    "component_accuracies": adaptive.get("component_accuracies", {}),
                }
        except Exception as e:
            logger.debug(f"Adaptive weights not available: {e}")

        # Fundamental score: scale from (-1,+1) to (-100,+100) to match other components
        fundamental_score = fund_result.get("score", 0) * 100

        # News-magnitude crisis override (takes precedence — reduce fundamental in crisis)
        news_magnitude = global_result.get("news_magnitude", 0)
        if news_magnitude >= 80:
            w_tech = 0.20
            w_sent = 0.20
            w_glob = 0.50
            w_fund = 0.05  # Fundamentals matter little in panic
        elif news_magnitude >= 60:
            w_tech = 0.30
            w_sent = 0.20
            w_glob = 0.35
            w_fund = 0.10
        elif news_magnitude >= 30 and not adaptive_info["adapted"]:
            boost = (news_magnitude - 30) / 30 * 0.10
            w_glob = SIGNAL_WEIGHT_GLOBAL + boost
            w_tech = SIGNAL_WEIGHT_TECHNICAL - boost * 0.6
            w_fund = SIGNAL_WEIGHT_FUNDAMENTAL - boost * 0.4

        # OI as optional component
        w_oi = 0.0
        oi_score = 0
        if oi_result.get("available") and news_magnitude < 80:
            oi_score = oi_result.get("score", 0)
            w_oi = 0.10
            # Scale other weights down to make room
            scale = 0.9
            w_tech *= scale
            w_sent *= scale
            w_glob *= scale
            w_fund *= scale
            # Renormalize
            total = w_tech + w_sent + w_glob + w_fund + w_oi
            w_tech /= total
            w_sent /= total
            w_glob /= total
            w_fund /= total
            w_oi /= total

        composite = (
            w_tech * technical_score +
            w_sent * sentiment_score +
            w_fund * fundamental_score +
            w_glob * global_score +
            w_oi * oi_score
        )

        # Support/Resistance proximity modifier (±4.5 pts max)
        sr_result = {"levels": {}, "proximity_signal": 0, "trend": "up"}
        if intraday_df is not None and not intraday_df.empty:
            try:
                sr_result = indicator_service.compute_support_resistance(
                    intraday_df, current_price=tech_details.get("current_price")
                )
                composite += sr_result["proximity_signal"] * 0.15
            except Exception as e:
                logger.warning(f"S/R computation failed: {e}")

        # Multi-Timeframe Confluence modifier
        mtf_result = {"level": "LOW", "boost": 0, "timeframes": [], "agreement_count": 0}
        try:
            mtf_result = self._compute_mtf_confluence(symbol, intraday_df, technical_score)
            composite += mtf_result["boost"] * 0.5
        except Exception as e:
            logger.warning(f"MTF confluence failed: {e}")

        # Sector-Relative Strength modifier
        if sector_result.get("available"):
            composite += sector_result["score"] * 0.10

        composite = max(-100, min(100, round(composite, 2)))

        # 6. Direction and confidence
        if composite > SIGNAL_DIRECTION_THRESHOLD:
            direction = "BULLISH"
        elif composite < -SIGNAL_DIRECTION_THRESHOLD:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        confidence = min(100, round(abs(composite), 2))

        # 7. Intraday candles for chart
        candles = []
        if intraday_df is not None and not intraday_df.empty:
            for _, row in intraday_df.tail(52).iterrows():
                candles.append({
                    "time": row.get("datetime_str", ""),
                    "open": round(float(row["open"]), 2),
                    "high": round(float(row["high"]), 2),
                    "low": round(float(row["low"]), 2),
                    "close": round(float(row["close"]), 2),
                    "volume": int(row["volume"]),
                })

        return {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "composite_score": composite,
            "timestamp": now_ist().isoformat(),
            "market_open": is_market_open(),
            "technical": {
                "score": technical_score,
                "weight": round(w_tech, 2),
                "details": tech_details,
                "raw": tech_raw,
            },
            "sentiment": {
                "score": sentiment_score,
                "weight": round(w_sent, 2),
                "headline_count": sent_result.get("headline_count", 0),
                "positive_count": sent_result.get("positive_count", 0),
                "negative_count": sent_result.get("negative_count", 0),
                "neutral_count": sent_result.get("neutral_count", 0),
                "headlines": sent_result.get("headlines", []),
            },
            "global_market": {
                "score": global_score,
                "weight": round(w_glob, 2),
                "news_magnitude": news_magnitude,
                "markets": global_result.get("markets", []),
                "headlines": global_result.get("headlines", []),
            },
            "intraday_candles": candles,
            "support_resistance": {
                "levels": sr_result.get("levels", {}),
                "proximity_signal": sr_result.get("proximity_signal", 0),
                "trend": sr_result.get("trend", "up"),
            },
            "mtf_confluence": mtf_result,
            "adaptive_weights": adaptive_info,
            "oi_analysis": {
                "available": oi_result.get("available", False),
                "score": oi_result.get("score", 0),
                "weight": round(w_oi, 2),
                "pcr": oi_result.get("pcr"),
                "max_pain": oi_result.get("max_pain"),
                "oi_signal": oi_result.get("oi_signal", ""),
                "call_oi_change": oi_result.get("call_oi_change"),
                "put_oi_change": oi_result.get("put_oi_change"),
            },
            "fundamental": {
                "score": fundamental_score,
                "weight": round(w_fund, 2),
                "raw_score": fund_result.get("score", 0),
                "classification": fund_result.get("classification", "balanced"),
                "details": fund_result.get("details", {}),
            },
            "sector_strength": sector_result,
            "sector_news_adjustment": sector_news_adjustment,
        }

    def _fetch_oi(self, symbol: str) -> dict:
        """Fetch OI analysis, returns default if service unavailable."""
        try:
            from app.services.oi_service import oi_service
            return oi_service.get_oi_analysis(symbol)
        except Exception as e:
            logger.debug(f"OI service unavailable: {e}")
            return {"score": 0, "available": False}

    def _fetch_sector_strength(self, symbol: str) -> dict:
        """Fetch sector-relative strength, returns default if unavailable."""
        try:
            from app.services.sector_service import sector_service
            return sector_service.get_sector_strength(symbol)
        except Exception as e:
            logger.debug(f"Sector strength unavailable: {e}")
            return {"available": False, "score": 0}

    def _fetch_fundamental_score(self, symbol: str) -> dict:
        """Fetch fundamental score for signal component."""
        try:
            from app.services.fundamental_service import fundamental_service
            from app.ai.fundamental_model import fundamental_model
            from app.utils.helpers import is_index

            if is_index(symbol):
                return {"score": 0, "classification": "balanced", "details": {}}

            fund_data = fundamental_service.get_fundamentals(symbol)
            if fund_data and fund_data.get("pe"):
                return fundamental_model.score(fund_data, symbol)
        except Exception as e:
            logger.debug(f"Fundamental score unavailable for {symbol}: {e}")
        return {"score": 0, "classification": "balanced", "details": {}}


    # ── Multi-Timeframe Signals with Entry/Exit Levels ──

    def get_multi_timeframe_signals(self, symbol: str) -> dict:
        """Compute separate buy/sell signals for Intraday, Short-term (2 weeks),
        and Long-term horizons with entry, target, and stop-loss levels."""
        from app.services.data_fetcher import data_fetcher

        intraday_df = None
        daily_df = None
        weekly_df = None
        sent_result = {"score": 0, "headline_count": 0}
        global_result = {"score": 0, "news_magnitude": 0}
        fund_result = {"score": 0, "classification": "balanced", "details": {}}

        with ThreadPoolExecutor(max_workers=7) as executor:
            f_intraday = executor.submit(data_fetcher.get_intraday_data, symbol, "5d", "15m")
            f_daily = executor.submit(data_fetcher.get_historical_data, symbol, period="6mo")
            f_weekly = executor.submit(data_fetcher.get_historical_data, symbol, period="2y")
            f_sent = executor.submit(sentiment_service.get_sentiment, symbol)
            f_global = executor.submit(global_market_service.get_global_signal)
            f_fund = executor.submit(self._fetch_fundamental_score, symbol)

            try:
                intraday_df = f_intraday.result(timeout=15)
            except Exception as e:
                logger.warning(f"MTF intraday failed for {symbol}: {e}")
            try:
                daily_df = f_daily.result(timeout=15)
            except Exception as e:
                logger.warning(f"MTF daily failed for {symbol}: {e}")
            try:
                weekly_df = f_weekly.result(timeout=15)
            except Exception as e:
                logger.warning(f"MTF weekly failed for {symbol}: {e}")
            try:
                sent_result = f_sent.result(timeout=15)
            except Exception:
                pass
            try:
                global_result = f_global.result(timeout=15)
            except Exception:
                pass
            try:
                fund_result = f_fund.result(timeout=15)
            except Exception:
                pass

        sentiment_score = sent_result.get("score", 0)
        global_score = global_result.get("score", 0)
        news_magnitude = global_result.get("news_magnitude", 0)
        fundamental_score = fund_result.get("score", 0) * 100

        current_price = 0
        if intraday_df is not None and not intraday_df.empty:
            current_price = float(intraday_df["close"].iloc[-1])
        elif daily_df is not None and not daily_df.empty:
            current_price = float(daily_df["close"].iloc[-1])

        # ── 1. INTRADAY SIGNAL ──
        intraday_signal = self._compute_timeframe_signal(
            label="Intraday",
            df=intraday_df,
            current_price=current_price,
            sentiment_score=sentiment_score,
            global_score=global_score,
            fundamental_score=fundamental_score,
            news_magnitude=news_magnitude,
            timeframe="intraday",
            symbol=symbol,
        )

        # ── 2. SHORT-TERM (2 WEEKS) SIGNAL ──
        short_term_signal = self._compute_timeframe_signal(
            label="Short Term (2 Weeks)",
            df=daily_df,
            current_price=current_price,
            sentiment_score=sentiment_score,
            global_score=global_score,
            fundamental_score=fundamental_score,
            news_magnitude=news_magnitude,
            timeframe="short_term",
            symbol=symbol,
        )

        # ── 3. LONG-TERM SIGNAL ──
        long_term_signal = self._compute_timeframe_signal(
            label="Long Term",
            df=weekly_df,
            current_price=current_price,
            sentiment_score=sentiment_score,
            global_score=global_score,
            fundamental_score=fundamental_score,
            news_magnitude=news_magnitude,
            timeframe="long_term",
            symbol=symbol,
        )

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "timestamp": now_ist().isoformat(),
            "market_open": is_market_open(),
            "intraday": intraday_signal,
            "short_term": short_term_signal,
            "long_term": long_term_signal,
        }

    def _compute_timeframe_signal(self, label, df, current_price, sentiment_score,
                                   global_score, fundamental_score, news_magnitude,
                                   timeframe, symbol):
        """Compute signal + entry/exit for one timeframe."""
        import ta as ta_lib

        if df is None or df.empty or len(df) < 20:
            return {
                "label": label,
                "direction": "NEUTRAL",
                "confidence": 0,
                "score": 0,
                "entry": None,
                "target": None,
                "stop_loss": None,
                "risk_reward": None,
                "reasoning": "Insufficient data",
            }

        # Compute technical indicators for this timeframe
        tech_result = indicator_service.compute_intraday_indicators(df)
        tech_score = tech_result["score"]
        details = tech_result.get("details", {})

        # Timeframe-specific weights
        if timeframe == "intraday":
            w_tech, w_sent, w_glob, w_fund = 0.60, 0.15, 0.15, 0.10
        elif timeframe == "short_term":
            w_tech, w_sent, w_glob, w_fund = 0.45, 0.20, 0.15, 0.20
        else:  # long_term
            w_tech, w_sent, w_glob, w_fund = 0.30, 0.15, 0.15, 0.40

        # Crisis override
        if news_magnitude >= 80:
            w_tech, w_sent, w_glob, w_fund = 0.20, 0.20, 0.50, 0.10
        elif news_magnitude >= 60:
            w_tech, w_sent, w_glob, w_fund = 0.30, 0.20, 0.35, 0.15

        composite = (
            w_tech * tech_score +
            w_sent * sentiment_score +
            w_glob * global_score +
            w_fund * fundamental_score
        )
        composite = max(-100, min(100, round(composite, 2)))

        if composite > SIGNAL_DIRECTION_THRESHOLD:
            direction = "BULLISH"
        elif composite < -SIGNAL_DIRECTION_THRESHOLD:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        confidence = min(100, round(abs(composite), 2))

        # ── Compute Entry / Target / Stop Loss ──
        entry, target, stop_loss, risk_reward, reasoning = self._compute_entry_exit(
            df, current_price, direction, timeframe, details
        )

        return {
            "label": label,
            "direction": direction,
            "confidence": confidence,
            "score": composite,
            "entry": entry,
            "target": target,
            "stop_loss": stop_loss,
            "risk_reward": risk_reward,
            "reasoning": reasoning,
            "weights": {
                "technical": round(w_tech, 2),
                "sentiment": round(w_sent, 2),
                "global": round(w_glob, 2),
                "fundamental": round(w_fund, 2),
            },
            "components": {
                "technical": round(tech_score, 2),
                "sentiment": round(sentiment_score, 2),
                "global": round(global_score, 2),
                "fundamental": round(fundamental_score, 2),
            },
        }

    def _compute_entry_exit(self, df, current_price, direction, timeframe, details):
        """Compute entry, target, stop-loss using S/R, ATR, and Fibonacci levels."""
        import ta as ta_lib

        if current_price <= 0:
            return None, None, None, None, "No price data"

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ATR for volatility-based levels
        try:
            atr_series = ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
            atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else current_price * 0.015
        except Exception:
            atr = current_price * 0.015

        # Swing high/low for the timeframe
        if timeframe == "intraday":
            lookback = min(26, len(df))  # ~1 day of 15m bars
            atr_multiplier_sl = 1.0
            atr_multiplier_target = 1.5
        elif timeframe == "short_term":
            lookback = min(10, len(df))  # ~2 weeks of daily bars
            atr_multiplier_sl = 1.5
            atr_multiplier_target = 2.5
        else:  # long_term
            lookback = min(60, len(df))  # ~3 months
            atr_multiplier_sl = 2.0
            atr_multiplier_target = 4.0

        swing_high = float(high.tail(lookback).max())
        swing_low = float(low.tail(lookback).min())
        fib_range = swing_high - swing_low

        # Fibonacci levels
        fib_382 = swing_high - 0.382 * fib_range
        fib_500 = swing_high - 0.500 * fib_range
        fib_618 = swing_high - 0.618 * fib_range
        fib_236 = swing_high - 0.236 * fib_range

        # EMAs for context
        ema21 = details.get("ema21", current_price)
        vwap = details.get("vwap", current_price)

        reasoning_parts = []

        if direction == "BULLISH":
            # Entry: nearest support below current price (Fib 38.2%, 50%, or VWAP)
            support_candidates = sorted([
                lvl for lvl in [fib_382, fib_500, fib_618, swing_low, ema21, vwap]
                if lvl < current_price
            ], reverse=True)
            entry = support_candidates[0] if support_candidates else current_price * 0.995
            # Don't set entry too far from current price
            if entry < current_price * 0.97:
                entry = current_price * 0.995

            # Stop loss below entry
            stop_loss = entry - (atr * atr_multiplier_sl)
            stop_loss = max(stop_loss, swing_low - atr * 0.5)

            # Target: nearest resistance above current price
            resistance_candidates = sorted([
                lvl for lvl in [fib_236, swing_high]
                if lvl > current_price
            ])
            target = resistance_candidates[0] if resistance_candidates else current_price + atr * atr_multiplier_target
            # Ensure minimum target distance
            if target < current_price * 1.005:
                target = current_price + atr * atr_multiplier_target

            reasoning_parts.append(f"Buy near support ₹{entry:.2f}")
            reasoning_parts.append(f"Target swing high/Fib ₹{target:.2f}")
            reasoning_parts.append(f"Stop below ₹{stop_loss:.2f} (ATR-based)")

        elif direction == "BEARISH":
            # Entry: nearest resistance above current price
            resistance_candidates = sorted([
                lvl for lvl in [fib_382, fib_236, swing_high, ema21, vwap]
                if lvl > current_price
            ])
            entry = resistance_candidates[0] if resistance_candidates else current_price * 1.005
            if entry > current_price * 1.03:
                entry = current_price * 1.005

            # Stop loss above entry
            stop_loss = entry + (atr * atr_multiplier_sl)
            stop_loss = min(stop_loss, swing_high + atr * 0.5)

            # Target: nearest support below current price
            support_candidates = sorted([
                lvl for lvl in [fib_618, fib_500, swing_low]
                if lvl < current_price
            ], reverse=True)
            target = support_candidates[0] if support_candidates else current_price - atr * atr_multiplier_target
            if target > current_price * 0.995:
                target = current_price - atr * atr_multiplier_target

            reasoning_parts.append(f"Sell near resistance ₹{entry:.2f}")
            reasoning_parts.append(f"Target support/Fib ₹{target:.2f}")
            reasoning_parts.append(f"Stop above ₹{stop_loss:.2f} (ATR-based)")

        else:  # NEUTRAL
            entry = round(current_price, 2)
            target = None
            stop_loss = None
            return entry, target, stop_loss, None, "No clear directional bias — wait for confirmation"

        # Risk/Reward ratio
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        entry = round(entry, 2)
        target = round(target, 2)
        stop_loss = round(stop_loss, 2)

        reasoning_parts.append(f"Risk:Reward = 1:{risk_reward:.1f}")
        reasoning = " | ".join(reasoning_parts)

        return entry, target, stop_loss, risk_reward, reasoning


signal_service = SignalService()
