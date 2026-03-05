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
    SIGNAL_DIRECTION_THRESHOLD,
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

        with ThreadPoolExecutor(max_workers=5) as executor:
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
        adaptive_info = {"adapted": False, "sample_size": 0, "component_accuracies": {}}

        try:
            from app.services.adaptive_weights import adaptive_weight_service
            adaptive = adaptive_weight_service.get_weights(symbol)
            if adaptive["adapted"]:
                w_tech = adaptive["weights"]["technical"]
                w_sent = adaptive["weights"]["sentiment"]
                w_glob = adaptive["weights"]["global"]
                adaptive_info = {
                    "adapted": True,
                    "sample_size": adaptive["sample_size"],
                    "component_accuracies": adaptive.get("component_accuracies", {}),
                }
        except Exception as e:
            logger.debug(f"Adaptive weights not available: {e}")

        # News-magnitude crisis override (takes precedence)
        news_magnitude = global_result.get("news_magnitude", 0)
        if news_magnitude >= 80:
            w_tech = 0.25
            w_sent = 0.25
            w_glob = 0.50
        elif news_magnitude >= 60:
            w_tech = 0.35
            w_sent = 0.25
            w_glob = 0.40
        elif news_magnitude >= 30 and not adaptive_info["adapted"]:
            boost = (news_magnitude - 30) / 30 * 0.15
            w_glob = SIGNAL_WEIGHT_GLOBAL + boost
            w_tech = SIGNAL_WEIGHT_TECHNICAL - boost

        # OI as 4th component
        w_oi = 0.0
        oi_score = 0
        if oi_result.get("available") and news_magnitude < 80:
            oi_score = oi_result.get("score", 0)
            w_oi = 0.10
            # Scale other weights down by 0.9 to make room
            scale = 0.9
            w_tech *= scale
            w_sent *= scale
            w_glob *= scale
            # Renormalize
            total = w_tech + w_sent + w_glob + w_oi
            w_tech /= total
            w_sent /= total
            w_glob /= total
            w_oi /= total

        composite = (
            w_tech * technical_score +
            w_sent * sentiment_score +
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


signal_service = SignalService()
