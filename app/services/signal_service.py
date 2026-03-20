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

        # 5. Dynamic weights — per-stock learning first, then sector adaptive, then static
        w_tech = SIGNAL_WEIGHT_TECHNICAL
        w_sent = SIGNAL_WEIGHT_SENTIMENT
        w_glob = SIGNAL_WEIGHT_GLOBAL
        w_fund = SIGNAL_WEIGHT_FUNDAMENTAL
        adaptive_info = {"adapted": False, "sample_size": 0, "component_accuracies": {}}
        stock_profile = None

        # Try per-stock learned weights first (most specific)
        try:
            from app.services.stock_learner import stock_learner
            stock_profile = stock_learner.get_profile(symbol)
            if stock_profile:
                w_tech = stock_profile["weights"]["technical"]
                w_sent = stock_profile["weights"]["sentiment"]
                w_glob = stock_profile["weights"]["global"]
                w_fund = stock_profile["weights"]["fundamental"]
                adaptive_info = {
                    "adapted": True,
                    "sample_size": stock_profile["sample_size"],
                    "component_accuracies": stock_profile.get("component_accuracies", {}),
                    "source": "per_stock",
                    "trend": stock_profile.get("trend", "stable"),
                    "overall_accuracy": stock_profile.get("overall_accuracy", 0),
                }
        except Exception as e:
            logger.debug(f"Per-stock learning not available for {symbol}: {e}")

        # Fall back to sector-level adaptive weights if no stock profile
        if not stock_profile:
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
                        "source": "sector",
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

        # 6. Direction and confidence — per-stock learned threshold, then volatility-adaptive
        dir_threshold = SIGNAL_DIRECTION_THRESHOLD
        if stock_profile and stock_profile.get("optimal_threshold"):
            dir_threshold = stock_profile["optimal_threshold"]
        elif intraday_df is not None and not intraday_df.empty and "close" in intraday_df.columns:
            try:
                recent_returns = intraday_df["close"].pct_change().dropna().tail(50)
                if len(recent_returns) > 10:
                    vol = recent_returns.std() * 100  # as percentage
                    # Low vol → tighter threshold (more signals), high vol → wider (fewer false signals)
                    dir_threshold = max(5, min(20, SIGNAL_DIRECTION_THRESHOLD * (vol / 1.0)))
            except Exception:
                pass

        if composite > dir_threshold:
            direction = "BULLISH"
        elif composite < -dir_threshold:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        confidence = min(100, round(abs(composite), 2))

        # Calibrate confidence with historical accuracy (if adaptive weights available)
        if adaptive_info["adapted"] and adaptive_info.get("component_accuracies"):
            avg_accuracy = sum(adaptive_info["component_accuracies"].values()) / max(
                len(adaptive_info["component_accuracies"]), 1
            )
            # Scale confidence: if avg accuracy is 60%, dampen confidence by 0.85
            # If avg accuracy is 80%, boost slightly by 1.05
            cal_factor = 0.5 + avg_accuracy / 200  # maps 0%→0.5, 100%→1.0
            confidence = min(100, round(confidence * cal_factor, 2))

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

    _TIMEFRAME_HORIZON_MAP = {
        "intraday": "1d",
        "short_term": "1w",
        "long_term": "3mo",
    }

    def _run_prediction_for_horizon(self, symbol: str, horizon: str,
                                     intraday_df=None, daily_df=None):
        """Run full XGBoost + Prophet prediction for a given horizon.
        Returns a rich dict with score, predicted_price, confidence,
        regime, volume conviction, confidence intervals, and contribution breakdown."""
        empty = {
            "score": 0, "predicted_price": None, "confidence": None,
            "regime": None, "volume_conviction": None,
            "confidence_lower": None, "confidence_upper": None,
            "contribution": None, "ensemble_method": None,
            "xgboost_weight": None, "prophet_weight": None,
        }
        try:
            from app.services.prediction_service import prediction_service
            from app.config import PREDICTION_HORIZONS
            import math

            cfg = PREDICTION_HORIZONS.get(horizon, {})
            is_intraday = cfg.get("intraday", False)

            result = prediction_service.predict(symbol, horizon=horizon)
            ensemble = result.get("ensemble")
            if not ensemble or not ensemble.get("predictions"):
                return empty

            predicted_price = ensemble["predictions"][-1]

            # Determine current price from the data used
            current_price = 0
            if is_intraday and intraday_df is not None and not intraday_df.empty:
                current_price = float(intraday_df["close"].iloc[-1])
            elif daily_df is not None and not daily_df.empty:
                current_price = float(daily_df["close"].iloc[-1])

            if current_price <= 0:
                return {**empty, "predicted_price": predicted_price}

            change_pct = ((predicted_price - current_price) / current_price) * 100

            horizon_days = cfg.get("days", 1) if not is_intraday else 1
            # Use sub-linear scaling: sqrt(days) grows too fast for long horizons,
            # dampening scores excessively. Cap effective days and use lower base vol.
            effective_days = min(horizon_days, 30) + max(0, horizon_days - 30) * 0.3
            expected_move = 1.5 * math.sqrt(max(1, effective_days))
            pred_score = (change_pct / expected_move) * 50
            pred_score = max(-100, min(100, round(pred_score, 2)))

            confidence_score = ensemble.get("confidence_score") or result.get("xgboost", {}).get("confidence_score")

            # Extract regime info
            regime_info = result.get("regime")
            regime_label = regime_info.get("label") if regime_info else None

            # Extract volume conviction
            vol_analysis = result.get("volume_analysis", {})
            volume_conviction = vol_analysis.get("conviction")  # "high", "moderate", "low"
            supports_prediction = vol_analysis.get("supports_prediction", False)

            # Extract Prophet confidence intervals (if available)
            prophet_data = result.get("prophet", {})
            conf_lower = prophet_data.get("confidence_lower", [None])
            conf_upper = prophet_data.get("confidence_upper", [None])
            confidence_lower = conf_lower[-1] if conf_lower else None
            confidence_upper = conf_upper[-1] if conf_upper else None

            # Contribution breakdown
            contribution = result.get("contribution_breakdown")

            return {
                "score": pred_score,
                "predicted_price": predicted_price,
                "confidence": confidence_score,
                "regime": regime_label,
                "volume_conviction": volume_conviction,
                "volume_supports": supports_prediction,
                "confidence_lower": confidence_lower,
                "confidence_upper": confidence_upper,
                "contribution": contribution,
                "ensemble_method": ensemble.get("method"),
                "xgboost_weight": ensemble.get("xgboost_weight"),
                "prophet_weight": ensemble.get("prophet_weight"),
            }
        except Exception as e:
            logger.warning(f"Prediction for {symbol} horizon={horizon} failed: {e}")
            return empty

    def get_multi_timeframe_signals(self, symbol: str) -> dict:
        """Compute separate buy/sell signals for Intraday, Short-term (2 weeks),
        and Long-term horizons using ML prediction models + technical analysis."""

        intraday_df = None
        daily_df = None
        weekly_df = None
        sent_result = {"score": 0, "headline_count": 0}
        global_result = {"score": 0, "news_magnitude": 0}
        fund_result = {"score": 0, "classification": "balanced", "details": {}}
        pred_results = {}  # horizon -> (score, predicted_price, confidence)

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Data fetches
            f_intraday = executor.submit(data_fetcher.get_intraday_data, symbol, "5d", "15m")
            f_daily = executor.submit(data_fetcher.get_historical_data, symbol, period="6mo")
            f_weekly = executor.submit(data_fetcher.get_historical_data, symbol, period="2y")
            f_sent = executor.submit(sentiment_service.get_sentiment, symbol)
            f_global = executor.submit(global_market_service.get_global_signal)
            f_fund = executor.submit(self._fetch_fundamental_score, symbol)

            # Wait for data first (predictions need it)
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

        # Run ML predictions for each horizon (in parallel)
        with ThreadPoolExecutor(max_workers=3) as pred_executor:
            pred_futures = {}
            for tf, horizon in self._TIMEFRAME_HORIZON_MAP.items():
                pred_futures[tf] = pred_executor.submit(
                    self._run_prediction_for_horizon, symbol, horizon,
                    intraday_df=intraday_df, daily_df=daily_df,
                )
            for tf, future in pred_futures.items():
                try:
                    pred_results[tf] = future.result(timeout=90)
                except Exception as e:
                    logger.warning(f"MTF prediction for {tf} failed: {e}")
                    pred_results[tf] = {
                        "score": 0, "predicted_price": None, "confidence": None,
                        "regime": None, "volume_conviction": None,
                        "confidence_lower": None, "confidence_upper": None,
                        "contribution": None, "ensemble_method": None,
                        "xgboost_weight": None, "prophet_weight": None,
                    }

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
            prediction=pred_results.get("intraday", {}),
        )

        # ── 2. SHORT-TERM (1 Week) SIGNAL ──
        short_term_signal = self._compute_timeframe_signal(
            label="Short Term (1 Week)",
            df=daily_df,
            current_price=current_price,
            sentiment_score=sentiment_score,
            global_score=global_score,
            fundamental_score=fundamental_score,
            news_magnitude=news_magnitude,
            timeframe="short_term",
            symbol=symbol,
            prediction=pred_results.get("short_term", {}),
        )

        # ── 3. LONG-TERM SIGNAL (3 Months) ──
        long_term_signal = self._compute_timeframe_signal(
            label="Long Term (3 Months)",
            df=weekly_df,
            current_price=current_price,
            sentiment_score=sentiment_score,
            global_score=global_score,
            fundamental_score=fundamental_score,
            news_magnitude=news_magnitude,
            timeframe="long_term",
            symbol=symbol,
            prediction=pred_results.get("long_term", {}),
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
                                   timeframe, symbol, prediction=None):
        """Compute signal + entry/exit for one timeframe using full ML prediction output."""
        if prediction is None:
            prediction = {}

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

        # Full ML prediction output
        pred_score = prediction.get("score", 0)
        predicted_price = prediction.get("predicted_price")
        pred_confidence = prediction.get("confidence")  # 0-100 model confidence
        regime = prediction.get("regime")  # "bull", "bear", "sideways", "volatile"
        volume_conviction = prediction.get("volume_conviction")  # "high", "moderate", "low"
        volume_supports = prediction.get("volume_supports", False)
        confidence_lower = prediction.get("confidence_lower")
        confidence_upper = prediction.get("confidence_upper")

        # ── Base weights by timeframe ──
        if timeframe == "intraday":
            w_pred, w_tech, w_sent, w_glob, w_fund = 0.35, 0.30, 0.15, 0.10, 0.10
        elif timeframe == "short_term":
            w_pred, w_tech, w_sent, w_glob, w_fund = 0.35, 0.20, 0.15, 0.10, 0.20
        else:  # long_term
            w_pred, w_tech, w_sent, w_glob, w_fund = 0.30, 0.15, 0.10, 0.10, 0.35

        # If prediction failed, redistribute its weight to technical
        if pred_score == 0 and predicted_price is None:
            w_tech += w_pred
            w_pred = 0

        # ── Confidence-based weight adjustment ──
        # High confidence prediction → boost prediction weight
        # Low confidence → reduce prediction weight, boost technical
        if pred_confidence is not None and w_pred > 0:
            if pred_confidence >= 70:
                boost = 0.10
                w_pred += boost
                w_tech -= boost * 0.5
                w_sent -= boost * 0.25
                w_glob -= boost * 0.25
            elif pred_confidence < 40:
                penalty = 0.10
                w_pred -= penalty
                w_tech += penalty

        # ── Regime-aware adjustment ──
        if regime and w_pred > 0:
            if regime == "sideways":
                # In sideways/choppy markets, models are less reliable
                shift = 0.08
                w_pred -= shift
                w_tech += shift * 0.5
                w_fund += shift * 0.5
            elif regime == "volatile":
                # In volatile markets, widen the blend — no single source dominates
                shift = 0.05
                w_pred -= shift
                w_tech -= shift
                w_sent += shift
                w_glob += shift
            elif regime in ("bull", "bear"):
                # In trending markets, trust the model more
                shift = 0.05
                w_pred += shift
                w_tech -= shift * 0.5
                w_fund -= shift * 0.5

        # Clamp all weights to [0, 1]
        w_pred = max(0, w_pred)
        w_tech = max(0, w_tech)
        w_sent = max(0, w_sent)
        w_glob = max(0, w_glob)
        w_fund = max(0, w_fund)

        # Re-normalize to sum to 1.0
        w_total = w_pred + w_tech + w_sent + w_glob + w_fund
        if w_total > 0:
            w_pred /= w_total
            w_tech /= w_total
            w_sent /= w_total
            w_glob /= w_total
            w_fund /= w_total

        # Crisis override — global news dominates
        if news_magnitude >= 80:
            w_pred, w_tech, w_sent, w_glob, w_fund = (w_pred * 0.4), 0.10, 0.15, 0.50, 0.05
            remaining = 1.0 - (w_pred + 0.10 + 0.15 + 0.50 + 0.05)
            w_tech += remaining
        elif news_magnitude >= 60:
            w_pred, w_tech, w_sent, w_glob, w_fund = (w_pred * 0.6), 0.15, 0.15, 0.35, 0.10
            remaining = 1.0 - (w_pred + 0.15 + 0.15 + 0.35 + 0.10)
            w_tech += remaining

        composite = (
            w_pred * pred_score +
            w_tech * tech_score +
            w_sent * sentiment_score +
            w_glob * global_score +
            w_fund * fundamental_score
        )

        # ── Volume conviction modifier ──
        # If volume strongly supports the prediction direction, boost composite
        if volume_supports and volume_conviction == "high" and pred_score != 0:
            direction_sign = 1 if pred_score > 0 else -1
            composite += direction_sign * 5  # +/- 5 pts for strong volume confirmation

        composite = max(-100, min(100, round(composite, 2)))

        # Timeframe-specific direction thresholds:
        # Long-term composites are naturally smaller (dampened prediction scores +
        # small fundamental scores), so use a lower threshold to avoid always-neutral.
        dir_thresholds = {"intraday": 10, "short_term": 8, "long_term": 5}
        dir_threshold = dir_thresholds.get(timeframe, SIGNAL_DIRECTION_THRESHOLD)

        if composite > dir_threshold:
            direction = "BULLISH"
        elif composite < -dir_threshold:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Confidence: blend composite magnitude with model confidence
        base_confidence = abs(composite)
        if pred_confidence is not None and w_pred > 0:
            # Weight model confidence into overall confidence
            confidence = min(100, round(base_confidence * 0.6 + pred_confidence * 0.4, 2))
        else:
            confidence = min(100, round(base_confidence, 2))

        # ── Compute Entry / Target / Stop Loss ──
        entry, target, stop_loss, risk_reward, reasoning = self._compute_entry_exit(
            df, current_price, direction, timeframe, details,
            predicted_price=predicted_price,
            pred_confidence=pred_confidence,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            volume_conviction=volume_conviction,
            regime=regime,
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
                "prediction": round(w_pred, 3),
                "technical": round(w_tech, 3),
                "sentiment": round(w_sent, 3),
                "global": round(w_glob, 3),
                "fundamental": round(w_fund, 3),
            },
            "components": {
                "prediction": round(pred_score, 2),
                "technical": round(tech_score, 2),
                "sentiment": round(sentiment_score, 2),
                "global": round(global_score, 2),
                "fundamental": round(fundamental_score, 2),
            },
            "predicted_price": round(predicted_price, 2) if predicted_price else None,
            "regime": regime,
            "volume_conviction": volume_conviction,
            "model_confidence": round(pred_confidence, 1) if pred_confidence else None,
        }

    def _compute_entry_exit(self, df, current_price, direction, timeframe, details,
                            predicted_price=None, pred_confidence=None,
                            confidence_lower=None, confidence_upper=None,
                            volume_conviction=None, regime=None):
        """Compute entry, target, stop-loss using S/R, ATR, Fibonacci levels,
        ML predicted price, confidence intervals, and regime context."""
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
            lookback = min(26, len(df))
            atr_multiplier_sl = 1.0
            atr_multiplier_target = 1.5
        elif timeframe == "short_term":
            lookback = min(10, len(df))
            atr_multiplier_sl = 1.5
            atr_multiplier_target = 2.5
        else:  # long_term
            lookback = min(60, len(df))
            atr_multiplier_sl = 2.0
            atr_multiplier_target = 4.0

        # ── Regime & volume-based ATR adjustments ──
        # Volatile regime → widen stop-loss for breathing room
        if regime == "volatile":
            atr_multiplier_sl *= 1.3
        # Low volume conviction → widen stop-loss (uncertain move)
        if volume_conviction == "low":
            atr_multiplier_sl *= 1.2
        # High volume conviction → tighter stop (strong directional move)
        elif volume_conviction == "high":
            atr_multiplier_sl *= 0.85
            atr_multiplier_target *= 1.15  # extend target with conviction

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

        # ── Determine AI/technical blend ratio from model confidence ──
        # High confidence → trust AI more; low confidence → trust technicals
        if pred_confidence is not None and pred_confidence >= 60:
            ai_weight = 0.70
        elif pred_confidence is not None and pred_confidence >= 40:
            ai_weight = 0.55
        else:
            ai_weight = 0.40  # low confidence or no prediction
        tech_weight = 1.0 - ai_weight

        reasoning_parts = []

        if direction == "BULLISH":
            # Entry: nearest support below current price
            support_candidates = sorted([
                lvl for lvl in [fib_382, fib_500, fib_618, swing_low, ema21, vwap]
                if lvl < current_price
            ], reverse=True)
            entry = support_candidates[0] if support_candidates else current_price * 0.995
            if entry < current_price * 0.97:
                entry = current_price * 0.995

            # Stop loss: use confidence_lower from Prophet if available
            atr_stop = entry - (atr * atr_multiplier_sl)
            if confidence_lower is not None and confidence_lower < entry:
                # Blend Prophet lower bound with ATR-based stop
                stop_loss = confidence_lower * 0.4 + atr_stop * 0.6
            else:
                stop_loss = atr_stop
            stop_loss = max(stop_loss, swing_low - atr * 0.5)
            # Floor: never more than 4% below entry
            min_sl = entry * 0.96
            stop_loss = max(stop_loss, min_sl)

            # Target: blend AI predicted price with technical target
            resistance_candidates = sorted([
                lvl for lvl in [fib_236, swing_high]
                if lvl > current_price
            ])
            tech_target = resistance_candidates[0] if resistance_candidates else current_price + atr * atr_multiplier_target

            if predicted_price and predicted_price > current_price * 1.005:
                target = predicted_price * ai_weight + tech_target * tech_weight
                # If we have upper confidence bound, cap target conservatively
                if confidence_upper is not None and confidence_upper > current_price:
                    # Don't exceed the upper confidence interval
                    target = min(target, confidence_upper)
                reasoning_parts.append(f"AI ₹{predicted_price:.2f}")
            else:
                target = tech_target

            if target < current_price * 1.005:
                target = current_price + atr * atr_multiplier_target

            reasoning_parts.append(f"Buy near ₹{entry:.2f}")
            reasoning_parts.append(f"Target ₹{target:.2f}")
            reasoning_parts.append(f"SL ₹{stop_loss:.2f}")

        elif direction == "BEARISH":
            # Bearish for long-only investors: "exit your position"
            # Entry = current price (where to sell/exit)
            entry = current_price

            # Stop loss: BELOW current price — if price drops to here, exit to limit losses
            atr_stop = entry - (atr * atr_multiplier_sl)
            if confidence_lower is not None and confidence_lower < entry:
                stop_loss = confidence_lower * 0.4 + atr_stop * 0.6
            else:
                stop_loss = atr_stop
            stop_loss = max(stop_loss, swing_low - atr * 0.5)
            # Floor: never more than 4% below entry
            min_sl = entry * 0.96
            stop_loss = max(stop_loss, min_sl)

            # Target: expected downside level (how far price may fall)
            support_candidates = sorted([
                lvl for lvl in [fib_618, fib_500, swing_low]
                if lvl < current_price
            ], reverse=True)
            tech_target = support_candidates[0] if support_candidates else current_price - atr * atr_multiplier_target

            if predicted_price and predicted_price < current_price * 0.995:
                target = predicted_price * ai_weight + tech_target * tech_weight
                if confidence_lower is not None and confidence_lower < current_price:
                    target = max(target, confidence_lower)
                reasoning_parts.append(f"AI ₹{predicted_price:.2f}")
            else:
                target = tech_target

            if target > current_price * 0.995:
                target = current_price - atr * atr_multiplier_target

            reasoning_parts.append(f"Exit near ₹{entry:.2f}")
            reasoning_parts.append(f"Downside ₹{target:.2f}")
            reasoning_parts.append(f"SL ₹{stop_loss:.2f}")

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

        # Add context badges to reasoning
        if regime:
            reasoning_parts.append(regime.capitalize())
        if volume_conviction and volume_conviction != "moderate":
            reasoning_parts.append(f"Vol:{volume_conviction}")
        reasoning_parts.append(f"R:R 1:{risk_reward:.1f}")
        reasoning = " | ".join(reasoning_parts)

        return entry, target, stop_loss, risk_reward, reasoning


signal_service = SignalService()
