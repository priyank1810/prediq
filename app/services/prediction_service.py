from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import ta
from concurrent.futures import ThreadPoolExecutor
from app.ai.xgboost_model import XGBoostPredictor
from app.ai.explainer import prediction_explainer
from app.services.data_fetcher import data_fetcher
from app.config import PREDICTION_HORIZONS
from app.utils.cache import cache

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self):
        self.xgboost = XGBoostPredictor()

    def _detect_regime(self, df: pd.DataFrame) -> dict:
        """Detect market regime using HMM when available, ADX heuristic as fallback."""
        try:
            from app.ai.regime_detector import regime_detector
            regime = regime_detector.detect(df)
            # Add backward-compatible keys
            regime["trending"] = regime.get("regime") in ("bull", "bear")
            regime["high_vol"] = regime.get("regime") == "volatile"
            regime["adx"] = regime.get("adx", 20.0)
            return regime
        except Exception as e:
            logger.debug(f"HMM regime detection failed, using heuristic: {e}")

        # Fallback to simple heuristic
        close = df["close"]
        high = df["high"]
        low = df["low"]

        try:
            adx_val = float(ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1])
        except Exception:
            adx_val = 20.0

        trending = adx_val > 25

        returns = close.pct_change()
        vol_20 = float(returns.rolling(20).std().iloc[-1]) if len(returns) >= 20 else 0.01
        vol_60 = float(returns.rolling(60).std().iloc[-1]) if len(returns) >= 60 else vol_20
        vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
        high_vol = vol_ratio > 1.2

        # Determine regime label
        avg_ret = float(returns.tail(20).mean()) if len(returns) >= 20 else 0
        if trending and avg_ret > 0:
            regime_label = "bull"
        elif trending and avg_ret <= 0:
            regime_label = "bear"
        elif high_vol:
            regime_label = "volatile"
        else:
            regime_label = "sideways"

        return {
            "regime": regime_label,
            "confidence": 0.6,
            "trending": trending,
            "high_vol": high_vol,
            "adx": round(adx_val, 2),
            "vol_ratio": round(vol_ratio, 3),
            "regime_onehot": {
                "bull": 1 if regime_label == "bull" else 0,
                "bear": 1 if regime_label == "bear" else 0,
                "sideways": 1 if regime_label == "sideways" else 0,
                "volatile": 1 if regime_label == "volatile" else 0,
            },
        }


    def _compute_volume_analysis(self, df: pd.DataFrame, ensemble_result: dict | None = None) -> dict:
        """Compute volume analysis: ratio, trend, conviction relative to prediction direction."""
        result = {}
        if df is None or df.empty or "volume" not in df.columns:
            return result

        vol = df["volume"].dropna()
        if len(vol) < 5:
            return result

        current_volume = int(vol.iloc[-1])
        avg_20 = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
        volume_ratio = round(current_volume / avg_20, 2) if avg_20 > 0 else 1.0

        # Volume trend: 5-day SMA slope
        sma5 = vol.rolling(5).mean()
        sma5_now = float(sma5.iloc[-1]) if len(sma5) >= 5 and pd.notna(sma5.iloc[-1]) else avg_20
        vol_trend_pct = round(((sma5_now - avg_20) / avg_20) * 100, 1) if avg_20 > 0 else 0.0

        if vol_trend_pct > 5:
            volume_trend = "increasing"
        elif vol_trend_pct < -5:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"

        # Determine prediction direction from ensemble
        direction = "neutral"
        if ensemble_result and ensemble_result.get("predictions"):
            preds = ensemble_result["predictions"]
            last_close = float(df["close"].iloc[-1])
            last_pred = preds[-1]
            if last_pred > last_close * 1.001:
                direction = "bullish"
            elif last_pred < last_close * 0.999:
                direction = "bearish"

        # Conviction logic
        if volume_ratio > 1.2:
            if volume_trend == "increasing":
                conviction = "high"
                supports = True
                detail = f"Above-average volume supports {direction} prediction"
            elif volume_trend == "decreasing":
                conviction = "moderate"
                supports = direction == "neutral"
                detail = "Volume above average but declining — watch for reversal"
            else:
                conviction = "high"
                supports = True
                detail = f"Strong volume confirms {direction} move"
        elif volume_ratio < 0.8:
            conviction = "low"
            supports = False
            if direction != "neutral":
                detail = f"Low volume undermines {direction} prediction — move lacks participation"
            else:
                detail = "Below-average volume — low conviction environment"
        else:
            conviction = "moderate"
            supports = True
            detail = f"Average volume — moderate conviction for {direction} outlook"

        return {
            "current_volume": current_volume,
            "avg_volume_20d": int(round(avg_20)),
            "volume_ratio": volume_ratio,
            "volume_trend": volume_trend,
            "volume_trend_pct": vol_trend_pct,
            "supports_prediction": supports,
            "conviction": conviction,
            "conviction_detail": detail,
        }

    def _get_fundamental_bias(self, symbol: str) -> dict | None:
        """Get fundamental score to feed into meta-learner."""
        try:
            from app.services.fundamental_service import fundamental_service
            from app.ai.fundamental_model import fundamental_model

            fundamentals = fundamental_service.get_fundamentals(symbol)
            if fundamentals and fundamentals.get("pe"):
                score = fundamental_model.score(fundamentals, symbol)
                return score
        except Exception as e:
            logger.debug(f"Fundamental bias fetch failed: {e}")
        return None

    def _get_shap_drivers(self, symbol: str, daily_df: pd.DataFrame) -> list | None:
        """Get SHAP-based feature importance for the XGBoost model."""
        try:
            from app.ai.shap_explainer import shap_explainer
            xgb_path = self.xgboost._model_path(symbol)
            if os.path.exists(xgb_path):
                saved = joblib.load(xgb_path)
                model = saved["model"]
                feature_cols = saved.get("feature_cols", [])

                feat_df = self.xgboost._build_tabular_features(daily_df, symbol=symbol)
                available_cols = [c for c in feature_cols if c in feat_df.columns]
                if available_cols and len(feat_df) > 0:
                    X = feat_df[available_cols].values[-10:]  # Last 10 rows
                    drivers = shap_explainer.explain_xgboost(model, X, available_cols)
                    return drivers
        except Exception as e:
            logger.debug(f"SHAP explanation failed: {e}")
        return None

    def _apply_market_context_adjustment(self, ensemble_preds: list, current_price: float,
                                            sentiment: dict | None, global_data: dict | None,
                                            horizon: str = "1d") -> tuple[list, dict]:
        """Adjust ensemble predictions based on sentiment and global market context.
        Returns (adjusted_predictions, adjustment_info).

        Context move and weight are scaled by horizon — short horizons get
        minimal adjustment since sentiment can't move a stock 5% in 15 minutes.
        """
        if not ensemble_preds or current_price <= 0:
            return ensemble_preds, {}

        sent_score = sentiment.get("score", 0) if sentiment else 0      # -100 to +100
        global_score = global_data.get("score", 0) if global_data else 0  # -100 to +100
        news_magnitude = global_data.get("news_magnitude", 0) if global_data else 0

        # Blend sentiment and global — global dominates during big events
        if news_magnitude >= 60:
            blended = sent_score * 0.4 + global_score * 0.6
        elif news_magnitude >= 30:
            blended = sent_score * 0.5 + global_score * 0.5
        else:
            blended = sent_score * 0.7 + global_score * 0.3

        if abs(blended) < 3:
            return ensemble_preds, {"adjustment_pct": 0, "blended_score": round(blended, 1),
                                    "news_magnitude": news_magnitude}

        # Horizon-scaled max context move — 15m can't move like 1mo
        cfg = PREDICTION_HORIZONS.get(horizon, {})
        is_intraday = cfg.get("intraday", False)
        horizon_days = cfg.get("days", 1) if not is_intraday else (cfg.get("bars", 1) * 15 / (6.25 * 60))
        # Scale: 15m→0.3%, 1h→0.8%, 1d→2%, 1w→4%, 1mo→8%
        max_context_move = min(8.0, max(0.3, horizon_days * 2.0))
        context_change_pct = (blended / 100.0) * max_context_move

        # Scale context weight down for short horizons
        if news_magnitude >= 80:
            base_context_w = 0.75
        elif news_magnitude >= 60:
            base_context_w = 0.55
        elif news_magnitude >= 30:
            base_context_w = 0.30
        else:
            base_context_w = 0.10

        # Horizon-aware context weight scaling
        if is_intraday:
            base_context_w *= 0.2
        elif horizon_days <= 1:
            base_context_w *= 1.0
        elif horizon_days <= 7:
            base_context_w *= 0.8
        elif horizon_days >= 90:
            base_context_w *= 0.15
        elif horizon_days >= 30:
            base_context_w *= 0.4

        context_w = base_context_w
        model_w = 1.0 - context_w

        # Compute context target for each prediction step
        adjusted = []
        for pred in ensemble_preds:
            context_target = current_price * (1 + context_change_pct / 100)
            adj = pred * model_w + context_target * context_w
            adjusted.append(round(adj, 2))

        # Compute effective adjustment for reporting
        last_orig = ensemble_preds[-1]
        last_adj = adjusted[-1]
        effective_adj_pct = ((last_adj - last_orig) / last_orig * 100) if last_orig else 0

        return adjusted, {
            "adjustment_pct": round(effective_adj_pct, 3),
            "blended_score": round(blended, 1),
            "news_magnitude": news_magnitude,
            "context_weight": round(context_w, 2),
            "model_weight": round(model_w, 2),
        }

    def _compute_contribution_breakdown(self, ensemble_result: dict, adj_info: dict = None,
                                        fund_adj_info: dict = None) -> dict:
        """Compute contribution breakdown for the prediction."""
        context_w = adj_info.get("context_weight", 0) if adj_info else 0
        sentiment_share = min(45.0, round(context_w * 55, 1))

        fund_share = 0.0
        if fund_adj_info and fund_adj_info.get("fund_score"):
            fund_share = min(15.0, round(abs(fund_adj_info["fund_score"]) * 15, 1))

        technical = 100.0 - sentiment_share - fund_share

        return {
            "technical": round(technical, 1),
            "fundamental": round(fund_share, 1),
            "sentiment": round(sentiment_share, 1),
        }

    def predict(self, symbol: str, horizon: str = "1d", models: list = None) -> dict:
        if models is None:
            models = ["xgboost"]

        cfg = PREDICTION_HORIZONS.get(horizon, {})
        is_intraday = cfg.get("intraday", False)

        result = {
            "symbol": symbol, "horizon": horizon, "horizon_label": cfg.get("label", horizon),
            "models_used": models,
        }

        model_results = {}

        if is_intraday:
            intraday_df = data_fetcher.get_intraday_data(symbol, period="5d", interval="15m")
            daily_df = data_fetcher.get_historical_data(symbol, period="2y")

            # Run models + sentiment/global fetches in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                if "xgboost" in models and intraday_df is not None and not intraday_df.empty:
                    futures["xgboost"] = executor.submit(self.xgboost.predict_intraday, intraday_df, symbol, horizon)

                from app.services.sentiment_service import sentiment_service
                from app.services.global_market_service import global_market_service
                futures["_sentiment"] = executor.submit(sentiment_service.get_sentiment, symbol)
                futures["_global"] = executor.submit(global_market_service.get_global_signal)

                for name, future in futures.items():
                    if name.startswith("_"):
                        continue
                    try:
                        model_result = future.result(timeout=60)
                        result[name] = model_result
                        model_results[name] = model_result
                    except Exception as e:
                        result[f"{name}_error"] = str(e)

                sentiment = None
                global_data = None
                try:
                    sentiment = futures["_sentiment"].result(timeout=15)
                except Exception:
                    pass
                try:
                    global_data = futures["_global"].result(timeout=15)
                except Exception:
                    pass

                # Sector-aware news impact adjustment (intraday branch)
                try:
                    if sentiment and global_data:
                        from app.services.sector_service import sector_service
                        sector_adj = sector_service.get_sector_adjusted_scores(
                            symbol, sentiment["score"], global_data["score"], global_data
                        )
                        sentiment = {**sentiment, "score": sector_adj["sentiment_score"]}
                        global_data = {**global_data, "score": sector_adj["global_score"]}
                except Exception:
                    pass
        else:
            df = data_fetcher.get_historical_data(symbol, period="2y")
            if df is None or df.empty:
                raise ValueError(f"No historical data available for {symbol}")

            # Run models + sentiment/global/fundamentals fetches in parallel
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {}
                if "xgboost" in models:
                    futures["xgboost"] = executor.submit(self.xgboost.predict, df, symbol, horizon)

                from app.services.sentiment_service import sentiment_service
                from app.services.global_market_service import global_market_service
                futures["_sentiment"] = executor.submit(sentiment_service.get_sentiment, symbol)
                futures["_global"] = executor.submit(global_market_service.get_global_signal)
                futures["_fundamentals"] = executor.submit(self._get_fundamental_bias, symbol)

                for name, future in futures.items():
                    if name.startswith("_"):
                        continue
                    try:
                        model_result = future.result(timeout=60)
                        result[name] = model_result
                        model_results[name] = model_result
                    except Exception as e:
                        result[f"{name}_error"] = str(e)

                sentiment = None
                global_data = None
                try:
                    sentiment = futures["_sentiment"].result(timeout=15)
                except Exception:
                    pass
                try:
                    global_data = futures["_global"].result(timeout=15)
                except Exception:
                    pass

                # Sector-aware news impact adjustment (daily branch)
                try:
                    if sentiment and global_data:
                        from app.services.sector_service import sector_service
                        sector_adj = sector_service.get_sector_adjusted_scores(
                            symbol, sentiment["score"], global_data["score"], global_data
                        )
                        sentiment = {**sentiment, "score": sector_adj["sentiment_score"]}
                        global_data = {**global_data, "score": sector_adj["global_score"]}
                except Exception:
                    pass

        # Compute ensemble
        if not is_intraday:
            daily_df = df

        # Detect market regime (HMM or heuristic)
        regime = None
        if daily_df is not None and not daily_df.empty and len(daily_df) >= 60:
            try:
                regime = self._detect_regime(daily_df)
                result["regime"] = {
                    "label": regime.get("regime", "unknown"),
                    "confidence": regime.get("confidence", 0),
                    "adx": regime.get("adx", 0),
                    "vol_ratio": regime.get("vol_ratio", 0),
                }
                logger.info(f"Market regime for {symbol}: {regime.get('regime')}")
            except Exception:
                pass

        # Get fundamental bias (already fetched in parallel for daily branch)
        fundamental = None
        try:
            if "_fundamentals" in futures:
                fundamental = futures["_fundamentals"].result(timeout=10)
            elif not is_intraday:
                fundamental = self._get_fundamental_bias(symbol)
        except Exception:
            pass

        if model_results:
            single = next(iter(model_results.values()))
            result["ensemble"] = {
                "predictions": list(single["predictions"]),
                "dates": list(single["dates"]),
            }

        # Apply market context adjustment to ensemble predictions
        adj_info = {}
        if result.get("ensemble") and result["ensemble"].get("predictions"):
            current_price = 0
            if daily_df is not None and not daily_df.empty:
                current_price = float(daily_df["close"].iloc[-1])
            elif is_intraday and intraday_df is not None and not intraday_df.empty:
                current_price = float(intraday_df["close"].iloc[-1])

            adjusted_preds, adj_info = self._apply_market_context_adjustment(
                result["ensemble"]["predictions"], current_price, sentiment, global_data,
                horizon=horizon
            )
            if adj_info.get("adjustment_pct", 0) != 0:
                result["ensemble"]["predictions_raw"] = result["ensemble"]["predictions"]
                result["ensemble"]["predictions"] = adjusted_preds
                result["ensemble"]["market_adjustment"] = adj_info

        # Apply fundamental bias adjustment to ensemble predictions
        fund_adj_info = {}
        if result.get("ensemble") and result["ensemble"].get("predictions") and fundamental:
            fund_score = fundamental.get("score", 0)
            if abs(fund_score) > 0.1:  # Only adjust if meaningful signal
                cfg = PREDICTION_HORIZONS.get(horizon, {})
                is_intra = cfg.get("intraday", False)
                horizon_days = cfg.get("days", 1) if not is_intra else 0

                # Fundamental bias matters more for longer horizons
                if is_intra:
                    fund_weight = 0.0  # No fundamental adjustment for intraday
                elif horizon_days <= 1:
                    fund_weight = 0.01  # Minimal for 1-day
                elif horizon_days <= 7:
                    fund_weight = 0.02
                elif horizon_days <= 30:
                    fund_weight = 0.04
                else:
                    fund_weight = 0.06  # Strongest for 3mo+

                if fund_weight > 0:
                    # Nudge predictions by fundamental score * weight
                    fund_adj_pct = fund_score * fund_weight
                    adjusted = [round(p * (1 + fund_adj_pct), 2)
                                for p in result["ensemble"]["predictions"]]
                    result["ensemble"]["predictions"] = adjusted
                    fund_adj_info = {
                        "fund_score": round(fund_score, 3),
                        "fund_adjustment_pct": round(fund_adj_pct * 100, 3),
                        "classification": fundamental.get("classification", "balanced"),
                    }
                    result["ensemble"]["fundamental_adjustment"] = fund_adj_info

        # Final sanity cap — max allowed change per horizon
        # Prevents any model from producing unrealistic predictions
        if result.get("ensemble") and result["ensemble"].get("predictions") and current_price > 0:
            max_change_pct = {
                "15m": 0.5, "1h": 1.5, "1d": 3.0, "1w": 6.0,
                "1mo": 12.0, "3mo": 25.0, "6mo": 40.0, "1y": 60.0,
            }.get(horizon, 5.0)
            capped = []
            for pred in result["ensemble"]["predictions"]:
                change_pct = abs((pred - current_price) / current_price) * 100
                if change_pct > max_change_pct:
                    direction = 1 if pred > current_price else -1
                    pred = round(current_price * (1 + direction * max_change_pct / 100), 2)
                capped.append(pred)
            result["ensemble"]["predictions"] = capped

        # Build combined sentiment: stock-specific + global headlines merged
        stock_headlines = sentiment.get("headlines", []) if sentiment else []
        global_headlines = global_data.get("headlines", []) if global_data else []

        # Merge: stock-specific headlines first, then global headlines
        combined_headlines = list(stock_headlines)
        seen_titles = {h.get("title", "").lower() for h in combined_headlines}
        for gh in global_headlines:
            title = gh.get("title", "")
            if title.lower() not in seen_titles:
                combined_headlines.append({
                    "title": title,
                    "sentiment": gh.get("sentiment", "neutral"),
                    "score": gh.get("score", 0),
                    "link": gh.get("link", ""),
                    "source": "global",
                    "big_event": gh.get("big_event", False),
                })
                seen_titles.add(title.lower())

        # Recount from merged headlines
        pos_count = sum(1 for h in combined_headlines if h.get("sentiment") == "positive")
        neg_count = sum(1 for h in combined_headlines if h.get("sentiment") == "negative")
        neu_count = sum(1 for h in combined_headlines if h.get("sentiment") == "neutral")

        # Blended sentiment score: stock-specific + global
        stock_score = sentiment.get("score", 0) if sentiment else 0
        global_score_val = global_data.get("news_score", global_data.get("score", 0)) if global_data else 0
        news_mag = global_data.get("news_magnitude", 0) if global_data else 0

        # Weight: normally stock-specific leads, but global dominates during big events
        if news_mag >= 60:
            blended_score = stock_score * 0.3 + global_score_val * 0.7
        elif news_mag >= 30:
            blended_score = stock_score * 0.5 + global_score_val * 0.5
        else:
            blended_score = stock_score * 0.7 + global_score_val * 0.3
        blended_score = max(-100, min(100, round(blended_score, 2)))

        merged_sentiment = {
            "score": blended_score,
            "stock_score": stock_score,
            "global_score": round(global_score_val, 2),
            "headline_count": len(combined_headlines),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "headlines": combined_headlines[:15],
            "news_magnitude": news_mag,
        }
        result["sentiment"] = merged_sentiment

        if global_data:
            result["global_market"] = {
                "score": global_data.get("score", 0),
                "news_magnitude": news_mag,
                "markets": global_data.get("markets", []),
                "headlines": global_data.get("headlines", []),
            }

        # SHAP drivers
        if not is_intraday and daily_df is not None:
            try:
                shap_drivers = self._get_shap_drivers(symbol, daily_df)
                if shap_drivers:
                    result["shap_drivers"] = shap_drivers
            except Exception:
                pass

        # Contribution breakdown — uses actual ensemble weights, not recomputed MAPEs
        if result.get("ensemble"):
            result["contribution_breakdown"] = self._compute_contribution_breakdown(
                result["ensemble"], adj_info, fund_adj_info
            )

        # Volume analysis
        try:
            vol_df = intraday_df if is_intraday and intraday_df is not None and not intraday_df.empty else daily_df
            vol_analysis = self._compute_volume_analysis(vol_df, result.get("ensemble"))
            if vol_analysis:
                result["volume_analysis"] = vol_analysis
        except Exception as e:
            logger.debug(f"Volume analysis failed: {e}")

        # Fundamental bias (already fetched earlier in parallel)
        if fundamental:
            result["fundamental"] = fundamental

        # Log predictions for accuracy tracking
        try:
            self._log_predictions(symbol, horizon, model_results, result.get("ensemble"), regime)
        except Exception as e:
            logger.debug(f"Prediction logging failed: {e}")

        # Generate explanation — pass MERGED sentiment (stock + global combined)
        # so the explainer shows correct headline counts
        try:
            explanation = prediction_explainer.explain(
                symbol=symbol,
                prediction_result=result,
                df=daily_df,
                sentiment=merged_sentiment,
                global_data=global_data,
                horizon=horizon,
            )
            result["explanation"] = explanation
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")

        return result


    def _log_predictions(self, symbol: str, horizon: str, model_results: dict,
                         ensemble_result: dict | None, regime: dict | None):
        """Log individual model and ensemble predictions to DB for accuracy tracking."""
        from datetime import date, timedelta
        from app.database import SessionLocal
        from app.models import PredictionLog
        from app.config import PREDICTION_HORIZONS

        cfg = PREDICTION_HORIZONS.get(horizon, {})
        is_intraday = cfg.get("intraday", False)
        if is_intraday:
            return  # Skip intraday — hard to backfill actual prices at exact timestamps

        horizon_days = cfg.get("days", 1)
        today = date.today()
        target = today + timedelta(days=horizon_days)
        regime_label = regime.get("regime") if regime else None

        # Determine sector
        sector = None
        try:
            from app.services.sector_service import sector_service
            sector = sector_service.get_sector_for_symbol(symbol)
        except Exception:
            pass

        db = SessionLocal()
        try:
            logs = []
            # Log each individual model
            for model_name, res in model_results.items():
                if res and res.get("predictions"):
                    final_price = res["predictions"][-1]
                    cl = res.get("confidence_lower", [None])
                    cu = res.get("confidence_upper", [None])
                    logs.append(PredictionLog(
                        symbol=symbol, model_type=model_name,
                        prediction_date=today, target_date=target,
                        predicted_price=round(final_price, 2),
                        confidence_lower=round(cl[-1], 2) if cl and cl[-1] else None,
                        confidence_upper=round(cu[-1], 2) if cu and cu[-1] else None,
                        sector=sector, regime=regime_label,
                    ))

            # Log ensemble
            if ensemble_result and ensemble_result.get("predictions"):
                final_price = ensemble_result["predictions"][-1]
                logs.append(PredictionLog(
                    symbol=symbol, model_type="ensemble",
                    prediction_date=today, target_date=target,
                    predicted_price=round(final_price, 2),
                    sector=sector, regime=regime_label,
                ))

            if logs:
                db.add_all(logs)
                db.commit()
                logger.info(f"Logged {len(logs)} predictions for {symbol} (horizon={horizon})")
        except Exception as e:
            db.rollback()
            logger.debug(f"Prediction log commit failed: {e}")
        finally:
            db.close()


prediction_service = PredictionService()
