from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import joblib
import ta
from concurrent.futures import ThreadPoolExecutor
from app.ai.prophet_model import ProphetPredictor
from app.ai.xgboost_model import XGBoostPredictor
from app.ai.explainer import prediction_explainer
from app.services.data_fetcher import data_fetcher
from app.config import PREDICTION_HORIZONS, MODEL_DIR
from app.utils.cache import cache
from app.ai.prophet_model import _patch_prophet

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self):
        self.prophet = ProphetPredictor()
        self.xgboost = XGBoostPredictor()

    def _meta_learner_path(self, symbol: str) -> str:
        safe = symbol.replace(" ", "_").replace("^", "")
        return str(MODEL_DIR / f"meta_{safe}.joblib")

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

    def _estimate_prophet_mape(self, prophet_result: dict, df=None, symbol: str = "") -> float:
        """Compute Prophet MAPE via walk-forward validation (5-fold anchored).
        Falls back to CI-based estimate for small datasets.
        Results are cached per symbol for 1 hour."""
        if symbol:
            cache_key = f"prophet_mape:{symbol}"
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        if df is not None and len(df) >= 200:
            try:
                from prophet import Prophet
                import logging
                logging.getLogger("prophet").setLevel(logging.WARNING)
                logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

                from app.ai.preprocessing import StockDataPreprocessor
                preprocessor = StockDataPreprocessor()
                prophet_df = preprocessor.prepare_prophet_data(df)

                n = len(prophet_df)
                n_splits = 5
                fold_size = n // (n_splits + 1)
                mapes = []

                for i in range(n_splits):
                    train_end = fold_size * (i + 1)
                    test_end = min(train_end + fold_size, n)
                    if test_end <= train_end or train_end < 60:
                        continue

                    train_df = prophet_df.iloc[:train_end]
                    test_df = prophet_df.iloc[train_end:test_end]

                    if len(test_df) < 3:
                        continue

                    model = Prophet(
                        daily_seasonality=True,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        seasonality_mode='multiplicative',
                        changepoint_prior_scale=0.05,
                        n_changepoints=30,
                    )
                    _patch_prophet(model)
                    model.fit(train_df)
                    _patch_prophet(model)

                    future = model.make_future_dataframe(periods=len(test_df))
                    forecast = model.predict(future)

                    test_forecast = forecast[forecast["ds"].isin(test_df["ds"])]
                    if len(test_forecast) >= 3:
                        actual = test_df.set_index("ds")["y"]
                        predicted = test_forecast.set_index("ds")["yhat"]
                        common = actual.index.intersection(predicted.index)
                        if len(common) >= 3:
                            a = actual.loc[common].values
                            p = predicted.loc[common].values
                            nonzero = a != 0
                            if nonzero.any():
                                fold_mape = float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100)
                                mapes.append(fold_mape)

                if mapes:
                    result = max(float(np.mean(mapes)), 0.01)
                    if symbol:
                        cache.set(f"prophet_mape:{symbol}", result, 3600)
                    return result
            except Exception:
                pass

        # Single 80/20 split fallback for medium datasets
        if df is not None and len(df) >= 100:
            try:
                from prophet import Prophet
                import logging
                logging.getLogger("prophet").setLevel(logging.WARNING)
                logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

                from app.ai.preprocessing import StockDataPreprocessor
                preprocessor = StockDataPreprocessor()
                prophet_df = preprocessor.prepare_prophet_data(df)

                split = int(len(prophet_df) * 0.8)
                train_df = prophet_df.iloc[:split]
                test_df = prophet_df.iloc[split:]

                if len(train_df) >= 60 and len(test_df) >= 5:
                    model = Prophet(
                        daily_seasonality=True,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        seasonality_mode='multiplicative',
                        changepoint_prior_scale=0.05,
                        n_changepoints=30,
                    )
                    _patch_prophet(model)
                    model.fit(train_df)
                    _patch_prophet(model)

                    future = model.make_future_dataframe(periods=len(test_df))
                    forecast = model.predict(future)

                    test_forecast = forecast[forecast["ds"].isin(test_df["ds"])]
                    if len(test_forecast) >= 3:
                        actual = test_df.set_index("ds")["y"]
                        predicted = test_forecast.set_index("ds")["yhat"]
                        common = actual.index.intersection(predicted.index)
                        if len(common) >= 3:
                            a = actual.loc[common].values
                            p = predicted.loc[common].values
                            nonzero = a != 0
                            if nonzero.any():
                                result = max(float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100), 0.01)
                                if symbol:
                                    cache.set(f"prophet_mape:{symbol}", result, 3600)
                                return result
            except Exception:
                pass

        # Fallback: CI-based estimate for small datasets
        if prophet_result.get("confidence_upper") and prophet_result.get("confidence_lower"):
            avg_range = sum(
                abs(u - l) for u, l in
                zip(prophet_result["confidence_upper"], prophet_result["confidence_lower"])
            ) / max(len(prophet_result["confidence_upper"]), 1)
            avg_pred = sum(abs(p) for p in prophet_result["predictions"]) / max(len(prophet_result["predictions"]), 1)
            return max((avg_range / (2 * avg_pred)) * 100 if avg_pred > 0 else 10.0, 0.01)
        return 10.0

    def _inverse_mape_ensemble(self, model_results: dict, daily_df=None, regime: dict = None,
                               symbol: str = "", horizon: str = "1d") -> dict:
        """Regime-aware, direction-consensus, horizon-weighted ensemble.

        Improvements over basic inverse-MAPE:
        1. Directional consensus: if 2/3 models agree, boost majority weights
        2. Horizon-aware: LSTM (momentum) weighted higher for short horizons,
           XGBoost/Prophet (mean-reversion) weighted higher for long horizons
        3. Regime adjustments (existing)
        """
        mapes = {}
        for name, result in model_results.items():
            if name == "prophet":
                mapes[name] = self._estimate_prophet_mape(result, df=daily_df, symbol=symbol)
            else:
                mapes[name] = max(result.get("mape", 5.0), 0.01)

        # Inverse MAPE weighting
        weights = {name: 1.0 / mape for name, mape in mapes.items()}

        # --- Regime-based multipliers ---
        if regime:
            if not regime.get("trending") and "prophet" in weights:
                weights["prophet"] *= 1.3
                logger.info(f"Regime: mean-reverting (ADX={regime.get('adx')}), Prophet weight boosted")
            if regime.get("high_vol") and "xgboost" in weights:
                weights["xgboost"] *= 1.2
                logger.info(f"Regime: high-volatility (ratio={regime.get('vol_ratio')}), XGBoost weight boosted")

        # --- Directional consensus ---
        # If 2/3 models agree on direction, boost majority and penalize outlier
        if daily_df is not None and not daily_df.empty and len(model_results) >= 2:
            current_price = float(daily_df["close"].iloc[-1])
            directions = {}
            for name, res in model_results.items():
                if res.get("predictions"):
                    pred_price = res["predictions"][-1]
                    change_pct = ((pred_price - current_price) / current_price) * 100
                    if change_pct > 0.3:
                        directions[name] = "bullish"
                    elif change_pct < -0.3:
                        directions[name] = "bearish"
                    else:
                        directions[name] = "neutral"

            if directions:
                bullish = [n for n, d in directions.items() if d == "bullish"]
                bearish = [n for n, d in directions.items() if d == "bearish"]

                majority = None
                if len(bullish) >= 2:
                    majority = bullish
                    minority = bearish
                elif len(bearish) >= 2:
                    majority = bearish
                    minority = bullish

                if majority:
                    for name in majority:
                        if name in weights:
                            weights[name] *= 1.4
                    for name in (minority or []):
                        if name in weights:
                            weights[name] *= 0.6
                    logger.info(f"Direction consensus: {len(majority)} models agree, boosted {majority}")

        total = sum(weights.values())
        weights = {name: w / total for name, w in weights.items()}

        # Compute ensemble predictions
        min_len = min(len(r["predictions"]) for r in model_results.values())
        ensemble_preds = []
        for i in range(min_len):
            val = sum(
                weights[name] * model_results[name]["predictions"][i]
                for name in model_results
            )
            ensemble_preds.append(round(val, 2))

        # Use dates from first model
        first_result = next(iter(model_results.values()))
        ensemble_dates = first_result["dates"][:min_len]

        result = {
            "predictions": ensemble_preds,
            "dates": ensemble_dates,
        }
        for name in model_results:
            result[f"{name}_weight"] = round(float(weights[name]), 3)

        return result

    def _neural_meta_learner(self, symbol: str, model_results: dict, daily_df=None,
                            regime: dict = None, horizon: str = "1d") -> dict:
        """Neural network meta-learner (2-layer feedforward) to combine model predictions.
        Falls back to Ridge when < 100 data points, then to inverse-MAPE."""
        meta_path = self._meta_learner_path(symbol)

        # Check if we have a trained meta-learner
        if os.path.exists(meta_path):
            try:
                meta_data = joblib.load(meta_path)
                meta_model = meta_data["model"]
                model_order = meta_data["model_order"]
                meta_type = meta_data.get("type", "ridge")

                min_len = min(len(r["predictions"]) for r in model_results.values())
                available = [m for m in model_order if m in model_results]

                if len(available) >= 2:
                    X_base = np.column_stack([
                        model_results[m]["predictions"][:min_len] for m in available
                    ])

                    # Add regime features if neural meta-learner
                    if meta_type == "neural" and regime and regime.get("regime_onehot"):
                        onehot = regime["regime_onehot"]
                        vol_ratio = regime.get("vol_ratio", 1.0)
                        regime_feats = np.array([[
                            onehot.get("bull", 0), onehot.get("bear", 0),
                            onehot.get("sideways", 0), onehot.get("volatile", 0),
                            vol_ratio
                        ]])
                        regime_tiled = np.tile(regime_feats, (min_len, 1))
                        X_meta = np.hstack([X_base, regime_tiled])
                    else:
                        X_meta = X_base

                    ensemble_preds = meta_model.predict(X_meta)
                    if hasattr(ensemble_preds, 'numpy'):
                        ensemble_preds = ensemble_preds.numpy()
                    ensemble_preds = [round(float(p), 2) for p in ensemble_preds.flatten()]

                    first_result = next(iter(model_results.values()))
                    result = {
                        "predictions": ensemble_preds,
                        "dates": first_result["dates"][:min_len],
                        "method": meta_type,
                    }
                    return result
            except Exception as e:
                logger.debug(f"Meta-learner load failed: {e}")

        # Fall back to inverse-MAPE
        result = self._inverse_mape_ensemble(model_results, daily_df=daily_df, regime=regime,
                                             symbol=symbol, horizon=horizon)

        # Try to train and save meta-learner for future use
        try:
            self._train_meta_learner(symbol, model_results, daily_df=daily_df, regime=regime)
        except Exception:
            pass

        return result

    def _train_meta_learner(self, symbol: str, model_results: dict, daily_df=None, regime: dict = None):
        """Train meta-learner: Neural network when enough data, Ridge otherwise."""
        if len(model_results) < 2:
            return

        min_len = min(len(r["predictions"]) for r in model_results.values())
        if min_len < 5:
            return

        model_order = sorted(model_results.keys())
        X = np.column_stack([
            model_results[m]["predictions"][:min_len] for m in model_order
        ])

        # Use the model with lowest MAPE as anchor target
        mapes = {}
        for name, result in model_results.items():
            if name == "prophet":
                mapes[name] = self._estimate_prophet_mape(result, df=daily_df, symbol=symbol)
            else:
                mapes[name] = max(result.get("mape", 5.0), 0.01)

        best_model = min(mapes, key=mapes.get)
        y = np.array(model_results[best_model]["predictions"][:min_len])

        meta_type = "ridge"

        # Try neural meta-learner when enough data
        if min_len >= 100 and regime:
            try:
                import tensorflow as tf

                # Add regime features
                onehot = regime.get("regime_onehot", {})
                vol_ratio = regime.get("vol_ratio", 1.0)
                regime_feats = np.array([[
                    onehot.get("bull", 0), onehot.get("bear", 0),
                    onehot.get("sideways", 0), onehot.get("volatile", 0),
                    vol_ratio
                ]])
                regime_tiled = np.tile(regime_feats, (min_len, 1))
                X_full = np.hstack([X, regime_tiled])

                input_dim = X_full.shape[1]
                nn_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
                    tf.keras.layers.Dense(1)
                ])
                nn_model.compile(optimizer='adam', loss='mse')
                nn_model.fit(X_full, y, epochs=50, batch_size=min(32, min_len), verbose=0)

                meta_path = self._meta_learner_path(symbol)
                joblib.dump({"model": nn_model, "model_order": model_order, "type": "neural"}, meta_path)
                return
            except Exception as e:
                logger.debug(f"Neural meta-learner training failed, using Ridge: {e}")

        # Ridge fallback
        from sklearn.linear_model import Ridge
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X, y)

        meta_path = self._meta_learner_path(symbol)
        joblib.dump({"model": meta_model, "model_order": model_order, "type": "ridge"}, meta_path)

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

    def _compute_contribution_breakdown(self, ensemble_result: dict, adj_info: dict = None) -> dict:
        """Compute contribution breakdown using actual ensemble weights.

        Uses the weights from the ensemble (already reflecting MAPE + regime adjustments)
        rather than recomputing from raw MAPEs which misses Prophet.
        """
        # Extract actual weights from ensemble result
        xgb_w = ensemble_result.get("xgboost_weight", 0)
        prophet_w = ensemble_result.get("prophet_weight", 0)

        total_model_w = xgb_w + prophet_w
        if total_model_w <= 0:
            return {"technical": 50.0, "seasonal": 15.0, "fundamental": 0.0, "sentiment": 35.0}

        # Normalize model weights to sum to 1
        xgb_w /= total_model_w
        prophet_w /= total_model_w

        # Sentiment contribution based on context weight
        context_w = adj_info.get("context_weight", 0) if adj_info else 0
        sentiment_share = min(45.0, round(context_w * 55, 1))
        model_share = 100.0 - sentiment_share

        # XGBoost → Technical, Prophet → Seasonal
        technical = xgb_w * model_share
        seasonal = prophet_w * model_share

        return {
            "technical": round(technical, 1),
            "seasonal": round(seasonal, 1),
            "fundamental": 0.0,
            "sentiment": round(sentiment_share, 1),
        }

    def _select_models_for_horizon(self, horizon: str) -> list:
        """Auto-select best model(s) for a given prediction horizon.

        - Intraday (15m, 1h): XGBoost only — feature-based, stable for short-term
        - 1 Day: XGBoost only — most reliable single model
        - 1 Week: XGBoost + Prophet — technical + seasonality
        - 1 Month+: Prophet only — trend/seasonality dominate, others drift
        """
        cfg = PREDICTION_HORIZONS.get(horizon, {})
        is_intraday = cfg.get("intraday", False)
        days = cfg.get("days", 1) if not is_intraday else 0

        if is_intraday:
            return ["xgboost"]
        elif days <= 1:
            return ["xgboost"]
        elif days <= 5:
            return ["xgboost", "prophet"]
        else:
            return ["prophet"]

    def predict(self, symbol: str, horizon: str = "1d", models: list = None) -> dict:
        if models is None:
            models = self._select_models_for_horizon(horizon)

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
                if "prophet" in models and daily_df is not None and not daily_df.empty:
                    futures["prophet"] = executor.submit(self.prophet.predict, daily_df, horizon, symbol)
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
        else:
            df = data_fetcher.get_historical_data(symbol, period="2y")
            if df is None or df.empty:
                raise ValueError(f"No historical data available for {symbol}")

            # Run models + sentiment/global fetches in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                if "prophet" in models:
                    futures["prophet"] = executor.submit(self.prophet.predict, df, horizon, symbol)
                if "xgboost" in models:
                    futures["xgboost"] = executor.submit(self.xgboost.predict, df, symbol, horizon)

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

        if len(model_results) >= 2:
            result["ensemble"] = self._neural_meta_learner(symbol, model_results, daily_df=daily_df,
                                                            regime=regime, horizon=horizon)
        elif len(model_results) == 1:
            single = next(iter(model_results.values()))
            result["ensemble"] = {
                "predictions": single["predictions"],
                "dates": single["dates"],
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
                result["ensemble"], adj_info
            )

        # Volume analysis
        try:
            vol_df = intraday_df if is_intraday and intraday_df is not None and not intraday_df.empty else daily_df
            vol_analysis = self._compute_volume_analysis(vol_df, result.get("ensemble"))
            if vol_analysis:
                result["volume_analysis"] = vol_analysis
        except Exception as e:
            logger.debug(f"Volume analysis failed: {e}")

        # Fundamental bias
        try:
            fundamental = self._get_fundamental_bias(symbol)
            if fundamental:
                result["fundamental"] = fundamental
        except Exception:
            pass

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


prediction_service = PredictionService()
