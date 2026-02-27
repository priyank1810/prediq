from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import joblib
import ta
from app.ai.lstm_model import LSTMPredictor
from app.ai.prophet_model import ProphetPredictor
from app.ai.xgboost_model import XGBoostPredictor
from app.ai.explainer import prediction_explainer
from app.services.data_fetcher import data_fetcher
from app.config import PREDICTION_HORIZONS, MODEL_DIR

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self):
        self.lstm = LSTMPredictor()
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

    def _estimate_prophet_mape(self, prophet_result: dict, df=None) -> float:
        """Compute Prophet MAPE via walk-forward validation (5-fold anchored).
        Falls back to CI-based estimate for small datasets."""
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
                    model.fit(train_df)

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
                    return max(float(np.mean(mapes)), 0.01)
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
                    model.fit(train_df)

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
                                mape = float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100)
                                return max(mape, 0.01)
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

    def _inverse_mape_ensemble(self, model_results: dict, daily_df=None, regime: dict = None) -> dict:
        """Regime-aware inverse-MAPE weighting across all available models."""
        mapes = {}
        for name, result in model_results.items():
            if name == "prophet":
                mapes[name] = self._estimate_prophet_mape(result, df=daily_df)
            else:
                mapes[name] = max(result.get("mape", 5.0), 0.01)

        # Inverse MAPE weighting
        weights = {name: 1.0 / mape for name, mape in mapes.items()}

        # Apply regime-based multipliers
        if regime:
            if regime.get("trending") and "lstm" in weights:
                weights["lstm"] *= 1.3
                logger.info(f"Regime: trending (ADX={regime.get('adx')}), LSTM weight boosted")
            if not regime.get("trending") and "prophet" in weights:
                weights["prophet"] *= 1.3
                logger.info(f"Regime: mean-reverting (ADX={regime.get('adx')}), Prophet weight boosted")
            if regime.get("high_vol") and "xgboost" in weights:
                weights["xgboost"] *= 1.2
                logger.info(f"Regime: high-volatility (ratio={regime.get('vol_ratio')}), XGBoost weight boosted")

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

    def _neural_meta_learner(self, symbol: str, model_results: dict, daily_df=None, regime: dict = None) -> dict:
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
        result = self._inverse_mape_ensemble(model_results, daily_df=daily_df, regime=regime)

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
                mapes[name] = self._estimate_prophet_mape(result, df=daily_df)
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

    def _compute_contribution_breakdown(self, model_results: dict, regime: dict = None) -> dict:
        """Compute contribution breakdown: technical, seasonal, fundamental, sentiment."""
        weights = {}
        for name in model_results:
            weights[name] = 1.0 / max(model_results[name].get("mape", 5.0), 0.01)

        total = sum(weights.values())
        if total <= 0:
            return {"technical": 33.0, "seasonal": 33.0, "fundamental": 0.0, "sentiment": 34.0}

        weights = {k: v / total for k, v in weights.items()}

        # Map models to contribution categories:
        # XGBoost → technical (it's feature-based)
        # Prophet → seasonal (it's time-series with seasonality)
        # LSTM → mix of technical and sentiment (uses context features)
        technical = weights.get("xgboost", 0) * 100 + weights.get("lstm", 0) * 50
        seasonal = weights.get("prophet", 0) * 100
        sentiment = weights.get("lstm", 0) * 30  # LSTM uses sentiment features
        fundamental = weights.get("lstm", 0) * 20  # Small fundamental component

        total_pct = technical + seasonal + sentiment + fundamental
        if total_pct > 0:
            factor = 100.0 / total_pct
            technical *= factor
            seasonal *= factor
            sentiment *= factor
            fundamental *= factor

        return {
            "technical": round(technical, 1),
            "seasonal": round(seasonal, 1),
            "fundamental": round(fundamental, 1),
            "sentiment": round(sentiment, 1),
        }

    def predict(self, symbol: str, horizon: str = "1d", models: list = None) -> dict:
        if models is None:
            models = ["lstm", "prophet", "xgboost"]

        cfg = PREDICTION_HORIZONS.get(horizon, {})
        is_intraday = cfg.get("intraday", False)

        result = {"symbol": symbol, "horizon": horizon, "horizon_label": cfg.get("label", horizon)}

        model_results = {}

        if is_intraday:
            intraday_df = data_fetcher.get_intraday_data(symbol, period="5d", interval="15m")
            daily_df = data_fetcher.get_historical_data(symbol, period="2y")

            if "lstm" in models:
                try:
                    lstm_result = self.lstm.predict_intraday(intraday_df, symbol, horizon)
                    result["lstm"] = lstm_result
                    model_results["lstm"] = lstm_result
                except Exception as e:
                    result["lstm_error"] = str(e)

            if "prophet" in models and daily_df is not None and not daily_df.empty:
                try:
                    prophet_result = self.prophet.predict(daily_df, horizon, symbol=symbol)
                    result["prophet"] = prophet_result
                    model_results["prophet"] = prophet_result
                except Exception as e:
                    result["prophet_error"] = str(e)

            if "xgboost" in models and intraday_df is not None and not intraday_df.empty:
                try:
                    xgb_result = self.xgboost.predict_intraday(intraday_df, symbol, horizon)
                    result["xgboost"] = xgb_result
                    model_results["xgboost"] = xgb_result
                except Exception as e:
                    result["xgboost_error"] = str(e)
        else:
            df = data_fetcher.get_historical_data(symbol, period="2y")
            if df is None or df.empty:
                raise ValueError(f"No historical data available for {symbol}")

            if "lstm" in models:
                try:
                    lstm_result = self.lstm.predict(df, symbol, horizon)
                    result["lstm"] = lstm_result
                    model_results["lstm"] = lstm_result
                except Exception as e:
                    result["lstm_error"] = str(e)

            if "prophet" in models:
                try:
                    prophet_result = self.prophet.predict(df, horizon, symbol=symbol)
                    result["prophet"] = prophet_result
                    model_results["prophet"] = prophet_result
                except Exception as e:
                    result["prophet_error"] = str(e)

            if "xgboost" in models:
                try:
                    xgb_result = self.xgboost.predict(df, symbol, horizon)
                    result["xgboost"] = xgb_result
                    model_results["xgboost"] = xgb_result
                except Exception as e:
                    result["xgboost_error"] = str(e)

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
            result["ensemble"] = self._neural_meta_learner(symbol, model_results, daily_df=daily_df, regime=regime)
        elif len(model_results) == 1:
            single = next(iter(model_results.values()))
            result["ensemble"] = {
                "predictions": single["predictions"],
                "dates": single["dates"],
            }

        # SHAP drivers (Step 4)
        if not is_intraday and daily_df is not None:
            try:
                shap_drivers = self._get_shap_drivers(symbol, daily_df)
                if shap_drivers:
                    result["shap_drivers"] = shap_drivers
            except Exception:
                pass

        # Contribution breakdown (Step 15)
        if model_results:
            result["contribution_breakdown"] = self._compute_contribution_breakdown(model_results, regime)

        # Fundamental bias (Step 8)
        try:
            fundamental = self._get_fundamental_bias(symbol)
            if fundamental:
                result["fundamental"] = fundamental
        except Exception:
            pass

        # Generate explanation
        try:
            sentiment = None
            global_data = None
            try:
                from app.services.sentiment_service import sentiment_service
                sentiment = sentiment_service.get_sentiment(symbol)
            except Exception:
                pass
            try:
                from app.services.global_market_service import global_market_service
                global_data = global_market_service.get_global_signal()
            except Exception:
                pass

            explanation = prediction_explainer.explain(
                symbol=symbol,
                prediction_result=result,
                df=daily_df,
                sentiment=sentiment,
                global_data=global_data,
            )
            result["explanation"] = explanation
        except Exception as e:
            logger.warning(f"Failed to generate explanation: {e}")

        return result


prediction_service = PredictionService()
