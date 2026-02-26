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

    @staticmethod
    def _detect_regime(df: pd.DataFrame) -> dict:
        """Detect market regime: trending vs mean-reverting, high-vol vs low-vol."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ADX for trend strength
        try:
            adx_val = float(ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1])
        except Exception:
            adx_val = 20.0

        trending = adx_val > 25

        # Volatility regime: 20-day vol / 60-day vol
        returns = close.pct_change()
        vol_20 = float(returns.rolling(20).std().iloc[-1]) if len(returns) >= 20 else 0.01
        vol_60 = float(returns.rolling(60).std().iloc[-1]) if len(returns) >= 60 else vol_20
        vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0
        high_vol = vol_ratio > 1.2

        return {
            "trending": trending,
            "high_vol": high_vol,
            "adx": round(adx_val, 2),
            "vol_ratio": round(vol_ratio, 3),
        }

    def _estimate_prophet_mape(self, prophet_result: dict, df=None) -> float:
        """Compute actual Prophet MAPE via 80/20 train/test split on historical data.
        Falls back to CI-based estimate for small datasets."""
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

                    # Align forecast with test dates
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
                logger.info(f"Regime: trending (ADX={regime['adx']}), LSTM weight boosted")
            if not regime.get("trending") and "prophet" in weights:
                weights["prophet"] *= 1.3
                logger.info(f"Regime: mean-reverting (ADX={regime['adx']}), Prophet weight boosted")
            if regime.get("high_vol") and "xgboost" in weights:
                weights["xgboost"] *= 1.2
                logger.info(f"Regime: high-volatility (ratio={regime['vol_ratio']}), XGBoost weight boosted")

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

    def _stacking_ensemble(self, symbol: str, model_results: dict, daily_df=None, regime: dict = None) -> dict:
        """Stacking meta-learner (Ridge) to combine model predictions.
        Falls back to inverse-MAPE if meta-learner not available."""
        from sklearn.linear_model import Ridge

        meta_path = self._meta_learner_path(symbol)

        # Check if we have a trained meta-learner
        if os.path.exists(meta_path):
            try:
                meta_data = joblib.load(meta_path)
                meta_model = meta_data["model"]
                model_order = meta_data["model_order"]

                # Build feature matrix from current predictions
                min_len = min(len(r["predictions"]) for r in model_results.values())
                available = [m for m in model_order if m in model_results]

                if len(available) >= 2:
                    X_meta = np.column_stack([
                        model_results[m]["predictions"][:min_len] for m in available
                    ])
                    ensemble_preds = meta_model.predict(X_meta).tolist()
                    ensemble_preds = [round(float(p), 2) for p in ensemble_preds]

                    first_result = next(iter(model_results.values()))
                    result = {
                        "predictions": ensemble_preds,
                        "dates": first_result["dates"][:min_len],
                        "method": "stacking",
                    }
                    # Approximate weights from Ridge coefficients
                    coefs = meta_model.coef_
                    total_coef = sum(abs(c) for c in coefs)
                    for i, name in enumerate(available):
                        result[f"{name}_weight"] = round(float(abs(coefs[i]) / total_coef), 3) if total_coef > 0 else round(1.0 / len(available), 3)
                    return result
            except Exception:
                pass

        # Train meta-learner if we have all 3 models' test predictions
        # For now, fall back to inverse-MAPE
        result = self._inverse_mape_ensemble(model_results, daily_df=daily_df, regime=regime)

        # Try to train and save meta-learner for future use
        try:
            self._train_meta_learner(symbol, model_results, daily_df=daily_df)
        except Exception:
            pass

        return result

    def _train_meta_learner(self, symbol: str, model_results: dict, daily_df=None):
        """Train Ridge meta-learner using best model (lowest MAPE) as anchor target."""
        from sklearn.linear_model import Ridge

        if len(model_results) < 2:
            return

        min_len = min(len(r["predictions"]) for r in model_results.values())
        if min_len < 5:
            return

        model_order = sorted(model_results.keys())
        X = np.column_stack([
            model_results[m]["predictions"][:min_len] for m in model_order
        ])

        # Use the model with lowest MAPE as anchor target (not circular weighted avg)
        mapes = {}
        for name, result in model_results.items():
            if name == "prophet":
                mapes[name] = self._estimate_prophet_mape(result, df=daily_df)
            else:
                mapes[name] = max(result.get("mape", 5.0), 0.01)

        best_model = min(mapes, key=mapes.get)
        y = np.array(model_results[best_model]["predictions"][:min_len])

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X, y)

        meta_path = self._meta_learner_path(symbol)
        joblib.dump({"model": meta_model, "model_order": model_order}, meta_path)

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

        # Compute ensemble (pass daily_df for Prophet MAPE calculation)
        # In non-intraday branch, `df` is the daily data; in intraday branch, `daily_df` is already set
        if not is_intraday:
            daily_df = df
        # else: daily_df is already assigned in the intraday branch above

        # Detect market regime for ensemble weighting
        regime = None
        if daily_df is not None and not daily_df.empty and len(daily_df) >= 60:
            try:
                regime = self._detect_regime(daily_df)
                logger.info(f"Market regime for {symbol}: {regime}")
            except Exception:
                pass

        if len(model_results) >= 2:
            result["ensemble"] = self._stacking_ensemble(symbol, model_results, daily_df=daily_df, regime=regime)
        elif len(model_results) == 1:
            single = next(iter(model_results.values()))
            result["ensemble"] = {
                "predictions": single["predictions"],
                "dates": single["dates"],
            }

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
