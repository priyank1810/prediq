import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from app.ai.preprocessing import StockDataPreprocessor
from app.config import PREDICTION_HORIZONS, MODEL_DIR, MODEL_FRESHNESS_HOURS

logger = logging.getLogger(__name__)


class ProphetPredictor:
    def _horizon_to_days(self, horizon: str) -> int:
        cfg = PREDICTION_HORIZONS.get(horizon, {})
        if cfg.get("intraday"):
            return 1  # Prophet works on daily data; for intraday we predict next day
        return cfg.get("days", 1)

    def _model_path(self, symbol: str) -> str:
        safe = symbol.replace(" ", "_").replace("^", "")
        return str(MODEL_DIR / f"prophet_{safe}.joblib")

    def _model_is_fresh(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        return age_hours < MODEL_FRESHNESS_HOURS

    def predict(self, df: pd.DataFrame, horizon: str = "1d", symbol: str = "") -> dict:
        from prophet import Prophet
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        preprocessor = StockDataPreprocessor()
        prophet_df = preprocessor.prepare_prophet_data(df)

        # Try adding regressors for better accuracy
        has_regressors = False
        try:
            import ta
            close = df["close"]
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
            volume = df["volume"].astype(float)
            if (volume == 0).all():
                volume = pd.Series(1.0, index=volume.index)
            vol_ma = volume.rolling(20).mean().replace(0, np.nan)
            volume_norm = volume / vol_ma

            prophet_df["rsi"] = rsi.values
            prophet_df["volume_norm"] = volume_norm.values
            prophet_df = prophet_df.dropna().reset_index(drop=True)

            if len(prophet_df) >= 60:
                has_regressors = True
        except Exception as e:
            logger.debug(f"Prophet regressor computation failed, using basic model: {e}")
            # Re-prepare without regressors
            prophet_df = preprocessor.prepare_prophet_data(df)

        # Try to load cached model
        model_path = self._model_path(symbol) if symbol else ""
        cached_model = None
        if model_path and self._model_is_fresh(model_path):
            try:
                cached_data = joblib.load(model_path)
                cached_model = cached_data["model"]
                cached_regressors = cached_data.get("has_regressors", False)
                if cached_regressors == has_regressors:
                    logger.info(f"Loaded cached Prophet model for {symbol}")
                else:
                    cached_model = None  # Regressor mismatch, retrain
            except Exception:
                cached_model = None

        if cached_model is not None:
            model = cached_model
        else:
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                n_changepoints=30,
            )

            if has_regressors:
                model.add_regressor("rsi")
                model.add_regressor("volume_norm")

            model.fit(prophet_df)

            # Cache the fitted model
            if model_path:
                try:
                    joblib.dump({"model": model, "has_regressors": has_regressors}, model_path)
                    logger.info(f"Cached Prophet model for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to cache Prophet model: {e}")

        target_bdays = self._horizon_to_days(horizon)
        calendar_days = max(1, int(target_bdays * 1.5) + 5)
        future = model.make_future_dataframe(periods=calendar_days)

        if has_regressors:
            last_rsi = float(prophet_df["rsi"].iloc[-1])
            last_vol = float(prophet_df["volume_norm"].iloc[-1])
            hist_len = len(prophet_df)

            # Apply mean-reversion for future dates
            # RSI reverts toward 50 (half-life 10 days): decay = 0.5^(1/10) ≈ 0.933
            # volume_norm reverts toward 1.0 (half-life 5 days): decay = 0.5^(1/5) ≈ 0.871
            rsi_decay = 0.5 ** (1.0 / 10.0)    # ~0.933
            vol_decay = 0.5 ** (1.0 / 5.0)     # ~0.871

            rsi_values = np.empty(len(future))
            vol_values = np.empty(len(future))

            # Historical rows get actual values
            rsi_values[:hist_len] = prophet_df["rsi"].values
            vol_values[:hist_len] = prophet_df["volume_norm"].values

            # Future rows get mean-reverting values
            for i in range(hist_len, len(future)):
                days_ahead = i - hist_len + 1
                rsi_values[i] = 50.0 + (last_rsi - 50.0) * (rsi_decay ** days_ahead)
                vol_values[i] = 1.0 + (last_vol - 1.0) * (vol_decay ** days_ahead)

            future["rsi"] = rsi_values
            future["volume_norm"] = vol_values

        forecast = model.predict(future)

        # Get only future predictions (after training data)
        last_train_date = prophet_df["ds"].max()
        future_preds = forecast[forecast["ds"] > last_train_date]

        # Filter to business days only
        future_preds = future_preds[future_preds["ds"].dt.dayofweek < 5]
        future_preds = future_preds.head(target_bdays)

        predictions = future_preds["yhat"].tolist()
        lower = future_preds["yhat_lower"].tolist()
        upper = future_preds["yhat_upper"].tolist()
        dates = future_preds["ds"].dt.strftime("%Y-%m-%d").tolist()

        return {
            "predictions": [round(float(p), 2) for p in predictions],
            "dates": dates,
            "confidence_lower": [round(float(v), 2) for v in lower],
            "confidence_upper": [round(float(v), 2) for v in upper],
        }
