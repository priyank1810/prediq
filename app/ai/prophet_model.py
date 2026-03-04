import logging
import pandas as pd
import numpy as np
from app.ai.preprocessing import StockDataPreprocessor
from app.config import PREDICTION_HORIZONS

logger = logging.getLogger(__name__)


def _patch_prophet(model):
    """Ensure stan_backend attribute exists on a Prophet instance."""
    if not hasattr(model, 'stan_backend'):
        model.stan_backend = None
    if not hasattr(model, 'uncertainty_samples'):
        model.uncertainty_samples = 200


class ProphetPredictor:
    def _horizon_to_days(self, horizon: str) -> int:
        cfg = PREDICTION_HORIZONS.get(horizon, {})
        if cfg.get("intraday"):
            return 1
        return cfg.get("days", 1)

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
            prophet_df = preprocessor.prepare_prophet_data(df)

        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            n_changepoints=30,
        )
        _patch_prophet(model)

        if has_regressors:
            model.add_regressor("rsi")
            model.add_regressor("volume_norm")

        model.fit(prophet_df)
        _patch_prophet(model)

        target_bdays = self._horizon_to_days(horizon)
        calendar_days = max(1, int(target_bdays * 1.5) + 5)
        future = model.make_future_dataframe(periods=calendar_days)

        if has_regressors:
            last_rsi = float(prophet_df["rsi"].iloc[-1])
            last_vol = float(prophet_df["volume_norm"].iloc[-1])
            hist_len = len(prophet_df)

            rsi_decay = 0.5 ** (1.0 / 10.0)
            vol_decay = 0.5 ** (1.0 / 5.0)

            rsi_values = np.empty(len(future))
            vol_values = np.empty(len(future))
            rsi_values[:hist_len] = prophet_df["rsi"].values
            vol_values[:hist_len] = prophet_df["volume_norm"].values

            for i in range(hist_len, len(future)):
                days_ahead = i - hist_len + 1
                rsi_values[i] = 50.0 + (last_rsi - 50.0) * (rsi_decay ** days_ahead)
                vol_values[i] = 1.0 + (last_vol - 1.0) * (vol_decay ** days_ahead)

            future["rsi"] = rsi_values
            future["volume_norm"] = vol_values

        forecast = model.predict(future)

        last_train_date = prophet_df["ds"].max()
        future_preds = forecast[forecast["ds"] > last_train_date]
        future_preds = future_preds[future_preds["ds"].dt.dayofweek < 5]
        future_preds = future_preds.head(target_bdays)

        predictions = future_preds["yhat"].tolist()
        dates = future_preds["ds"].dt.strftime("%Y-%m-%d").tolist()

        lower = future_preds["yhat_lower"].tolist() if "yhat_lower" in future_preds.columns else []
        upper = future_preds["yhat_upper"].tolist() if "yhat_upper" in future_preds.columns else []

        if not lower or not upper or any(np.isnan(v) for v in lower + upper):
            lower = [round(float(p) * 0.97, 2) for p in predictions]
            upper = [round(float(p) * 1.03, 2) for p in predictions]

        return {
            "predictions": [round(float(p), 2) for p in predictions],
            "dates": dates,
            "confidence_lower": [round(float(v), 2) for v in lower],
            "confidence_upper": [round(float(v), 2) for v in upper],
        }
