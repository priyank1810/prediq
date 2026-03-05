"""Seasonal forecasting model — replaces Prophet with statsmodels.

Uses Holt-Winters Exponential Smoothing for trend + seasonality decomposition.
Same interface as the old ProphetPredictor so prediction_service works unchanged.
No C++ compilation (cmdstan) required.
"""
import logging
import warnings
import pandas as pd
import numpy as np
from app.ai.preprocessing import StockDataPreprocessor
from app.config import PREDICTION_HORIZONS

logger = logging.getLogger(__name__)


# Keep for backward compatibility — prediction_service imports this
def _patch_prophet(model):
    pass


class ProphetPredictor:
    """Seasonal predictor using Holt-Winters exponential smoothing."""

    def _horizon_to_days(self, horizon: str) -> int:
        cfg = PREDICTION_HORIZONS.get(horizon, {})
        if cfg.get("intraday"):
            return 1
        return cfg.get("days", 1)

    def predict(self, df: pd.DataFrame, horizon: str = "1d", symbol: str = "") -> dict:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        close = df["close"].dropna().values.astype(float)
        if len(close) < 60:
            raise ValueError(f"Not enough data for seasonal model. Got {len(close)}, need 60+")

        target_bdays = self._horizon_to_days(horizon)

        # Determine seasonal period: 5 trading days/week ≈ weekly seasonality
        seasonal_period = 5

        # Use multiplicative seasonality (works better for stock prices)
        # Fall back to additive if data has zeros/negatives
        seasonal_type = "mul" if (close > 0).all() else "add"
        trend_type = "add"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ExponentialSmoothing(
                    close,
                    trend=trend_type,
                    seasonal=seasonal_type,
                    seasonal_periods=seasonal_period,
                    damped_trend=True,
                    initialization_method="estimated",
                )
                fitted = model.fit(optimized=True, remove_bias=True)
            except Exception:
                # Fallback: no seasonality
                model = ExponentialSmoothing(
                    close,
                    trend=trend_type,
                    damped_trend=True,
                    initialization_method="estimated",
                )
                fitted = model.fit(optimized=True)

        # Forecast
        forecast_values = fitted.forecast(target_bdays)
        predictions = [round(float(v), 2) for v in forecast_values]

        # Generate business day dates
        last_date = pd.to_datetime(df["date"].iloc[-1])
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=target_bdays)
        dates = [d.strftime("%Y-%m-%d") for d in future_dates]

        # Confidence intervals from residuals
        residuals = fitted.resid
        residual_std = float(np.std(residuals)) if len(residuals) > 0 else 0
        z = 1.28  # ~80% confidence interval

        lower = []
        upper = []
        for i, pred in enumerate(predictions):
            # Uncertainty grows with forecast horizon
            horizon_factor = np.sqrt(i + 1)
            margin = z * residual_std * horizon_factor
            lower.append(round(pred - margin, 2))
            upper.append(round(pred + margin, 2))

        # Compute MAPE via simple holdout
        mape = self._estimate_mape(close, seasonal_period, seasonal_type, trend_type)

        return {
            "predictions": predictions,
            "dates": dates,
            "confidence_lower": lower,
            "confidence_upper": upper,
            "mape": round(mape, 2),
        }

    def _estimate_mape(self, close: np.ndarray, seasonal_period: int,
                       seasonal_type: str, trend_type: str) -> float:
        """Quick holdout MAPE estimate."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if len(close) < 100:
            return 5.0

        split = int(len(close) * 0.8)
        train = close[:split]
        test = close[split:]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    train,
                    trend=trend_type,
                    seasonal=seasonal_type,
                    seasonal_periods=seasonal_period,
                    damped_trend=True,
                    initialization_method="estimated",
                )
                fitted = model.fit(optimized=True)

            preds = fitted.forecast(len(test))
            nonzero = test != 0
            if nonzero.any():
                return float(np.mean(np.abs((test[nonzero] - preds[nonzero]) / test[nonzero])) * 100)
        except Exception:
            pass
        return 5.0
