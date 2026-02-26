import numpy as np
import pandas as pd
from app.services.data_fetcher import data_fetcher
from app.ai.lstm_model import LSTMPredictor
from app.ai.prophet_model import ProphetPredictor


class BacktestService:
    def __init__(self):
        self.lstm = LSTMPredictor()
        self.prophet = ProphetPredictor()

    def backtest(self, symbol: str, model_type: str = "lstm", test_days: int = 30) -> dict:
        df = data_fetcher.get_historical_data(symbol, period="2y")
        if df is None or df.empty or len(df) < 100:
            raise ValueError(f"Not enough data to backtest {symbol}")

        actuals = []
        predictions = []

        # Use last `test_days` as test set
        train_end = len(df) - test_days
        if train_end < 80:
            raise ValueError("Not enough training data for backtest")

        for i in range(test_days):
            train_df = df.iloc[:train_end + i]
            actual_price = df.iloc[train_end + i]["close"]

            try:
                if model_type == "lstm":
                    result = self.lstm.predict(train_df, symbol, "1d")
                else:
                    result = self.prophet.predict(train_df, "1d")

                pred_price = result["predictions"][0]
                actuals.append(actual_price)
                predictions.append(pred_price)
            except Exception:
                continue

        if len(actuals) == 0:
            raise ValueError("Backtest produced no results")

        actuals = np.array(actuals)
        predictions = np.array(predictions)

        mae = float(np.mean(np.abs(actuals - predictions)))
        mape = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)

        # Directional accuracy
        if len(actuals) > 1:
            actual_direction = np.diff(actuals) > 0
            pred_direction = np.diff(predictions) > 0
            directional_accuracy = float(np.mean(actual_direction == pred_direction) * 100)
        else:
            directional_accuracy = 0

        return {
            "symbol": symbol,
            "model_type": model_type,
            "total_predictions": len(actuals),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "directional_accuracy": round(directional_accuracy, 2),
        }


backtest_service = BacktestService()
