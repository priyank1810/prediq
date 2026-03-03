import logging
import numpy as np
import pandas as pd
import ta
from app.services.data_fetcher import data_fetcher
from app.ai.lstm_model import LSTMPredictor
from app.ai.prophet_model import ProphetPredictor

logger = logging.getLogger(__name__)


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

    def _score_new_system(self, window: pd.DataFrame) -> float:
        """6-indicator technical score (RSI+Vol+BB+VWAP+MA+Candle)."""
        from app.services.indicator_service import indicator_service
        if len(window) < 20:
            return 0.0
        # Add datetime_str for compatibility
        w = window.copy()
        if "datetime_str" not in w.columns:
            if "date" in w.columns:
                w["datetime_str"] = w["date"].astype(str)
            else:
                w["datetime_str"] = [str(i) for i in range(len(w))]
        result = indicator_service.compute_intraday_indicators(w)
        return result["score"]

    def _score_old_system(self, window: pd.DataFrame) -> float:
        """3-indicator score: RSI 40% + BB 30% + MA 30%."""
        if len(window) < 20:
            return 0.0
        close = window["close"]
        current_price = float(close.iloc[-1])

        # RSI
        rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        if rsi <= 30:
            rsi_score = 60 + (30 - rsi) * 4
        elif rsi <= 40:
            rsi_score = (40 - rsi) * 6
        elif rsi <= 60:
            rsi_score = 0
        elif rsi <= 70:
            rsi_score = -(rsi - 60) * 6
        else:
            rsi_score = -60 - (rsi - 70) * 4
        rsi_score = max(-100, min(100, rsi_score))

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = float(bb.bollinger_hband().iloc[-1]) if not pd.isna(bb.bollinger_hband().iloc[-1]) else current_price * 1.02
        bb_lower = float(bb.bollinger_lband().iloc[-1]) if not pd.isna(bb.bollinger_lband().iloc[-1]) else current_price * 0.98
        bb_range = bb_upper - bb_lower if bb_upper != bb_lower else 1
        bb_pos = (current_price - bb_lower) / bb_range
        if bb_pos <= 0.2:
            bb_score = 60 + (0.2 - bb_pos) * 200
        elif bb_pos <= 0.4:
            bb_score = (0.4 - bb_pos) * 300
        elif bb_pos <= 0.6:
            bb_score = 0
        elif bb_pos <= 0.8:
            bb_score = -(bb_pos - 0.6) * 300
        else:
            bb_score = -60 - (bb_pos - 0.8) * 200
        bb_score = max(-100, min(100, bb_score))

        # MA Crossover (5 vs 9)
        ma5 = ta.trend.SMAIndicator(close, window=5).sma_indicator()
        ma9 = ta.trend.SMAIndicator(close, window=9).sma_indicator()
        ma5_now = float(ma5.iloc[-1]) if not pd.isna(ma5.iloc[-1]) else current_price
        ma9_now = float(ma9.iloc[-1]) if not pd.isna(ma9.iloc[-1]) else current_price
        ma_spread_pct = ((ma5_now - ma9_now) / current_price * 100) if current_price > 0 else 0
        ma_score = max(-100, min(100, ma_spread_pct * 300))

        return max(-100, min(100, 0.40 * rsi_score + 0.30 * bb_score + 0.30 * ma_score))

    def backtest_signal(self, symbol: str, test_days: int = 60) -> dict:
        """Walk-forward backtest comparing new vs old signal system."""
        df = data_fetcher.get_historical_data(symbol, period="2y")
        if df is None or df.empty or len(df) < 100:
            raise ValueError(f"Not enough data to backtest {symbol}")

        train_end = len(df) - test_days
        if train_end < 40:
            raise ValueError("Not enough training data for signal backtest")

        def run_system(score_fn, label):
            correct = 0
            directional = 0
            neutral = 0
            total = 0
            daily_results = []

            for i in range(test_days):
                idx = train_end + i
                if idx >= len(df):
                    break

                window = df.iloc[max(0, idx - 60):idx]
                if len(window) < 20:
                    continue

                score = score_fn(window)
                actual_next = float(df.iloc[idx]["close"])
                actual_prev = float(df.iloc[idx - 1]["close"])
                actual_dir = "UP" if actual_next > actual_prev else "DOWN"

                if score > 5:
                    predicted_dir = "UP"
                    directional += 1
                elif score < -5:
                    predicted_dir = "DOWN"
                    directional += 1
                else:
                    predicted_dir = "NEUTRAL"
                    neutral += 1

                was_correct = (
                    (predicted_dir == "UP" and actual_dir == "UP") or
                    (predicted_dir == "DOWN" and actual_dir == "DOWN") or
                    (predicted_dir == "NEUTRAL" and abs(actual_next - actual_prev) / actual_prev < 0.015)
                )
                if was_correct:
                    correct += 1
                total += 1

                daily_results.append({
                    "date": str(df.iloc[idx].get("date", idx)),
                    "score": round(score, 2),
                    "predicted": predicted_dir,
                    "actual": actual_dir,
                    "correct": was_correct,
                })

            accuracy = round(correct / total * 100, 1) if total > 0 else 0
            return {
                "system": label,
                "accuracy": accuracy,
                "total_days": total,
                "directional_calls": directional,
                "neutral_calls": neutral,
                "correct": correct,
                "daily_results": daily_results[-30:],
            }

        new_result = run_system(self._score_new_system, "new")
        old_result = run_system(self._score_old_system, "old")

        return {
            "symbol": symbol,
            "test_days": test_days,
            "new_system": new_result,
            "old_system": old_result,
        }


backtest_service = BacktestService()
