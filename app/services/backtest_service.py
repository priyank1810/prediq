import logging
import math
import numpy as np
import pandas as pd
import ta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.data_fetcher import data_fetcher
from app.config import (
    SIGNAL_DIRECTION_THRESHOLD,
    ACCURACY_BASE_THRESHOLD,
    ACCURACY_NEUTRAL_THRESHOLD,
    LOW_VOLATILITY_SYMBOLS,
    LOW_VOLATILITY_THRESHOLD,
    NIFTY_50_SYMBOLS,
)

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
COMMISSION_PCT = 0.001  # 0.1% round-trip (brokerage + STT + charges)


class BacktestService:

    # ────────────────────────────────────────────────────────────────
    # 1. Signal system backtest (old vs new, kept for backward compat)
    # ────────────────────────────────────────────────────────────────

    def _score_new_system(self, window: pd.DataFrame) -> float:
        from app.services.indicator_service import indicator_service
        if len(window) < 20:
            return 0.0
        w = window.copy()
        if "datetime_str" not in w.columns:
            if "date" in w.columns:
                w["datetime_str"] = w["date"].astype(str)
            else:
                w["datetime_str"] = [str(i) for i in range(len(w))]
        result = indicator_service.compute_intraday_indicators(w)
        return result["score"]

    def _score_old_system(self, window: pd.DataFrame) -> float:
        if len(window) < 20:
            return 0.0
        close = window["close"]
        current_price = float(close.iloc[-1])
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
        ma5 = ta.trend.SMAIndicator(close, window=5).sma_indicator()
        ma9 = ta.trend.SMAIndicator(close, window=9).sma_indicator()
        ma5_now = float(ma5.iloc[-1]) if not pd.isna(ma5.iloc[-1]) else current_price
        ma9_now = float(ma9.iloc[-1]) if not pd.isna(ma9.iloc[-1]) else current_price
        ma_spread_pct = ((ma5_now - ma9_now) / current_price * 100) if current_price > 0 else 0
        ma_score = max(-100, min(100, ma_spread_pct * 300))
        return max(-100, min(100, 0.40 * rsi_score + 0.30 * bb_score + 0.30 * ma_score))

    def backtest_signal(self, symbol: str, test_days: int = 60) -> dict:
        """Walk-forward backtest comparing new vs old signal system (backward compat)."""
        df = data_fetcher.get_historical_data(symbol, period="2y")
        if df is None or df.empty or len(df) < 100:
            raise ValueError(f"Not enough data to backtest {symbol}")
        train_end = len(df) - test_days
        if train_end < 40:
            raise ValueError("Not enough training data for signal backtest")
        correctness_threshold = (
            LOW_VOLATILITY_THRESHOLD if symbol in LOW_VOLATILITY_SYMBOLS
            else ACCURACY_BASE_THRESHOLD
        )

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
                pct_move = (actual_next - actual_prev) / actual_prev
                actual_dir = "UP" if actual_next > actual_prev else "DOWN"
                if score > SIGNAL_DIRECTION_THRESHOLD:
                    predicted_dir = "UP"
                    directional += 1
                elif score < -SIGNAL_DIRECTION_THRESHOLD:
                    predicted_dir = "DOWN"
                    directional += 1
                else:
                    predicted_dir = "NEUTRAL"
                    neutral += 1
                was_correct = (
                    (predicted_dir == "UP" and pct_move >= correctness_threshold) or
                    (predicted_dir == "DOWN" and pct_move <= -correctness_threshold) or
                    (predicted_dir == "NEUTRAL" and abs(pct_move) < ACCURACY_NEUTRAL_THRESHOLD)
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
        return {"symbol": symbol, "test_days": test_days, "new_system": new_result, "old_system": old_result}

    # ────────────────────────────────────────────────────────────────
    # 2. Full signal trade simulation
    # ────────────────────────────────────────────────────────────────

    def backtest_trades(
        self,
        symbol: str,
        test_days: int = 120,
        initial_capital: float = 100000.0,
        min_confidence: int = 0,
    ) -> dict:
        """Simulate trades based on signal scores with full P&L and risk metrics."""
        df = data_fetcher.get_historical_data(symbol, period="2y")
        if df is None or df.empty or len(df) < 100:
            raise ValueError(f"Not enough data to backtest {symbol}")

        train_end = len(df) - test_days
        if train_end < 60:
            raise ValueError("Not enough training data")

        capital = initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity_curve = []
        daily_returns = []

        for i in range(test_days):
            idx = train_end + i
            if idx >= len(df):
                break

            window = df.iloc[max(0, idx - 60):idx]
            if len(window) < 20:
                equity_curve.append(capital)
                daily_returns.append(0.0)
                continue

            score = self._score_new_system(window)
            confidence = abs(score)
            current_price = float(df.iloc[idx]["close"])
            date_str = str(df.iloc[idx].get("date", idx))

            if position != 0:
                pnl_pct = (current_price - entry_price) / entry_price * (1 if position > 0 else -1)
                pnl_pct -= COMMISSION_PCT
                trade_pnl = capital * abs(position) / initial_capital * pnl_pct * initial_capital
                should_close = (
                    (position > 0 and score < -SIGNAL_DIRECTION_THRESHOLD) or
                    (position < 0 and score > SIGNAL_DIRECTION_THRESHOLD) or
                    (confidence < min_confidence)
                )
                if should_close:
                    capital += trade_pnl
                    trades.append({
                        "date": date_str,
                        "action": "CLOSE",
                        "price": current_price,
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "capital": round(capital, 2),
                    })
                    position = 0

            if position == 0 and confidence >= min_confidence:
                if score > SIGNAL_DIRECTION_THRESHOLD:
                    position = 1
                    entry_price = current_price
                    trades.append({"date": date_str, "action": "BUY", "price": current_price, "score": round(score, 1)})
                elif score < -SIGNAL_DIRECTION_THRESHOLD:
                    position = -1
                    entry_price = current_price
                    trades.append({"date": date_str, "action": "SHORT", "price": current_price, "score": round(score, 1)})

            if position != 0:
                unrealized = (current_price - entry_price) / entry_price * (1 if position > 0 else -1)
                mtm = capital + capital * unrealized
            else:
                mtm = capital
            equity_curve.append(round(mtm, 2))

            if len(equity_curve) >= 2 and equity_curve[-2] != 0:
                daily_returns.append((equity_curve[-1] - equity_curve[-2]) / equity_curve[-2])
            else:
                daily_returns.append(0.0)

        if position != 0 and len(df) > train_end:
            final_price = float(df.iloc[min(train_end + test_days - 1, len(df) - 1)]["close"])
            pnl_pct = (final_price - entry_price) / entry_price * (1 if position > 0 else -1) - COMMISSION_PCT
            capital += capital * pnl_pct
            trades.append({"date": "END", "action": "CLOSE", "price": final_price, "pnl_pct": round(pnl_pct * 100, 3)})

        metrics = self._compute_metrics(trades, daily_returns, equity_curve, initial_capital)

        start_price = float(df.iloc[train_end]["close"])
        end_price = float(df.iloc[min(train_end + test_days - 1, len(df) - 1)]["close"])
        buy_hold_return = (end_price - start_price) / start_price * 100

        return {
            "symbol": symbol,
            "test_days": test_days,
            "initial_capital": initial_capital,
            "final_capital": round(capital, 2),
            "total_return_pct": round((capital - initial_capital) / initial_capital * 100, 2),
            "buy_hold_return_pct": round(buy_hold_return, 2),
            "metrics": metrics,
            "trades": trades[-50:],
            "equity_curve": equity_curve[-120:],
        }

    # ────────────────────────────────────────────────────────────────
    # 3. Prediction model backtest (walk-forward)
    # ────────────────────────────────────────────────────────────────

    def backtest_predictions(
        self,
        symbol: str,
        horizon: str = "1d",
        test_days: int = 60,
    ) -> dict:
        """Walk-forward backtest of XGBoost, Prophet, and Ensemble predictions."""
        from app.ai.xgboost_model import XGBoostPredictor
        from app.ai.prophet_model import ProphetPredictor
        from app.config import PREDICTION_HORIZONS

        if horizon not in PREDICTION_HORIZONS:
            raise ValueError(f"Invalid horizon: {horizon}")

        h_config = PREDICTION_HORIZONS[horizon]
        is_intraday = h_config.get("intraday", False)

        if is_intraday:
            df = data_fetcher.get_intraday_data(symbol, period="60d", interval=h_config.get("interval", "15m"))
            bars_ahead = h_config.get("bars", 1)
        else:
            df = data_fetcher.get_historical_data(symbol, period="2y")
            bars_ahead = h_config.get("days", 1)

        if df is None or df.empty or len(df) < 200:
            raise ValueError(f"Not enough data for {symbol} prediction backtest")

        test_start = len(df) - test_days - bars_ahead
        if test_start < 120:
            raise ValueError("Not enough training data for prediction backtest")

        xgb = XGBoostPredictor()
        prophet = ProphetPredictor()

        results = {"xgboost": [], "prophet": [], "ensemble": []}

        for i in range(test_days):
            idx = test_start + i
            if idx + bars_ahead >= len(df):
                break

            train_df = df.iloc[:idx].copy()
            actual_price = float(df.iloc[idx + bars_ahead]["close"])
            current_price = float(df.iloc[idx]["close"])

            xgb_pred = None
            try:
                if is_intraday:
                    xgb_result = xgb.predict_intraday(train_df, symbol=symbol, horizon=horizon)
                else:
                    xgb_result = xgb.predict(train_df, symbol=symbol, horizon=horizon)
                if xgb_result and xgb_result.get("predictions"):
                    preds_list = xgb_result["predictions"]
                    xgb_pred = preds_list[min(bars_ahead - 1, len(preds_list) - 1)]
            except Exception as e:
                logger.debug(f"XGBoost predict failed at step {i}: {e}")

            prophet_pred = None
            try:
                prophet_result = prophet.predict(train_df, horizon=horizon, symbol=symbol)
                if prophet_result and prophet_result.get("predictions"):
                    preds_list = prophet_result["predictions"]
                    prophet_pred = preds_list[min(bars_ahead - 1, len(preds_list) - 1)]
            except Exception as e:
                logger.debug(f"Prophet predict failed at step {i}: {e}")

            preds = [p for p in [xgb_pred, prophet_pred] if p is not None]
            ensemble_pred = np.mean(preds) if preds else None

            for model_name, pred in [("xgboost", xgb_pred), ("prophet", prophet_pred), ("ensemble", ensemble_pred)]:
                if pred is not None:
                    error = pred - actual_price
                    pct_error = error / actual_price * 100
                    pred_dir = "UP" if pred > current_price else "DOWN"
                    actual_dir = "UP" if actual_price > current_price else "DOWN"
                    results[model_name].append({
                        "predicted": round(pred, 2),
                        "actual": round(actual_price, 2),
                        "current": round(current_price, 2),
                        "error_pct": round(pct_error, 3),
                        "direction_correct": pred_dir == actual_dir,
                    })

        summary = {}
        for model_name, entries in results.items():
            if not entries:
                summary[model_name] = {"status": "no_predictions", "count": 0}
                continue

            errors = [abs(e["error_pct"]) for e in entries]
            dir_correct = sum(1 for e in entries if e["direction_correct"])
            total = len(entries)

            sim_returns = []
            for e in entries:
                actual_move = (e["actual"] - e["current"]) / e["current"]
                if e["direction_correct"]:
                    sim_returns.append(abs(actual_move) - COMMISSION_PCT)
                else:
                    sim_returns.append(-abs(actual_move) - COMMISSION_PCT)

            cumulative_return = 1.0
            for r in sim_returns:
                cumulative_return *= (1 + r)
            cumulative_return = (cumulative_return - 1) * 100

            summary[model_name] = {
                "count": total,
                "mape": round(np.mean(errors), 3),
                "median_ape": round(np.median(errors), 3),
                "directional_accuracy": round(dir_correct / total * 100, 1),
                "cumulative_return_pct": round(cumulative_return, 2),
                "avg_error_pct": round(np.mean([e["error_pct"] for e in entries]), 3),
                "best_5": sorted(entries, key=lambda x: abs(x["error_pct"]))[:5],
                "worst_5": sorted(entries, key=lambda x: -abs(x["error_pct"]))[:5],
            }

        return {
            "symbol": symbol,
            "horizon": horizon,
            "test_days": test_days,
            "models": summary,
        }

    # ────────────────────────────────────────────────────────────────
    # 4. Multi-symbol portfolio backtest
    # ────────────────────────────────────────────────────────────────

    def backtest_portfolio(
        self,
        symbols: Optional[list] = None,
        test_days: int = 60,
    ) -> dict:
        """Run signal backtest across multiple symbols and aggregate."""
        if symbols is None:
            symbols = NIFTY_50_SYMBOLS[:10]

        results = []
        errors = []

        def _run(sym):
            try:
                return self.backtest_trades(sym, test_days=test_days)
            except Exception as e:
                return {"symbol": sym, "error": str(e)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_run, s): s for s in symbols}
            for future in as_completed(futures):
                res = future.result()
                if "error" in res:
                    errors.append(res)
                else:
                    results.append(res)

        if not results:
            return {"symbols": symbols, "results": [], "errors": errors, "aggregate": None}

        total_returns = [r["total_return_pct"] for r in results]
        buy_holds = [r["buy_hold_return_pct"] for r in results]
        all_sharpes = [r["metrics"]["sharpe_ratio"] for r in results if r["metrics"].get("sharpe_ratio") is not None]
        all_win_rates = [r["metrics"]["win_rate"] for r in results]

        aggregate = {
            "symbols_tested": len(results),
            "avg_return_pct": round(np.mean(total_returns), 2),
            "median_return_pct": round(np.median(total_returns), 2),
            "avg_buy_hold_pct": round(np.mean(buy_holds), 2),
            "alpha_pct": round(np.mean(total_returns) - np.mean(buy_holds), 2),
            "avg_sharpe": round(np.mean(all_sharpes), 2) if all_sharpes else None,
            "avg_win_rate": round(np.mean(all_win_rates), 1),
            "best": max(results, key=lambda r: r["total_return_pct"])["symbol"],
            "worst": min(results, key=lambda r: r["total_return_pct"])["symbol"],
        }

        per_symbol = [
            {
                "symbol": r["symbol"],
                "return_pct": r["total_return_pct"],
                "buy_hold_pct": r["buy_hold_return_pct"],
                "alpha": round(r["total_return_pct"] - r["buy_hold_return_pct"], 2),
                "sharpe": r["metrics"]["sharpe_ratio"],
                "max_drawdown": r["metrics"]["max_drawdown_pct"],
                "win_rate": r["metrics"]["win_rate"],
                "total_trades": r["metrics"]["total_trades"],
            }
            for r in sorted(results, key=lambda r: -r["total_return_pct"])
        ]

        return {
            "test_days": test_days,
            "aggregate": aggregate,
            "per_symbol": per_symbol,
            "errors": errors,
        }

    # ────────────────────────────────────────────────────────────────
    # Helper: compute risk/return metrics
    # ────────────────────────────────────────────────────────────────

    def _compute_metrics(self, trades, daily_returns, equity_curve, initial_capital):
        closed = [t for t in trades if t.get("pnl_pct") is not None]
        wins = [t for t in closed if t["pnl_pct"] > 0]
        losses = [t for t in closed if t["pnl_pct"] <= 0]

        win_rate = round(len(wins) / len(closed) * 100, 1) if closed else 0
        avg_win = round(np.mean([t["pnl_pct"] for t in wins]), 3) if wins else 0
        avg_loss = round(np.mean([abs(t["pnl_pct"]) for t in losses]), 3) if losses else 0
        profit_factor = round(
            sum(t["pnl_pct"] for t in wins) / sum(abs(t["pnl_pct"]) for t in losses), 2
        ) if losses and sum(abs(t["pnl_pct"]) for t in losses) > 0 else float("inf") if wins else 0

        if len(daily_returns) > 1:
            ret_arr = np.array(daily_returns)
            mean_ret = np.mean(ret_arr)
            std_ret = np.std(ret_arr, ddof=1)
            sharpe = round(mean_ret / std_ret * math.sqrt(TRADING_DAYS_PER_YEAR), 2) if std_ret > 0 else None
        else:
            sharpe = None

        peak = initial_capital
        max_dd = 0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": round(max_dd * 100, 2),
        }


backtest_service = BacktestService()
