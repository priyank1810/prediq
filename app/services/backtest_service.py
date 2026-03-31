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
        """Walk-forward backtest of XGBoost predictions."""
        from app.ai.xgboost_model import XGBoostPredictor
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

        results = {"xgboost": [], "ensemble": []}

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

            ensemble_pred = xgb_pred  # Single model = ensemble

            for model_name, pred in [("xgboost", xgb_pred), ("ensemble", ensemble_pred)]:
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


    # ────────────────────────────────────────────────────────────────
    # 5. Visual backtest (equity curve, drawdown, trades, monthly returns)
    # ────────────────────────────────────────────────────────────────

    def run_visual_backtest(
        self,
        symbol: str,
        strategy_params: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """Full visual backtest with equity curve, drawdown, trade list, and monthly returns.

        strategy_params keys:
            signal_type: 'composite' | 'technical' | 'sentiment'  (default 'composite')
            confidence_threshold: int 0-100  (default 0)
            stop_loss_pct: float  (default 5.0)
            take_profit_pct: float  (default 10.0)
            holding_period_days: int  (default 0 = unlimited)
        """
        df = data_fetcher.get_historical_data(symbol, period="5y")
        if df is None or df.empty or len(df) < 100:
            raise ValueError(f"Not enough data to backtest {symbol}")

        # Ensure we have a proper date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif df.index.name == "date" or df.index.name == "Date":
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"] if "date" in df.columns else df["Date"])
        else:
            df["date"] = pd.to_datetime(df.index)

        df = df.sort_values("date").reset_index(drop=True)

        # Apply date filters
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        if len(df) < 60:
            raise ValueError("Not enough data in selected date range")

        # Extract strategy params
        signal_type = strategy_params.get("signal_type", "composite")
        confidence_threshold = strategy_params.get("confidence_threshold", 0)
        stop_loss_pct = strategy_params.get("stop_loss_pct", 5.0) / 100.0
        take_profit_pct = strategy_params.get("take_profit_pct", 10.0) / 100.0
        holding_period_days = strategy_params.get("holding_period_days", 0)

        initial_capital = 100000.0
        capital = initial_capital
        position = 0  # 0=flat, 1=long
        entry_price = 0.0
        entry_date = None
        entry_idx = 0
        trades = []
        equity_curve = []
        daily_returns = []

        for i in range(60, len(df)):
            window = df.iloc[max(0, i - 60):i]
            current_price = float(df.iloc[i]["close"])
            current_date = df.iloc[i]["date"]
            date_str = current_date.strftime("%Y-%m-%d")

            # Compute signal score
            score = self._get_signal_score(window, signal_type)
            confidence = abs(score)

            # Check exit conditions if in position
            if position == 1:
                pnl_pct_unrealized = (current_price - entry_price) / entry_price
                days_held = i - entry_idx

                should_exit = False
                exit_reason = ""

                # Stop loss
                if pnl_pct_unrealized <= -stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"
                # Take profit
                elif pnl_pct_unrealized >= take_profit_pct:
                    should_exit = True
                    exit_reason = "take_profit"
                # Holding period
                elif holding_period_days > 0 and days_held >= holding_period_days:
                    should_exit = True
                    exit_reason = "holding_period"
                # Signal reversal
                elif score < -SIGNAL_DIRECTION_THRESHOLD:
                    should_exit = True
                    exit_reason = "signal_reversal"

                if should_exit:
                    pnl = (current_price - entry_price) / entry_price - COMMISSION_PCT
                    capital *= (1 + pnl)
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": date_str,
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(current_price, 2),
                        "pnl": round(pnl * capital / (1 + pnl), 2),
                        "pnl_pct": round(pnl * 100, 2),
                        "direction": "LONG",
                        "holding_days": days_held,
                        "exit_reason": exit_reason,
                    })
                    position = 0

            # Check entry conditions if flat
            if position == 0 and confidence >= confidence_threshold:
                if score > SIGNAL_DIRECTION_THRESHOLD:
                    position = 1
                    entry_price = current_price
                    entry_date = date_str
                    entry_idx = i
                    capital -= capital * COMMISSION_PCT  # entry commission

            # Mark-to-market equity
            if position == 1:
                unrealized = (current_price - entry_price) / entry_price
                mtm = capital * (1 + unrealized)
            else:
                mtm = capital

            equity_curve.append({"date": date_str, "equity": round(mtm, 2)})

            if len(equity_curve) >= 2:
                prev_eq = equity_curve[-2]["equity"]
                daily_returns.append((mtm - prev_eq) / prev_eq if prev_eq > 0 else 0.0)
            else:
                daily_returns.append(0.0)

        # Close any open position at the end
        if position == 1 and len(df) > 0:
            final_price = float(df.iloc[-1]["close"])
            pnl = (final_price - entry_price) / entry_price - COMMISSION_PCT
            capital *= (1 + pnl)
            trades.append({
                "entry_date": entry_date,
                "exit_date": df.iloc[-1]["date"].strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 2),
                "exit_price": round(final_price, 2),
                "pnl": round(pnl * capital / (1 + pnl), 2),
                "pnl_pct": round(pnl * 100, 2),
                "direction": "LONG",
                "holding_days": len(df) - 1 - entry_idx,
                "exit_reason": "end_of_period",
            })

        # Compute drawdown curve
        drawdown_curve = self._compute_drawdown_curve(equity_curve)

        # Compute monthly returns
        monthly_returns = self._compute_monthly_returns(equity_curve)

        # Compute full metrics
        metrics = self._compute_visual_metrics(
            trades, daily_returns, equity_curve, initial_capital
        )

        return {
            "symbol": symbol,
            "strategy_params": strategy_params,
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve,
            "trades": trades,
            "metrics": metrics,
            "monthly_returns": monthly_returns,
        }

    def _get_signal_score(self, window: pd.DataFrame, signal_type: str) -> float:
        """Get signal score based on signal type."""
        if signal_type == "technical":
            return self._score_old_system(window)
        elif signal_type == "sentiment":
            # Use new system which includes sentiment
            return self._score_new_system(window)
        else:
            # composite — use new system (default)
            return self._score_new_system(window)

    def _compute_drawdown_curve(self, equity_curve: list) -> list:
        """Compute drawdown percentage at each point."""
        if not equity_curve:
            return []
        peak = equity_curve[0]["equity"]
        drawdown = []
        for point in equity_curve:
            eq = point["equity"]
            if eq > peak:
                peak = eq
            dd_pct = ((peak - eq) / peak * 100) if peak > 0 else 0
            drawdown.append({
                "date": point["date"],
                "drawdown_pct": round(dd_pct, 2),
            })
        return drawdown

    def _compute_monthly_returns(self, equity_curve: list) -> list:
        """Compute monthly return percentages."""
        if not equity_curve or len(equity_curve) < 2:
            return []

        monthly = {}
        for point in equity_curve:
            month_key = point["date"][:7]  # YYYY-MM
            if month_key not in monthly:
                monthly[month_key] = {"first": point["equity"], "last": point["equity"]}
            monthly[month_key]["last"] = point["equity"]

        result = []
        prev_last = None
        for month_key in sorted(monthly.keys()):
            m = monthly[month_key]
            base = prev_last if prev_last is not None else m["first"]
            ret_pct = ((m["last"] - base) / base * 100) if base > 0 else 0
            result.append({
                "month": month_key,
                "return_pct": round(ret_pct, 2),
            })
            prev_last = m["last"]

        return result

    def _compute_visual_metrics(self, trades, daily_returns, equity_curve, initial_capital):
        """Compute comprehensive metrics for visual backtest."""
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        total_trades = len(trades)

        win_rate = round(len(wins) / total_trades * 100, 1) if total_trades > 0 else 0
        avg_win = round(np.mean([t["pnl_pct"] for t in wins]), 2) if wins else 0
        avg_loss = round(np.mean([abs(t["pnl_pct"]) for t in losses]), 2) if losses else 0

        gross_profit = sum(t["pnl_pct"] for t in wins)
        gross_loss = sum(abs(t["pnl_pct"]) for t in losses)
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0
        )

        avg_holding = round(np.mean([t["holding_days"] for t in trades]), 1) if trades else 0

        # Total return
        final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
        total_return = round((final_equity - initial_capital) / initial_capital * 100, 2)

        # CAGR
        if equity_curve and len(equity_curve) > 1:
            days = len(equity_curve)
            years = days / TRADING_DAYS_PER_YEAR
            if years > 0 and final_equity > 0:
                cagr = round(((final_equity / initial_capital) ** (1 / years) - 1) * 100, 2)
            else:
                cagr = 0
        else:
            cagr = 0

        # Sharpe, Sortino
        ret_arr = np.array(daily_returns) if daily_returns else np.array([0.0])
        mean_ret = float(np.mean(ret_arr))
        std_ret = float(np.std(ret_arr, ddof=1)) if len(ret_arr) > 1 else 0
        sharpe = round(mean_ret / std_ret * math.sqrt(TRADING_DAYS_PER_YEAR), 2) if std_ret > 0 else 0

        downside = ret_arr[ret_arr < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0
        sortino = round(mean_ret / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR), 2) if downside_std > 0 else 0

        # Max drawdown
        peak = initial_capital
        max_dd = 0
        max_dd_date = ""
        for point in equity_curve:
            eq = point["equity"]
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_date = point["date"]

        max_dd_pct = round(max_dd * 100, 2)

        # Calmar ratio
        calmar = round(cagr / max_dd_pct, 2) if max_dd_pct > 0 else 0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd_pct,
            "max_drawdown_date": max_dd_date,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": total_trades,
            "avg_holding_days": avg_holding,
        }

    # ────────────────────────────────────────────────────────────────
    # 6. Monte Carlo simulation
    # ────────────────────────────────────────────────────────────────

    def run_monte_carlo(self, equity_curve: list, simulations: int = 1000) -> dict:
        """Shuffle trade returns to generate simulated equity paths with percentile bands."""
        if not equity_curve or len(equity_curve) < 10:
            raise ValueError("Not enough equity curve data for Monte Carlo simulation")

        # Extract daily returns from the equity curve
        returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]["equity"]
            curr = equity_curve[i]["equity"]
            if prev > 0:
                returns.append(curr / prev - 1)

        if len(returns) < 5:
            raise ValueError("Not enough return data for Monte Carlo")

        returns_arr = np.array(returns)
        n_days = len(returns_arr)
        initial = equity_curve[0]["equity"]

        # Run simulations: shuffle daily returns each time
        sim_paths = np.zeros((simulations, n_days + 1))
        sim_paths[:, 0] = initial

        for s in range(simulations):
            shuffled = np.random.permutation(returns_arr)
            for d in range(n_days):
                sim_paths[s, d + 1] = sim_paths[s, d] * (1 + shuffled[d])

        # Compute percentile bands at each time step
        percentiles = [5, 25, 50, 75, 95]
        bands = {str(p): [] for p in percentiles}
        dates = [equity_curve[0]["date"]] + [pt["date"] for pt in equity_curve[1:]]

        for d in range(n_days + 1):
            vals = sim_paths[:, d]
            for p in percentiles:
                pval = float(np.percentile(vals, p))
                bands[str(p)].append({
                    "date": dates[d] if d < len(dates) else dates[-1],
                    "equity": round(pval, 2),
                })

        # Terminal distribution stats
        terminal = sim_paths[:, -1]
        terminal_stats = {
            "mean": round(float(np.mean(terminal)), 2),
            "median": round(float(np.median(terminal)), 2),
            "std": round(float(np.std(terminal)), 2),
            "p5": round(float(np.percentile(terminal, 5)), 2),
            "p95": round(float(np.percentile(terminal, 95)), 2),
            "prob_profit": round(float(np.mean(terminal > initial) * 100), 1),
        }

        return {
            "simulations": simulations,
            "bands": bands,
            "terminal_stats": terminal_stats,
        }


backtest_service = BacktestService()
