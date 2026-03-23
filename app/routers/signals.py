import asyncio
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func
from app.database import SessionLocal
from app.models import SignalLog, PredictionLog
from app.utils.helpers import validate_symbol

router = APIRouter()


@router.get("/market-mood")
def get_market_mood():
    """Get Market Mood Score (0-100)."""
    try:
        from app.services.market_mood_service import market_mood_service
        return market_mood_service.get_mood()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market mood computation failed: {str(e)}")


@router.get("/stats/by-sector")
def stats_by_sector():
    """Get signal accuracy grouped by sector."""
    db = SessionLocal()
    try:
        from sqlalchemy import case
        rows = (
            db.query(
                SignalLog.sector,
                func.count(SignalLog.id).label("total"),
                func.sum(case((SignalLog.was_correct == True, 1), else_=0)).label("correct"),
            )
            .filter(SignalLog.was_correct.isnot(None), SignalLog.sector.isnot(None))
            .group_by(SignalLog.sector)
            .order_by(func.count(SignalLog.id).desc())
            .all()
        )
        return [
            {"sector": row.sector, "total": row.total, "correct": int(row.correct),
             "accuracy": round(int(row.correct) / row.total * 100, 1) if row.total > 0 else 0}
            for row in rows
        ]
    finally:
        db.close()


@router.get("/stats/by-horizon")
def stats_by_horizon():
    """Get prediction accuracy grouped by horizon."""
    db = SessionLocal()
    try:
        from sqlalchemy import case, func as sqlfunc
        logs = db.query(PredictionLog).filter(PredictionLog.actual_price.isnot(None)).all()
        stats = {}
        for log in logs:
            # Infer horizon from date difference
            days = (log.target_date - log.prediction_date).days
            if days <= 1:
                horizon = "1d"
            elif days <= 5:
                horizon = "1w"
            elif days <= 22:
                horizon = "1mo"
            elif days <= 66:
                horizon = "3mo"
            else:
                horizon = "1y"

            if horizon not in stats:
                stats[horizon] = {"total": 0, "mape_sum": 0}
            stats[horizon]["total"] += 1
            if log.actual_price > 0:
                mape = abs(log.predicted_price - log.actual_price) / log.actual_price * 100
                stats[horizon]["mape_sum"] += mape

        return [
            {"horizon": h, "total": d["total"],
             "avg_mape": round(d["mape_sum"] / d["total"], 2) if d["total"] > 0 else 0}
            for h, d in stats.items()
        ]
    finally:
        db.close()


@router.get("/stats/prediction-leaderboard")
def prediction_leaderboard():
    """Prediction accuracy leaderboard: compare Prophet vs XGBoost vs Ensemble.

    Returns per-model stats (MAPE, directional accuracy, win rate) and
    breakdowns by symbol and sector.
    """
    from datetime import date
    from sqlalchemy import case
    db = SessionLocal()
    try:
        # Only consider predictions where actual_price has been filled
        base = db.query(PredictionLog).filter(
            PredictionLog.actual_price.isnot(None),
            PredictionLog.actual_price > 0,
        )

        all_logs = base.all()
        if not all_logs:
            return {"models": [], "by_symbol": [], "by_sector": []}

        # --- Per-model aggregate stats ---
        # Accuracy threshold based on prediction horizon
        def _get_accuracy_threshold(log):
            """Stricter thresholds for shorter timeframes."""
            days = (log.target_date - log.prediction_date).days if log.target_date and log.prediction_date else 1
            if days <= 0:
                return 0.3   # intraday: 0.3%
            elif days <= 1:
                return 0.5   # 1 day: 0.5%
            elif days <= 5:
                return 1.0   # 1 week: 1%
            elif days <= 22:
                return 1.5   # 1 month: 1.5%
            else:
                return 2.0   # 3+ months: 2%

        model_stats = {}
        for log in all_logs:
            m = log.model_type
            if m not in model_stats:
                model_stats[m] = {"mape_sum": 0, "correct_dir": 0, "accurate": 0, "total": 0}
            s = model_stats[m]
            s["total"] += 1
            mape = abs(log.predicted_price - log.actual_price) / log.actual_price * 100
            s["mape_sum"] += mape

            threshold = _get_accuracy_threshold(log)
            if mape <= threshold:
                s["accurate"] += 1
            if mape <= 5:
                s["correct_dir"] += 1

        models = []
        for model_name in ["ensemble", "prophet", "xgboost"]:
            s = model_stats.get(model_name)
            if not s or s["total"] == 0:
                continue
            models.append({
                "model": model_name,
                "total": s["total"],
                "avg_mape": round(s["mape_sum"] / s["total"], 2),
                "win_rate": round(s["accurate"] / s["total"] * 100, 1),
                "directional_accuracy": round(s["correct_dir"] / s["total"] * 100, 1),
            })

        # Sort by lowest MAPE (best first)
        models.sort(key=lambda x: x["avg_mape"])

        # --- Per-model per-symbol breakdown (top 15 most predicted) ---
        symbol_stats = {}
        for log in all_logs:
            key = (log.model_type, log.symbol)
            if key not in symbol_stats:
                symbol_stats[key] = {"mape_sum": 0, "total": 0}
            s = symbol_stats[key]
            s["total"] += 1
            s["mape_sum"] += abs(log.predicted_price - log.actual_price) / log.actual_price * 100

        by_symbol = []
        for (model, symbol), s in symbol_stats.items():
            if s["total"] >= 2:  # Need at least 2 predictions
                by_symbol.append({
                    "model": model, "symbol": symbol,
                    "total": s["total"],
                    "avg_mape": round(s["mape_sum"] / s["total"], 2),
                })
        by_symbol.sort(key=lambda x: x["avg_mape"])
        by_symbol = by_symbol[:30]

        # --- Per-model per-sector breakdown ---
        sector_stats = {}
        for log in all_logs:
            if not log.sector:
                continue
            key = (log.model_type, log.sector)
            if key not in sector_stats:
                sector_stats[key] = {"mape_sum": 0, "total": 0}
            s = sector_stats[key]
            s["total"] += 1
            s["mape_sum"] += abs(log.predicted_price - log.actual_price) / log.actual_price * 100

        by_sector = []
        for (model, sector), s in sector_stats.items():
            if s["total"] >= 2:
                by_sector.append({
                    "model": model, "sector": sector,
                    "total": s["total"],
                    "avg_mape": round(s["mape_sum"] / s["total"], 2),
                })
        by_sector.sort(key=lambda x: x["avg_mape"])

        return {"models": models, "by_symbol": by_symbol, "by_sector": by_sector}
    finally:
        db.close()


@router.get("/stats/by-regime")
def stats_by_regime():
    """Get signal accuracy grouped by market regime."""
    db = SessionLocal()
    try:
        from sqlalchemy import case
        rows = (
            db.query(
                SignalLog.regime,
                func.count(SignalLog.id).label("total"),
                func.sum(case((SignalLog.was_correct == True, 1), else_=0)).label("correct"),
            )
            .filter(SignalLog.was_correct.isnot(None), SignalLog.regime.isnot(None))
            .group_by(SignalLog.regime)
            .order_by(func.count(SignalLog.id).desc())
            .all()
        )
        return [
            {"regime": row.regime, "total": row.total, "correct": int(row.correct),
             "accuracy": round(int(row.correct) / row.total * 100, 1) if row.total > 0 else 0}
            for row in rows
        ]
    finally:
        db.close()


@router.get("/stats/backtest-pnl")
def backtest_pnl_simulation():
    """Hypothetical P&L simulation from prediction signals."""
    db = SessionLocal()
    try:
        logs = (
            db.query(PredictionLog)
            .filter(PredictionLog.actual_price.isnot(None))
            .order_by(PredictionLog.prediction_date)
            .limit(500)
            .all()
        )
        if not logs:
            return {"trades": 0, "total_pnl_pct": 0, "win_rate": 0, "max_drawdown": 0}

        wins = 0
        total = 0
        pnl_series = []
        cumulative = 0

        for log in logs:
            if log.actual_price <= 0:
                continue
            pct = (log.actual_price - log.predicted_price) / log.predicted_price * 100
            # Assume we go long if predicted > current implicit price, short otherwise
            trade_pnl = -abs(pct) if abs(pct) > 5 else abs(pct) * 0.5  # Simplified
            predicted_direction = 1 if log.predicted_price > log.actual_price * 0.99 else -1
            actual_direction = 1 if log.actual_price > log.predicted_price * 0.99 else -1
            correct = predicted_direction == actual_direction
            if correct:
                wins += 1
            total += 1
            cumulative += (1 if correct else -1) * 0.5
            pnl_series.append(round(cumulative, 2))

        # Max drawdown
        peak = 0
        max_dd = 0
        for val in pnl_series:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd

        return {
            "trades": total,
            "total_pnl_pct": round(cumulative, 2),
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "max_drawdown": round(max_dd, 2),
            "pnl_series": pnl_series[-100:],  # Last 100 data points
        }
    finally:
        db.close()


@router.get("/stats/backtest-signal")
def backtest_signal(symbol: str = Query(...), test_days: int = Query(60, ge=10, le=252)):
    """Backtest new vs old signal system on historical data."""
    try:
        from app.services.backtest_service import backtest_service
        return backtest_service.backtest_signal(symbol.upper(), test_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.get("/stats/backtest-trades")
def backtest_trades(
    symbol: str = Query(...),
    test_days: int = Query(120, ge=20, le=500),
    capital: float = Query(100000, gt=0),
    min_confidence: int = Query(0, ge=0, le=100),
):
    """Full trade simulation with P&L, Sharpe ratio, max drawdown, win rate."""
    try:
        from app.services.backtest_service import backtest_service
        return backtest_service.backtest_trades(
            symbol.upper(), test_days=test_days,
            initial_capital=capital, min_confidence=min_confidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trade backtest failed: {str(e)}")


@router.get("/stats/backtest-predictions")
def backtest_predictions(
    symbol: str = Query(...),
    horizon: str = Query("1d"),
    test_days: int = Query(60, ge=10, le=252),
):
    """Walk-forward backtest of XGBoost, Prophet, and Ensemble prediction models."""
    try:
        from app.services.backtest_service import backtest_service
        return backtest_service.backtest_predictions(symbol.upper(), horizon=horizon, test_days=test_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction backtest failed: {str(e)}")


@router.get("/stats/backtest-portfolio")
def backtest_portfolio(
    symbols: str = Query("", description="Comma-separated symbols (empty = top 10 Nifty)"),
    test_days: int = Query(60, ge=10, le=252),
):
    """Multi-symbol portfolio backtest with aggregate metrics."""
    try:
        from app.services.backtest_service import backtest_service
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] or None
        return backtest_service.backtest_portfolio(symbols=sym_list, test_days=test_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio backtest failed: {str(e)}")


@router.post("/stats/visual-backtest")
def visual_backtest(payload: dict):
    """Run a visual backtest with equity curve, drawdown, trades, and metrics."""
    try:
        from app.services.backtest_service import backtest_service
        symbol = payload.get("symbol", "").upper()
        if not symbol:
            raise ValueError("symbol is required")
        strategy_params = payload.get("strategy_params", {})
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")
        return backtest_service.run_visual_backtest(
            symbol, strategy_params,
            start_date=start_date, end_date=end_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual backtest failed: {str(e)}")


@router.post("/stats/monte-carlo")
def monte_carlo(payload: dict):
    """Run Monte Carlo simulation on a backtest equity curve."""
    try:
        from app.services.backtest_service import backtest_service
        equity_curve = payload.get("equity_curve")
        simulations = payload.get("simulations", 1000)
        if not equity_curve:
            raise ValueError("equity_curve is required")
        return backtest_service.run_monte_carlo(equity_curve, simulations=simulations)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monte Carlo simulation failed: {str(e)}")


@router.get("/scan/high-confidence")
def scan_high_confidence(threshold: int = Query(60, ge=0, le=100)):
    """Get the most recent high-confidence signal for each symbol."""
    db = SessionLocal()
    try:
        # Subquery: latest signal per symbol
        subq = (
            db.query(
                SignalLog.symbol,
                func.max(SignalLog.id).label("max_id"),
            )
            .group_by(SignalLog.symbol)
            .subquery()
        )

        logs = (
            db.query(SignalLog)
            .join(subq, SignalLog.id == subq.c.max_id)
            .filter(SignalLog.confidence >= threshold)
            .filter(SignalLog.direction != "NEUTRAL")
            .order_by(SignalLog.confidence.desc())
            .all()
        )

        return [
            {
                "symbol": log.symbol,
                "direction": log.direction,
                "confidence": log.confidence,
                "composite_score": log.composite_score,
                "technical_score": log.technical_score,
                "sentiment_score": log.sentiment_score,
                "global_score": log.global_score,
                "price_at_signal": log.price_at_signal,
                "was_correct": log.was_correct,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ]
    finally:
        db.close()


@router.get("/stats/near-bullish")
def get_near_bullish_stocks():
    """Get popular stocks that are close to turning bullish — opportunities to watch."""
    try:
        from app.utils.cache import cache
        near = cache.get("near_bullish_stocks") or []
        return near
    except Exception:
        return []


@router.get("/stats/virtual-portfolio")
def get_virtual_portfolio(capital: float = Query(10000, gt=0)):
    """Get virtual portfolio performance based on AI trade signals."""
    try:
        from app.services.virtual_portfolio import virtual_portfolio
        return virtual_portfolio.get_portfolio(capital)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/trades")
def trade_prediction_stats(symbol: str = Query(None)):
    """Get trade prediction accuracy stats (target hit vs stop-loss hit vs expired)."""
    try:
        from app.services.trade_tracker import trade_tracker
        return trade_tracker.get_accuracy_stats(symbol=symbol.upper() if symbol else None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/trades/open")
def open_trade_predictions():
    """Get all currently open trade predictions being tracked."""
    db = SessionLocal()
    try:
        from app.models import TradeSignalLog
        open_trades = (
            db.query(TradeSignalLog)
            .filter(TradeSignalLog.status == "open")
            .order_by(TradeSignalLog.created_at.desc())
            .limit(50)
            .all()
        )
        return [{
            "id": t.id,
            "symbol": t.symbol,
            "timeframe": t.timeframe,
            "direction": t.direction,
            "confidence": t.confidence,
            "current_price": t.current_price,
            "predicted_price": t.predicted_price,
            "entry": t.entry,
            "target": t.target,
            "stop_loss": t.stop_loss,
            "risk_reward": t.risk_reward,
            "highest_price": t.highest_price,
            "lowest_price": t.lowest_price,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "expires_at": t.expires_at.isoformat() if t.expires_at else None,
        } for t in open_trades]
    finally:
        db.close()


@router.post("/stats/trades/validate")
def manually_validate_trades():
    """Manually trigger validation of open trade predictions."""
    try:
        from app.services.trade_tracker import trade_tracker
        result = trade_tracker.validate_open_signals()
        learn_result = trade_tracker.learn_from_trades()
        return {**result, "learning": learn_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stats/learning/rebuild")
def rebuild_all_learning_profiles():
    """Manually trigger a rebuild of all stock learning profiles."""
    try:
        from app.services.stock_learner import stock_learner
        result = stock_learner.rebuild_all_profiles()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/learning/{symbol}")
def get_stock_learning_profile(symbol: str):
    """Get the per-stock learning profile showing what the AI has learned about this stock."""
    try:
        from app.services.stock_learner import stock_learner
        profile = stock_learner.get_profile(symbol.upper())
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Not enough signal history for {symbol.upper()} (need {10}+ validated signals)"
            )
        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/accuracy")
def signal_accuracy_stats():
    """Get signal accuracy stats grouped by symbol."""
    db = SessionLocal()
    try:
        from sqlalchemy import case
        rows = (
            db.query(
                SignalLog.symbol,
                func.count(SignalLog.id).label("total"),
                func.sum(case((SignalLog.was_correct == True, 1), else_=0)).label("correct"),
            )
            .filter(SignalLog.was_correct.isnot(None))
            .group_by(SignalLog.symbol)
            .order_by(func.count(SignalLog.id).desc())
            .limit(20)
            .all()
        )
        return [
            {
                "symbol": row.symbol,
                "total": row.total,
                "correct": int(row.correct),
                "accuracy": round(int(row.correct) / row.total * 100, 1) if row.total > 0 else 0,
            }
            for row in rows
        ]
    finally:
        db.close()


@router.get("/multi-timeframe/{symbol}")
async def get_multi_timeframe_signals(symbol: str):
    """Get separate Intraday, Short-term (2 weeks), and Long-term signals with entry/exit levels."""
    try:
        sym = symbol.upper()
        from app.services.signal_service import signal_service
        result = await asyncio.to_thread(signal_service.get_multi_timeframe_signals, sym)
        if not result:
            raise ValueError(f"No data for {sym}")
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-timeframe signal failed: {str(e)}")


# --- Existing endpoints (must come AFTER scan/ and stats/ routes) ---

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal computation failed: {str(e)}")


@router.get("/{symbol}/history")
def get_signal_history(
    symbol: str,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        db = SessionLocal()
        try:
            logs = (
                db.query(SignalLog)
                .filter(SignalLog.symbol == symbol.upper())
                .order_by(SignalLog.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": log.id,
                    "direction": log.direction,
                    "confidence": log.confidence,
                    "composite_score": log.composite_score,
                    "technical_score": log.technical_score,
                    "sentiment_score": log.sentiment_score,
                    "global_score": log.global_score,
                    "price_at_signal": log.price_at_signal,
                    "price_after_15min": log.price_after_15min,
                    "price_after_30min": getattr(log, "price_after_30min", None),
                    "price_after_1hr": getattr(log, "price_after_1hr", None),
                    "was_correct": log.was_correct,
                    "was_correct_30min": getattr(log, "was_correct_30min", None),
                    "was_correct_1hr": getattr(log, "was_correct_1hr", None),
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in logs
            ]
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
