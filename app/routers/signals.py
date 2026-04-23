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


@router.get("/stats/predictions/{symbol}")
def prediction_history(symbol: str, page: int = 1, per_page: int = 20):
    """Get prediction history for a specific stock from PredictionLog."""
    db = SessionLocal()
    try:
        from sqlalchemy import func
        sym = symbol.upper()
        query = db.query(PredictionLog).filter(
            PredictionLog.symbol == sym,
            PredictionLog.actual_price.isnot(None),
        )
        total = query.count()
        logs = query.order_by(PredictionLog.target_date.desc()).offset((page - 1) * per_page).limit(per_page).all()

        predictions = []
        for log in logs:
            mape = abs(log.predicted_price - log.actual_price) / log.actual_price * 100 if log.actual_price else None
            predictions.append({
                "model": log.model_type,
                "prediction_date": log.prediction_date.isoformat() if log.prediction_date else None,
                "target_date": log.target_date.isoformat() if log.target_date else None,
                "predicted_price": round(log.predicted_price, 2),
                "actual_price": round(log.actual_price, 2) if log.actual_price else None,
                "mape": round(mape, 2) if mape is not None else None,
                "accurate": mape is not None and mape <= 2,
                "direction_correct": mape is not None and mape <= 5,
            })

        return {
            "symbol": sym,
            "predictions": predictions,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, -(-total // per_page)),
        }
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
        for model_name in ["xgboost"]:
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

        # --- Per-symbol breakdown (aggregate across models, min 5 predictions) ---
        symbol_stats = {}
        for log in all_logs:
            sym = log.symbol
            if sym not in symbol_stats:
                symbol_stats[sym] = {"mape_sum": 0, "total": 0, "correct_dir": 0, "accurate": 0}
            s = symbol_stats[sym]
            s["total"] += 1
            mape = abs(log.predicted_price - log.actual_price) / log.actual_price * 100
            s["mape_sum"] += mape
            threshold = _get_accuracy_threshold(log)
            if mape <= threshold:
                s["accurate"] += 1
            # Direction: predicted move vs actual move (using MAPE < 5% as proxy)
            if mape <= 5:
                s["correct_dir"] += 1

        by_symbol = []
        for symbol, s in symbol_stats.items():
            if s["total"] >= 5:
                by_symbol.append({
                    "symbol": symbol,
                    "total": s["total"],
                    "avg_mape": round(s["mape_sum"] / s["total"], 2),
                    "win_rate": round(s["accurate"] / s["total"] * 100, 1),
                    "direction_accuracy": round(s["correct_dir"] / s["total"] * 100, 1),
                })
        by_symbol.sort(key=lambda x: -x["win_rate"])
        by_symbol = by_symbol[:20]

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


@router.get("/stats/trades/history")
def trade_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=5, le=100),
    direction: str = Query(None, description="BULLISH or BEARISH"),
    status: str = Query(None, description="target_hit, sl_hit, correct, wrong"),
    symbol: str = Query(None),
    timeframe: str = Query(None),
    min_pnl: float = Query(None, description="Minimum P&L %"),
    max_pnl: float = Query(None, description="Maximum P&L %"),
    min_confidence: float = Query(None, description="Minimum confidence %"),
    sort_by: str = Query("created_at", description="created_at, outcome_pct, confidence, symbol"),
    sort_order: str = Query("desc", description="asc or desc"),
    date_from: str = Query(None, description="YYYY-MM-DD"),
    date_to: str = Query(None, description="YYYY-MM-DD"),
):
    """Paginated, filterable, sortable trade history."""
    from app.models import TradeSignalLog
    from datetime import datetime

    db = SessionLocal()
    try:
        query = db.query(TradeSignalLog).filter(
            TradeSignalLog.status != "open",
            ~TradeSignalLog.timeframe.in_(["intraday_10m", "intraday_15m"]),
        )

        # Filters
        if direction:
            query = query.filter(TradeSignalLog.direction == direction.upper())
        if status:
            query = query.filter(TradeSignalLog.status == status)
        if symbol:
            query = query.filter(TradeSignalLog.symbol == symbol.upper())
        if timeframe:
            query = query.filter(TradeSignalLog.timeframe == timeframe)
        if min_pnl is not None:
            query = query.filter(TradeSignalLog.outcome_pct >= min_pnl)
        if max_pnl is not None:
            query = query.filter(TradeSignalLog.outcome_pct <= max_pnl)
        if min_confidence is not None:
            query = query.filter(TradeSignalLog.confidence >= min_confidence)
        if date_from:
            try:
                dt = datetime.strptime(date_from, "%Y-%m-%d")
                query = query.filter(TradeSignalLog.created_at >= dt)
            except ValueError:
                pass
        if date_to:
            try:
                dt = datetime.strptime(date_to, "%Y-%m-%d")
                from datetime import timedelta
                query = query.filter(TradeSignalLog.created_at < dt + timedelta(days=1))
            except ValueError:
                pass

        # Count before pagination
        total = query.count()

        # Sort
        sort_col = getattr(TradeSignalLog, sort_by, TradeSignalLog.created_at)
        if sort_order == "asc":
            query = query.order_by(sort_col.asc())
        else:
            query = query.order_by(sort_col.desc())

        # Paginate
        offset = (page - 1) * per_page
        trades = query.offset(offset).limit(per_page).all()

        results = [{
            "id": t.id,
            "symbol": t.symbol,
            "timeframe": t.timeframe,
            "direction": t.direction,
            "confidence": t.confidence,
            "entry": t.entry,
            "target": t.target,
            "stop_loss": t.stop_loss,
            "predicted_price": t.predicted_price,
            "outcome_price": t.outcome_price,
            "outcome_pct": t.outcome_pct,
            "status": t.status,
            "highest_price": t.highest_price,
            "lowest_price": t.lowest_price,
            "prediction_error": round(abs(t.predicted_price - t.outcome_price) / t.outcome_price * 100, 2)
                if t.predicted_price and t.outcome_price and t.outcome_price > 0 else None,
            "model_used": t.model_used or "v1",
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
        } for t in trades]

        return {
            "trades": results,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }
    finally:
        db.close()


@router.get("/stats/ai-analysis")
def ai_analysis():
    """Comprehensive AI analysis — detailed breakdown of all predictions."""
    from app.models import TradeSignalLog
    from sqlalchemy import func
    from collections import defaultdict

    db = SessionLocal()
    try:
        # Filter out stale index trades with bad data
        from app.config import INDICES
        index_symbols = list(INDICES.keys())
        resolved = db.query(TradeSignalLog).filter(
            TradeSignalLog.status != "open",
            ~TradeSignalLog.symbol.in_(index_symbols),
            ~TradeSignalLog.timeframe.in_(["intraday_10m", "intraday_15m"]),
        ).all()
        open_count = db.query(TradeSignalLog).filter(TradeSignalLog.status == "open").count()

        if not resolved:
            return {"total": 0}

        total = len(resolved)
        wins = [r for r in resolved if r.status in ("target_hit", "correct")]
        losses = [r for r in resolved if r.status in ("sl_hit", "wrong")]

        # Overall
        win_rate = round(len(wins) / total * 100, 1)
        avg_win = round(sum(r.outcome_pct for r in wins if r.outcome_pct) / max(len(wins), 1), 2)
        avg_loss = round(sum(r.outcome_pct for r in losses if r.outcome_pct) / max(len(losses), 1), 2)

        # By direction
        by_direction = {}
        for dir_val in ["BULLISH", "BEARISH"]:
            d_trades = [r for r in resolved if r.direction == dir_val]
            if d_trades:
                d_wins = sum(1 for r in d_trades if r.status in ("target_hit", "correct"))
                d_pnls = [r.outcome_pct for r in d_trades if r.outcome_pct]
                by_direction[dir_val] = {
                    "total": len(d_trades),
                    "win_rate": round(d_wins / len(d_trades) * 100, 1),
                    "avg_pnl": round(sum(d_pnls) / len(d_pnls), 2) if d_pnls else 0,
                    "best": round(max(d_pnls), 2) if d_pnls else 0,
                    "worst": round(min(d_pnls), 2) if d_pnls else 0,
                }

        # By timeframe
        by_timeframe = {}
        for tf in set(r.timeframe for r in resolved if r.timeframe):
            tf_trades = [r for r in resolved if r.timeframe == tf]
            tf_wins = sum(1 for r in tf_trades if r.status in ("target_hit", "correct"))
            tf_pnls = [r.outcome_pct for r in tf_trades if r.outcome_pct]
            by_timeframe[tf] = {
                "total": len(tf_trades),
                "win_rate": round(tf_wins / len(tf_trades) * 100, 1),
                "avg_pnl": round(sum(tf_pnls) / len(tf_pnls), 2) if tf_pnls else 0,
                "target_hits": sum(1 for r in tf_trades if r.status == "target_hit"),
                "sl_hits": sum(1 for r in tf_trades if r.status == "sl_hit"),
            }

        # By confidence bucket (5% increments)
        def _conf_buckets(rows):
            out = []
            for lo in range(0, 100, 5):
                hi = lo + 5
                bucket = [r for r in rows if r.confidence and lo <= r.confidence < hi]
                if bucket:
                    b_wins = sum(1 for r in bucket if r.status in ("target_hit", "correct"))
                    b_losses = sum(1 for r in bucket if r.status in ("sl_hit", "wrong"))
                    b_pnls = [r.outcome_pct for r in bucket if r.outcome_pct]
                    out.append({
                        "range": f"{lo}-{hi}%",
                        "total": len(bucket),
                        "wins": b_wins,
                        "losses": b_losses,
                        "win_rate": round(b_wins / len(bucket) * 100, 1),
                        "loss_rate": round(b_losses / len(bucket) * 100, 1),
                        "avg_pnl": round(sum(b_pnls) / len(b_pnls), 2) if b_pnls else 0,
                    })
            return out

        by_confidence = _conf_buckets(resolved)

        # By confidence × timeframe
        by_confidence_tf = {}
        for tf in set(r.timeframe for r in resolved if r.timeframe):
            tf_rows = [r for r in resolved if r.timeframe == tf]
            buckets = _conf_buckets(tf_rows)
            if buckets:
                by_confidence_tf[tf] = buckets

        # By time of day
        by_hour = []
        for h in range(9, 16):
            h_trades = [r for r in resolved if r.created_at and r.created_at.hour == h]
            if h_trades:
                h_wins = sum(1 for r in h_trades if r.status in ("target_hit", "correct"))
                by_hour.append({
                    "hour": f"{h}:00-{h+1}:00",
                    "total": len(h_trades),
                    "win_rate": round(h_wins / len(h_trades) * 100, 1),
                })

        # By stock (top 15)
        stock_stats = defaultdict(lambda: {"wins": 0, "total": 0, "pnl_sum": 0})
        for r in resolved:
            s = stock_stats[r.symbol]
            s["total"] += 1
            if r.status in ("target_hit", "correct"):
                s["wins"] += 1
            if r.outcome_pct:
                s["pnl_sum"] += r.outcome_pct

        by_stock = sorted([
            {
                "symbol": sym,
                "total": s["total"],
                "win_rate": round(s["wins"] / s["total"] * 100, 1),
                "avg_pnl": round(s["pnl_sum"] / s["total"], 2),
            }
            for sym, s in stock_stats.items()
        ], key=lambda x: -x["total"])[:15]

        # By status breakdown
        status_counts = defaultdict(int)
        for r in resolved:
            status_counts[r.status] += 1

        # Prediction error
        pred_errors = [
            abs(r.predicted_price - r.outcome_price) / r.outcome_price * 100
            for r in resolved
            if r.predicted_price and r.outcome_price and r.outcome_price > 0
        ]

        # Best and worst trades
        sorted_by_pnl = sorted([r for r in resolved if r.outcome_pct], key=lambda x: -x.outcome_pct)
        best_trades = [{
            "symbol": r.symbol, "timeframe": r.timeframe, "direction": r.direction,
            "pnl": r.outcome_pct, "confidence": r.confidence,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        } for r in sorted_by_pnl[:5]]
        worst_trades = [{
            "symbol": r.symbol, "timeframe": r.timeframe, "direction": r.direction,
            "pnl": r.outcome_pct, "confidence": r.confidence,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        } for r in sorted_by_pnl[-5:]]

        # Daily P&L trend with by-timeframe breakdown
        def _day_stats():
            return {
                "pnl_sum": 0, "trades": 0, "wins": 0, "losses": 0,
                "by_timeframe": defaultdict(lambda: {"total": 0, "wins": 0, "pnl_sum": 0}),
            }
        daily_pnl = defaultdict(_day_stats)
        for r in resolved:
            if r.created_at:
                day = r.created_at.strftime("%Y-%m-%d")
                d = daily_pnl[day]
                d["trades"] += 1
                pct = r.outcome_pct or 0
                d["pnl_sum"] += pct
                is_win = pct > 0
                if is_win:
                    d["wins"] += 1
                elif pct < 0:
                    d["losses"] += 1
                tf = r.timeframe or "unknown"
                d["by_timeframe"][tf]["total"] += 1
                d["by_timeframe"][tf]["pnl_sum"] += pct
                if is_win:
                    d["by_timeframe"][tf]["wins"] += 1

        daily_trend = []
        for day, v in sorted(daily_pnl.items()):
            tf_stats = {
                tf: {
                    "total": s["total"],
                    "wins": s["wins"],
                    "win_rate": round(s["wins"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
                    "total_pnl": round(s["pnl_sum"], 2),
                    "avg_pnl": round(s["pnl_sum"] / s["total"], 2) if s["total"] > 0 else 0,
                }
                for tf, s in v["by_timeframe"].items()
            }
            daily_trend.append({
                "date": day,
                "trades": v["trades"],
                "wins": v["wins"],
                "losses": v["losses"],
                "win_rate": round(v["wins"] / v["trades"] * 100, 1) if v["trades"] > 0 else 0,
                "total_pnl": round(v["pnl_sum"], 2),
                "avg_pnl": round(v["pnl_sum"] / v["trades"], 2) if v["trades"] > 0 else 0,
                "pnl": round(v["pnl_sum"] / v["trades"], 2) if v["trades"] > 0 else 0,  # backwards compat
                "by_timeframe": tf_stats,
            })

        return {
            "total": total,
            "open": open_count,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "status_counts": dict(status_counts),
            "by_direction": by_direction,
            "by_timeframe": by_timeframe,
            "by_confidence": by_confidence,
            "by_confidence_tf": by_confidence_tf,
            "by_hour": by_hour,
            "by_stock": by_stock,
            "prediction_error": {
                "avg": round(sum(pred_errors) / len(pred_errors), 2) if pred_errors else None,
                "within_1pct": round(sum(1 for e in pred_errors if e <= 1) / len(pred_errors) * 100, 1) if pred_errors else 0,
                "within_2pct": round(sum(1 for e in pred_errors if e <= 2) / len(pred_errors) * 100, 1) if pred_errors else 0,
                "total": len(pred_errors),
            },
            "best_trades": best_trades,
            "worst_trades": worst_trades,
            "daily_trend": daily_trend[-30:],
        }
    finally:
        db.close()


@router.get("/stats/job-status")
def job_queue_status():
    """Diagnostic: show job queue state."""
    from app.models import JobQueue
    from sqlalchemy import func
    db = SessionLocal()
    try:
        counts = dict(db.query(JobQueue.status, func.count()).group_by(JobQueue.status).all())
        recent = db.query(JobQueue).order_by(JobQueue.created_at.desc()).limit(10).all()
        return {
            "counts": counts,
            "recent": [{
                "id": j.id, "type": j.job_type, "status": j.status,
                "error": j.error[:200] if j.error else None,
                "created": j.created_at.isoformat() if j.created_at else None,
                "started": j.started_at.isoformat() if j.started_at else None,
                "completed": j.completed_at.isoformat() if j.completed_at else None,
            } for j in recent]
        }
    finally:
        db.close()


@router.get("/stats/scan-status")
def get_scan_status():
    """Get last scan status — when it ran, how many stocks, results."""
    try:
        from app.utils.cache import cache
        status = cache.get("last_scan_status")
        return status or {"message": "No scan has run yet"}
    except Exception:
        return {"message": "Scan status unavailable"}


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


@router.get("/stats/trades/confidence-analysis")
def confidence_bucket_analysis():
    """Temporary: WR and avg P&L by confidence bucket per timeframe group."""
    db = SessionLocal()
    try:
        from app.models import TradeSignalLog
        from collections import defaultdict
        trades = (
            db.query(TradeSignalLog)
            .filter(
                TradeSignalLog.status.in_(["target_hit","sl_hit","correct","wrong","expired"]),
                TradeSignalLog.confidence != None,
                TradeSignalLog.timeframe != None,
                ~TradeSignalLog.timeframe.in_(["intraday_10m","intraday_15m"]),
            )
            .all()
        )
        buckets = defaultdict(list)
        for t in trades:
            tf = "intraday" if (t.timeframe or "").startswith("intraday") else "short_term"
            c = t.confidence or 0
            if c < 45: b = "<45"
            elif c < 50: b = "45-49"
            elif c < 55: b = "50-54"
            elif c < 60: b = "55-59"
            elif c < 65: b = "60-64"
            else: b = "65+"
            buckets[(tf, b)].append({"pnl": t.outcome_pct or 0, "status": t.status})
        result = {}
        for (tf, b), ts in buckets.items():
            wins = sum(1 for t in ts if t["pnl"] > 0)
            avg = sum(t["pnl"] for t in ts) / len(ts)
            result[f"{tf}/{b}"] = {
                "total": len(ts), "win_rate": round(wins/len(ts)*100,1),
                "avg_pnl": round(avg, 3),
                "target_hit": sum(1 for t in ts if t["status"]=="target_hit"),
                "sl_hit": sum(1 for t in ts if t["status"]=="sl_hit"),
            }
        return {"total_trades": len(trades), "buckets": result}
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
