from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func
from app.services.signal_service import signal_service
from app.database import SessionLocal
from app.models import SignalLog, PredictionLog

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
        logs = (
            db.query(SignalLog.sector, SignalLog.was_correct)
            .filter(SignalLog.was_correct.isnot(None), SignalLog.sector.isnot(None))
            .all()
        )
        stats = {}
        for sector, was_correct in logs:
            if sector not in stats:
                stats[sector] = {"total": 0, "correct": 0}
            stats[sector]["total"] += 1
            if was_correct:
                stats[sector]["correct"] += 1
        return [
            {"sector": s, "total": d["total"], "correct": d["correct"],
             "accuracy": round(d["correct"] / d["total"] * 100, 1) if d["total"] > 0 else 0}
            for s, d in sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)
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


@router.get("/stats/by-regime")
def stats_by_regime():
    """Get signal accuracy grouped by market regime."""
    db = SessionLocal()
    try:
        logs = (
            db.query(SignalLog.regime, SignalLog.was_correct)
            .filter(SignalLog.was_correct.isnot(None), SignalLog.regime.isnot(None))
            .all()
        )
        stats = {}
        for regime, was_correct in logs:
            if regime not in stats:
                stats[regime] = {"total": 0, "correct": 0}
            stats[regime]["total"] += 1
            if was_correct:
                stats[regime]["correct"] += 1
        return [
            {"regime": r, "total": d["total"], "correct": d["correct"],
             "accuracy": round(d["correct"] / d["total"] * 100, 1) if d["total"] > 0 else 0}
            for r, d in sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)
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


@router.get("/stats/accuracy")
def signal_accuracy_stats():
    """Get signal accuracy stats grouped by symbol."""
    db = SessionLocal()
    try:
        all_logs = (
            db.query(SignalLog.symbol, SignalLog.was_correct)
            .filter(SignalLog.was_correct.isnot(None))
            .all()
        )

        stats = {}
        for symbol, was_correct in all_logs:
            if symbol not in stats:
                stats[symbol] = {"total": 0, "correct": 0}
            stats[symbol]["total"] += 1
            if was_correct:
                stats[symbol]["correct"] += 1

        return sorted(
            [
                {
                    "symbol": symbol,
                    "total": s["total"],
                    "correct": s["correct"],
                    "accuracy": round(s["correct"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
                }
                for symbol, s in stats.items()
            ],
            key=lambda x: x["total"],
            reverse=True,
        )[:20]
    finally:
        db.close()


# --- Existing endpoints (must come AFTER scan/ and stats/ routes) ---

@router.get("/{symbol}")
def get_intraday_signal(symbol: str):
    try:
        signal = signal_service.get_signal(symbol.upper())
        if not signal:
            raise HTTPException(status_code=404, detail=f"No signal data for {symbol}")
        return signal
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal computation failed: {str(e)}")


@router.get("/{symbol}/history")
def get_signal_history(symbol: str, limit: int = 20):
    try:
        db = SessionLocal()
        try:
            logs = (
                db.query(SignalLog)
                .filter(SignalLog.symbol == symbol.upper())
                .order_by(SignalLog.created_at.desc())
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
                    "was_correct": log.was_correct,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in logs
            ]
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
