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


# --- Existing endpoints (must come AFTER scan/ and stats/ routes) ---

@router.get("/{symbol}")
def get_intraday_signal(symbol: str):
    try:
        sym = symbol.upper()
        signal = signal_service.get_signal(sym)
        if not signal:
            raise HTTPException(status_code=404, detail=f"No signal data for {symbol}")

        # Log to SignalLog on-demand (replaces background scanner)
        try:
            from datetime import timedelta
            from app.utils.helpers import now_ist
            db = SessionLocal()
            try:
                # Only log if last log for this symbol is >1 min old (avoid spam)
                last_log = (
                    db.query(SignalLog)
                    .filter(SignalLog.symbol == sym)
                    .order_by(SignalLog.created_at.desc())
                    .first()
                )
                should_log = (
                    last_log is None
                    or (now_ist() - last_log.created_at) > timedelta(minutes=1)
                )
                if should_log:
                    price = None
                    candles = signal.get("intraday_candles", [])
                    if candles:
                        price = candles[-1].get("close")
                    log = SignalLog(
                        symbol=sym,
                        direction=signal["direction"],
                        confidence=signal["confidence"],
                        composite_score=signal["composite_score"],
                        technical_score=signal["technical"]["score"],
                        sentiment_score=signal["sentiment"]["score"],
                        global_score=signal["global_market"]["score"],
                        price_at_signal=price,
                    )
                    db.add(log)
                    db.commit()
            finally:
                db.close()
        except Exception:
            pass  # Don't fail the response if logging fails

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
