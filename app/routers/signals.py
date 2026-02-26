from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func
from app.services.signal_service import signal_service
from app.database import SessionLocal
from app.models import SignalLog

router = APIRouter()


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
