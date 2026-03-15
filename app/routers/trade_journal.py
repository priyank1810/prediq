from fastapi import APIRouter, HTTPException, Query
from app.database import SessionLocal
from app.models import TradeJournal
from app.schemas import TradeJournalCreate
from app.utils.helpers import validate_symbol

router = APIRouter()

@router.get("/")
def list_trades(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    db = SessionLocal()
    try:
        trades = (
            db.query(TradeJournal)
            .order_by(TradeJournal.created_at.desc())
            .offset(offset).limit(limit).all()
        )
        return [
            {
                "id": t.id, "symbol": t.symbol, "action": t.action,
                "price": t.price, "quantity": t.quantity, "notes": t.notes,
                "signal_direction": t.signal_direction,
                "signal_confidence": t.signal_confidence,
                "pnl": t.pnl, "pnl_pct": t.pnl_pct, "tags": t.tags,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in trades
        ]
    finally:
        db.close()

@router.post("/")
def create_trade(data: TradeJournalCreate):
    try:
        sym = validate_symbol(data.symbol)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {data.symbol!r}")
    if data.action not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail="Action must be 'buy' or 'sell'")
    db = SessionLocal()
    try:
        trade = TradeJournal(
            symbol=sym, action=data.action, price=data.price,
            quantity=data.quantity, notes=data.notes,
            signal_direction=data.signal_direction,
            signal_confidence=data.signal_confidence,
            pnl=data.pnl, pnl_pct=data.pnl_pct, tags=data.tags,
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)
        return {"id": trade.id, "symbol": trade.symbol, "action": trade.action, "created_at": trade.created_at.isoformat()}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.delete("/{trade_id}")
def delete_trade(trade_id: int):
    db = SessionLocal()
    try:
        trade = db.query(TradeJournal).filter(TradeJournal.id == trade_id).first()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        db.delete(trade)
        db.commit()
        return {"ok": True}
    finally:
        db.close()

@router.get("/stats")
def trade_stats():
    """Summary stats: total trades, win rate, avg P&L."""
    from sqlalchemy import func
    db = SessionLocal()
    try:
        total = db.query(func.count(TradeJournal.id)).scalar() or 0
        sells = db.query(TradeJournal).filter(TradeJournal.action == "sell", TradeJournal.pnl.isnot(None)).all()
        wins = sum(1 for s in sells if (s.pnl or 0) > 0)
        total_pnl = sum(s.pnl or 0 for s in sells)
        avg_pnl_pct = sum(s.pnl_pct or 0 for s in sells) / len(sells) if sells else 0
        return {
            "total_trades": total,
            "sell_trades": len(sells),
            "wins": wins,
            "win_rate": round(wins / len(sells) * 100, 1) if sells else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_pct": round(avg_pnl_pct, 2),
        }
    finally:
        db.close()
