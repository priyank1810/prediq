from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import WatchlistItem, SignalLog
from app.schemas import WatchlistItemCreate
from app.services.data_fetcher import data_fetcher
from app.utils.helpers import is_index

router = APIRouter()


@router.get("")
def list_watchlist(db: Session = Depends(get_db)):
    items = (
        db.query(WatchlistItem)
        .order_by(WatchlistItem.added_at.desc())
        .all()
    )
    return [
        {
            "id": item.id,
            "symbol": item.symbol,
            "item_type": item.item_type,
            "added_at": item.added_at.isoformat() if item.added_at else None,
        }
        for item in items
    ]


@router.post("")
def add_to_watchlist(
    data: WatchlistItemCreate,
    db: Session = Depends(get_db),
):
    symbol = data.symbol.upper()
    existing = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.symbol == symbol)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail=f"{symbol} already in watchlist")

    item_type = "index" if is_index(symbol) else data.item_type
    item = WatchlistItem(symbol=symbol, item_type=item_type)
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"id": item.id, "symbol": item.symbol, "item_type": item.item_type}


@router.delete("/{symbol}")
def remove_from_watchlist(
    symbol: str,
    db: Session = Depends(get_db),
):
    item = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.symbol == symbol.upper())
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail=f"{symbol} not in watchlist")
    db.delete(item)
    db.commit()
    return {"ok": True}


@router.get("/overview")
def watchlist_overview(db: Session = Depends(get_db)):
    items = (
        db.query(WatchlistItem)
        .order_by(WatchlistItem.added_at.desc())
        .all()
    )
    results = []
    for item in items:
        try:
            quote = data_fetcher.get_live_quote(item.symbol)
        except Exception:
            quote = {}

        latest_signal = (
            db.query(SignalLog)
            .filter(SignalLog.symbol == item.symbol)
            .order_by(SignalLog.created_at.desc())
            .first()
        )

        results.append({
            "symbol": item.symbol,
            "item_type": item.item_type,
            "ltp": quote.get("ltp", 0),
            "change": quote.get("change", 0),
            "pct_change": quote.get("pct_change", 0),
            "signal_direction": latest_signal.direction if latest_signal else None,
            "signal_confidence": latest_signal.confidence if latest_signal else None,
        })
    return results
