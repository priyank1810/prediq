from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import WatchlistItem, SignalLog
from app.schemas import WatchlistItemCreate
from app.services.data_fetcher import data_fetcher
from app.utils.helpers import is_index
from app.auth import get_current_active_user

router = APIRouter()


@router.get("")
def list_watchlist(db: Session = Depends(get_db), user=Depends(get_current_active_user)):
    items = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id)
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
    user=Depends(get_current_active_user),
):
    symbol = data.symbol.upper()
    existing = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.symbol == symbol, WatchlistItem.user_id == user.id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail=f"{symbol} already in watchlist")

    item_type = "index" if is_index(symbol) else data.item_type
    item = WatchlistItem(symbol=symbol, item_type=item_type, user_id=user.id)
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"id": item.id, "symbol": item.symbol, "item_type": item.item_type}


@router.delete("/{symbol}")
def remove_from_watchlist(
    symbol: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    item = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.symbol == symbol.upper(), WatchlistItem.user_id == user.id)
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail=f"{symbol} not in watchlist")
    db.delete(item)
    db.commit()
    return {"ok": True}


@router.get("/overview")
def watchlist_overview(db: Session = Depends(get_db), user=Depends(get_current_active_user)):
    items = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.added_at.desc())
        .all()
    )
    if not items:
        return []

    # Batch-fetch all quotes in one call instead of N individual calls
    symbols = [item.symbol for item in items]
    quotes = data_fetcher.get_bulk_quotes(symbols)
    quote_map = {q["symbol"]: q for q in quotes if q.get("symbol")}

    # Batch-fetch latest signals for all symbols
    from sqlalchemy import func
    latest_ids = (
        db.query(func.max(SignalLog.id))
        .filter(SignalLog.symbol.in_(symbols))
        .group_by(SignalLog.symbol)
    )
    signals = (
        db.query(SignalLog)
        .filter(SignalLog.id.in_(latest_ids))
        .all()
    )
    signal_map = {s.symbol: s for s in signals}

    results = []
    for item in items:
        quote = quote_map.get(item.symbol, {})
        latest_signal = signal_map.get(item.symbol)

        # Volume ratio: current volume / 20-day avg
        volume = quote.get("volume", 0) or 0
        avg_volume = quote.get("avg_volume", 0) or 0
        volume_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else None

        results.append({
            "symbol": item.symbol,
            "item_type": item.item_type,
            "ltp": quote.get("ltp", 0),
            "change": quote.get("change", 0),
            "pct_change": quote.get("pct_change", 0),
            "open": quote.get("open", 0),
            "day_high": quote.get("high", 0),
            "day_low": quote.get("low", 0),
            "volume_ratio": volume_ratio,
            "sentiment_score": latest_signal.sentiment_score if latest_signal else None,
            "signal_direction": latest_signal.direction if latest_signal else None,
            "signal_confidence": latest_signal.confidence if latest_signal else None,
        })
    return results
