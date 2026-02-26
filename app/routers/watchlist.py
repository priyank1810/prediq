from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import WatchlistItem, SignalLog, User
from app.schemas import WatchlistItemCreate
from app.services.data_fetcher import data_fetcher
from app.utils.helpers import is_index
from app.auth import get_current_active_user

router = APIRouter()


@router.get("")
def list_watchlist(db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
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
    user: User = Depends(get_current_active_user),
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
    user: User = Depends(get_current_active_user),
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
def watchlist_overview(db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    items = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id)
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
