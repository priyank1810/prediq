from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, PortfolioHolding, PriceAlert, WatchlistItem, PredictionLog, SignalLog
from app.schemas import UserResponse, UserUpdate
from app.auth import get_admin_user

router = APIRouter()


@router.get("/users", response_model=list[UserResponse])
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    return db.query(User).order_by(User.created_at.desc()).all()


@router.put("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    data: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if data.is_active is not None:
        user.is_active = data.is_active
    if data.role is not None:
        if data.role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")
        user.role = data.role

    db.commit()
    db.refresh(user)
    return user


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete user's data
    db.query(PortfolioHolding).filter(PortfolioHolding.user_id == user_id).delete()
    db.query(PriceAlert).filter(PriceAlert.user_id == user_id).delete()
    db.query(WatchlistItem).filter(WatchlistItem.user_id == user_id).delete()
    db.delete(user)
    db.commit()
    return {"message": "User deleted"}


@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    return {
        "total_users": db.query(User).count(),
        "active_users": db.query(User).filter(User.is_active == True).count(),
        "total_holdings": db.query(PortfolioHolding).count(),
        "total_alerts": db.query(PriceAlert).count(),
        "total_watchlist_items": db.query(WatchlistItem).count(),
        "total_predictions": db.query(PredictionLog).count(),
        "total_signals": db.query(SignalLog).count(),
    }
