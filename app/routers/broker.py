"""Broker order placement and management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_active_user
from app.database import get_db
from app.schemas import OrderCreate, OrderResponse
from app.services.broker_service import broker_service

router = APIRouter()


@router.post("/order", response_model=OrderResponse)
async def place_order(
    order_data: OrderCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    """Place a new order via broker or as paper trade."""
    try:
        order = broker_service.place_order(db, user.id, order_data.model_dump())
        return order
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.delete("/order/{order_id}", response_model=OrderResponse)
async def cancel_order(
    order_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    """Cancel a pending or placed order."""
    try:
        order = broker_service.cancel_order(db, user.id, order_id)
        return order
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/positions")
async def get_positions(user=Depends(get_current_active_user)):
    """Fetch live positions from broker."""
    positions = broker_service.get_positions(user.id)
    return {"positions": positions}


@router.get("/orders", response_model=list[OrderResponse])
async def get_order_book(
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    """Fetch today's order book."""
    orders = broker_service.get_order_book(db, user.id)
    return orders


@router.post("/sync")
async def sync_portfolio(
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    """Sync broker holdings to local portfolio."""
    result = broker_service.sync_portfolio(db, user.id)
    return result


@router.get("/status")
async def broker_status(user=Depends(get_current_active_user)):
    """Check broker connection status."""
    return broker_service.get_broker_status()


@router.get("/recent", response_model=list[OrderResponse])
async def recent_orders(
    limit: int = 5,
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    """Get recent orders."""
    orders = broker_service.get_recent_orders(db, user.id, limit=min(limit, 50))
    return orders
