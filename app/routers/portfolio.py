from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import PortfolioHoldingCreate
from app.services.portfolio_service import portfolio_service
from app.auth import get_current_active_user
from app.models import User

router = APIRouter()


@router.get("")
def list_holdings(db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    return portfolio_service.get_holdings(db, user_id=user.id)


@router.get("/summary")
def get_summary(db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    return portfolio_service.get_summary(db, user_id=user.id)


@router.post("")
def add_holding(
    data: PortfolioHoldingCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    holding = portfolio_service.add_holding(db, data.model_dump(), user_id=user.id)
    return {"id": holding.id, "message": "Holding added successfully"}


@router.delete("/{holding_id}")
def delete_holding(
    holding_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    success = portfolio_service.delete_holding(db, holding_id, user_id=user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Holding not found")
    return {"message": "Holding deleted"}
