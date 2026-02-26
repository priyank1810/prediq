from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import PortfolioHoldingCreate
from app.services.portfolio_service import portfolio_service

router = APIRouter()


@router.get("")
def list_holdings(db: Session = Depends(get_db)):
    return portfolio_service.get_holdings(db)


@router.get("/summary")
def get_summary(db: Session = Depends(get_db)):
    return portfolio_service.get_summary(db)


@router.post("")
def add_holding(
    data: PortfolioHoldingCreate,
    db: Session = Depends(get_db),
):
    holding = portfolio_service.add_holding(db, data.model_dump())
    return {"id": holding.id, "message": "Holding added successfully"}


@router.delete("/{holding_id}")
def delete_holding(
    holding_id: int,
    db: Session = Depends(get_db),
):
    success = portfolio_service.delete_holding(db, holding_id)
    if not success:
        raise HTTPException(status_code=404, detail="Holding not found")
    return {"message": "Holding deleted"}
