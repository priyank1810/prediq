"""IPO section API: upcoming list, detail, scorecard."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.ipo_service import ipo_service

router = APIRouter()


@router.get("/upcoming")
def upcoming(db: Session = Depends(get_db)):
    return ipo_service.get_upcoming(db)


@router.get("/scorecard")
def scorecard(db: Session = Depends(get_db)):
    return ipo_service.get_scorecard(db)


@router.get("/{key}")
def detail(key: str, db: Session = Depends(get_db)):
    out = ipo_service.get_detail(db, key)
    if not out:
        raise HTTPException(status_code=404, detail="IPO not found")
    return out
