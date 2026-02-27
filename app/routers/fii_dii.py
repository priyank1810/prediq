from fastapi import APIRouter, Query
from app.services.fii_dii_service import fii_dii_service

router = APIRouter()


@router.get("/daily")
def get_fii_dii_daily():
    """Get today's FII/DII activity."""
    return fii_dii_service.get_daily()


@router.get("/history")
def get_fii_dii_history(days: int = Query(30, ge=1, le=365)):
    """Get FII/DII activity history."""
    return fii_dii_service.get_history(days)
