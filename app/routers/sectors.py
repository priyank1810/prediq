from fastapi import APIRouter
from app.services.sector_service import sector_service

router = APIRouter()


@router.get("/heatmap")
def get_sector_heatmap():
    """Get sector performance data for heat map visualization."""
    return sector_service.get_heatmap()
