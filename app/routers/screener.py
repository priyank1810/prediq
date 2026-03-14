from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter()


class ScreenerFilters(BaseModel):
    rsi_oversold: bool = False
    rsi_overbought: bool = False
    macd_bullish: bool = False
    macd_bearish: bool = False
    above_sma_20: bool = False
    above_sma_50: bool = False
    above_sma_200: bool = False
    below_sma_20: bool = False
    below_sma_50: bool = False
    below_sma_200: bool = False
    volume_spike: bool = False
    price_change_min: Optional[float] = Field(None, description="Minimum price change %")
    price_change_max: Optional[float] = Field(None, description="Maximum price change %")


@router.post("/scan")
def scan_stocks(filters: ScreenerFilters):
    from app.services.screener_service import screener_service

    filter_dict = filters.model_dump(exclude_none=True)

    # Remove False values so they don't count as active filters
    active_filters = {k: v for k, v in filter_dict.items() if v is not False}

    if not active_filters:
        return {"results": [], "count": 0, "message": "No filters selected"}

    results = screener_service.scan(active_filters)
    return {
        "results": results,
        "count": len(results),
        "scanned": 20,  # POPULAR_STOCKS count
    }
