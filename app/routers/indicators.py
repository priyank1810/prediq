from fastapi import APIRouter, HTTPException
from app.services.data_fetcher import data_fetcher
from app.services.indicator_service import indicator_service

router = APIRouter()


@router.get("/{symbol}/patterns")
def get_patterns(symbol: str):
    """Detect chart patterns in historical data."""
    try:
        from app.ai.pattern_detector import pattern_detector

        df = data_fetcher.get_historical_data(symbol.upper(), period="6mo")
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        patterns = pattern_detector.detect(df)
        return {"symbol": symbol.upper(), "patterns": patterns}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}")
def get_indicators(symbol: str):
    try:
        df = data_fetcher.get_historical_data(symbol.upper(), period="1y")
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        indicators = indicator_service.compute_all(df)
        indicators["symbol"] = symbol.upper()
        return indicators
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
