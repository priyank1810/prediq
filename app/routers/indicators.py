from fastapi import APIRouter, HTTPException, Query
from app.services.data_fetcher import data_fetcher
from app.services.indicator_service import indicator_service
from app.utils.cache import cache

CACHE_TTL_INDICATORS = 300  # 5 minutes
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]

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
def get_indicators(symbol: str, period: str = Query("1y")):
    if period not in VALID_PERIODS:
        period = "1y"
    # Indicators need at least 1mo of daily data to compute SMAs
    # For short periods, use a minimum of 3mo for computation then trim
    compute_period = period if period not in ("1d", "5d") else "3mo"
    try:
        sym = symbol.upper()
        cache_key = f"indicators:{sym}:{period}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        df = data_fetcher.get_historical_data(sym, period=compute_period)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        indicators = indicator_service.compute_all(df)
        indicators["symbol"] = sym

        # Trim indicator data to match the requested period's date range
        if period != compute_period:
            indicators = _trim_indicators_to_period(indicators, period)

        cache.set(cache_key, indicators, CACHE_TTL_INDICATORS)
        return indicators
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _trim_indicators_to_period(indicators: dict, period: str) -> dict:
    """Trim indicator date/value arrays to roughly match the requested period."""
    from datetime import datetime, timedelta

    period_days = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    days = period_days.get(period, 365)
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    trimmed = {}
    for key, val in indicators.items():
        if isinstance(val, dict) and "dates" in val and "values" in val:
            dates = val["dates"]
            values = val["values"]
            # Find first index >= cutoff date
            start_idx = 0
            for i, d in enumerate(dates):
                if str(d) >= cutoff:
                    start_idx = i
                    break
            trimmed[key] = {"dates": dates[start_idx:], "values": values[start_idx:]}
        else:
            trimmed[key] = val
    return trimmed
