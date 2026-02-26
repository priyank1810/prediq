from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.services.data_fetcher import data_fetcher
from app.utils.helpers import market_status

router = APIRouter()


class BulkQuoteRequest(BaseModel):
    symbols: list[str]


@router.get("/search")
def search_stocks(q: str = Query("", min_length=0)):
    if not q:
        return data_fetcher.get_popular_stocks()
    return data_fetcher.search_stocks(q)


@router.get("/market-status")
def get_market_status():
    return {"status": market_status()}


@router.get("/data-source")
def get_data_source():
    """Returns which data provider is active."""
    try:
        from app.services.angel_provider import angel_provider
        if angel_provider.is_available:
            return {"source": "angel_one", "label": "Real-Time", "delay": 0}
    except Exception:
        pass
    return {"source": "yfinance", "label": "Delayed 15-20 min", "delay": 15}


@router.post("/quotes/bulk")
def get_bulk_quotes(req: BulkQuoteRequest):
    symbols = [s.upper() for s in req.symbols[:40]]
    return data_fetcher.get_bulk_quotes(symbols)


@router.get("/{symbol}/quote")
def get_quote(symbol: str):
    try:
        quote = data_fetcher.get_live_quote(symbol.upper())
        if not quote:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        return quote
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/{symbol}/history")
def get_history(symbol: str, period: str = Query("1y")):
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail=f"Invalid period. Use one of: {valid_periods}")
    try:
        sym = symbol.upper()

        # Short periods use intraday (15-min) candles for a meaningful chart
        if period in ("1d", "5d"):
            import pandas as pd
            df = data_fetcher.get_intraday_data(sym, period=period, interval="15m")
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail=f"No intraday data for {symbol}")
            # Convert datetime to Unix timestamp (seconds) for LightweightCharts
            df["date"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**9
            cols = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]
            return df.to_dict(orient="records")

        df = data_fetcher.get_historical_data(sym, period)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
