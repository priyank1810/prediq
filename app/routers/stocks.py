from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator
from app.services.data_fetcher import data_fetcher
from app.services.market_movers import market_movers_service
from app.utils.helpers import market_status, validate_symbol

router = APIRouter()


class BulkQuoteRequest(BaseModel):
    symbols: list[str]

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v):
        validated = []
        for s in v[:40]:
            try:
                validated.append(validate_symbol(s))
            except ValueError:
                continue  # Skip invalid symbols silently
        return validated


@router.get("/search")
def search_stocks(q: str = Query("", min_length=0, max_length=30)):
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


@router.get("/market-movers")
def get_market_movers(count: int = Query(10, ge=1, le=50)):
    """Return top gainers and losers from the broader market."""
    return market_movers_service.get_market_movers(count)


@router.post("/quotes/bulk")
def get_bulk_quotes(req: BulkQuoteRequest):
    return data_fetcher.get_bulk_quotes(req.symbols)


def _validated_symbol(symbol: str) -> str:
    try:
        return validate_symbol(symbol)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol!r}")


@router.get("/ai/search")
def ai_search(q: str = Query(..., min_length=2, description="Natural language search query")):
    """Natural language stock search. Examples: 'oversold banking stocks', 'top gainers', 'undervalued stocks'."""
    try:
        from app.services.nl_search_service import nl_search_service
        return nl_search_service.search(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")



@router.get("/{symbol}/fundamentals")
def get_fundamentals(symbol: str):
    """Get fundamental data, financials, and quarterly results for a stock."""
    sym = _validated_symbol(symbol)
    try:
        from app.services.fundamental_service import fundamental_service
        data = fundamental_service.get_fundamentals(sym)
        if not data or not data.get("symbol"):
            # Return empty structure for indices/ETFs that lack fundamentals
            return {"symbol": sym, "pe": None, "pb": None, "roe": None, "de": None,
                    "dividend_yield": None, "market_cap": None, "revenue": None,
                    "profit_margin": None, "operating_margin": None,
                    "earnings_quarterly": [], "income_quarterly": [],
                    "message": "Fundamental data not available for this symbol"}
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch fundamentals: {str(e)}")


@router.get("/{symbol}/news")
def get_stock_news(symbol: str):
    """Get latest news headlines with sentiment for a stock."""
    sym = _validated_symbol(symbol)
    try:
        from app.services.sentiment_service import sentiment_service
        data = sentiment_service.get_sentiment(sym)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {str(e)}")


@router.get("/{symbol}/quote")
def get_quote(symbol: str):
    sym = _validated_symbol(symbol)
    try:
        quote = data_fetcher.get_live_quote(sym)
        if not quote:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        return quote
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/earnings/upcoming")
async def upcoming_earnings(symbols: str = Query("", description="Comma-separated symbols")):
    """Get upcoming earnings dates for specified symbols."""
    import asyncio
    from app.services.earnings_service import earnings_service

    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        # Default to popular stocks
        sym_list = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                     "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC"]

    result = await asyncio.to_thread(earnings_service.get_earnings, sym_list[:20])
    return result


@router.get("/{symbol}/history")
def get_history(symbol: str, period: str = Query("1y")):
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
    sym = _validated_symbol(symbol)
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail=f"Invalid period. Use one of: {valid_periods}")
    try:

        # Short periods use intraday (15-min) candles for a meaningful chart
        if period in ("1d", "5d"):
            import pandas as pd
            df = data_fetcher.get_intraday_data(sym, period=period, interval="15m")
            # Fallback to daily data if intraday unavailable (weekends/holidays)
            if df is None or df.empty:
                df = data_fetcher.get_historical_data(sym, "1mo")
                if df is not None and not df.empty:
                    return df.to_dict(orient="records")
                raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
            # Convert datetime to Unix timestamp (seconds) for LightweightCharts
            dt_col = pd.to_datetime(df["datetime"])
            # Convert to UTC first if timezone-aware, then add IST offset
            # so LightweightCharts (which displays UTC) shows IST face values
            if dt_col.dt.tz is not None:
                dt_col = dt_col.dt.tz_convert("UTC").dt.tz_localize(None)
            df["date"] = dt_col.astype("int64") // 10**9
            df["date"] = df["date"] + 19800  # +5h30m IST offset for display
            # Sort by timestamp and drop duplicates (prevent chart errors)
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            # Filter out bad candles: >8% move from previous close is suspect
            if len(df) > 1 and "close" in df.columns:
                prev_close = df["close"].shift(1)
                pct_change = ((df["close"] - prev_close) / prev_close).abs()
                # Keep first row (NaN) and rows with <8% move
                valid = pct_change.isna() | (pct_change < 0.08)
                df = df[valid].reset_index(drop=True)

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
