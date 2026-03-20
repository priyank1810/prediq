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


@router.get("/ai/summary/{symbol}")
def ai_summary(symbol: str):
    """Get AI-generated plain-English summary for a stock's current signal."""
    sym = _validated_symbol(symbol)
    try:
        from app.services.signal_service import signal_service
        from app.services.fundamental_service import fundamental_service
        from app.services.ai_summary_service import ai_summary_service

        # Get signal data (use cached/last signal if market closed)
        from app.utils.helpers import is_market_open
        if is_market_open():
            signal_data = signal_service.get_signal(sym)
        else:
            from app.database import SessionLocal
            from app.models import SignalLog
            db = SessionLocal()
            try:
                last = db.query(SignalLog).filter(SignalLog.symbol == sym).order_by(SignalLog.created_at.desc()).first()
                if last:
                    signal_data = {
                        "direction": last.direction, "confidence": last.confidence,
                        "composite_score": last.composite_score,
                        "technical": {"score": last.technical_score, "details": {}},
                        "sentiment": {"score": last.sentiment_score, "headline_count": 0,
                                      "positive_count": 0, "negative_count": 0},
                        "global_market": {"score": last.global_score, "news_magnitude": 0},
                        "fundamental": {"score": 0}, "oi_analysis": {"available": False},
                        "mtf_confluence": {"level": "LOW"}, "support_resistance": {"levels": {}},
                        "stock_learning": {"available": False},
                    }
                else:
                    signal_data = None
            finally:
                db.close()

        if not signal_data:
            raise HTTPException(status_code=404, detail=f"No signal data for {sym}")

        # Get fundamentals
        fundamentals = None
        try:
            fundamentals = fundamental_service.get_fundamentals(sym)
        except Exception:
            pass

        summary = ai_summary_service.generate_signal_summary(signal_data, fundamentals)

        # Add earnings analysis
        if fundamentals:
            summary["earnings_analysis"] = ai_summary_service.generate_earnings_analysis(fundamentals)

        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@router.get("/ai/earnings/{symbol}")
def ai_earnings_analysis(symbol: str):
    """AI-generated earnings analysis for a stock."""
    sym = _validated_symbol(symbol)
    try:
        from app.services.fundamental_service import fundamental_service
        from app.services.ai_summary_service import ai_summary_service

        fundamentals = fundamental_service.get_fundamentals(sym)
        if not fundamentals or not fundamentals.get("symbol"):
            return {"available": False, "summary": "No earnings data available for this symbol."}

        return ai_summary_service.generate_earnings_analysis(fundamentals)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Earnings analysis failed: {str(e)}")


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
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail=f"No intraday data for {symbol}")
            # Convert datetime to Unix timestamp (seconds) for LightweightCharts
            dt_col = pd.to_datetime(df["datetime"])
            df["date"] = dt_col.astype("int64") // 10**9
            # LightweightCharts displays UTC; if timestamps are tz-aware (IST),
            # add IST offset so chart axis shows IST time instead of UTC
            if dt_col.dt.tz is not None:
                df["date"] = df["date"] + 19800  # +5h30m IST offset
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
