"""Direct Yahoo Finance API access via curl_cffi.

Bypasses yfinance library which gets blocked on cloud IPs (Lightsail, Render, etc.).
Uses curl_cffi with Chrome impersonation to avoid Yahoo bot detection.
"""
import logging
import pandas as pd
from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
_TIMEOUT = 15


def yahoo_chart(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance chart API.

    Returns DataFrame with columns: date, open, high, low, close, volume.
    For intraday intervals the 'date' column contains full timestamps.
    Returns empty DataFrame on failure.
    """
    url = _CHART_URL.format(symbol=symbol)
    params = {"range": period, "interval": interval, "includePrePost": "false"}
    try:
        r = cffi_requests.get(url, params=params, impersonate="chrome", timeout=_TIMEOUT)
        data = r.json()
        result = data["chart"]["result"][0]
        ts = result["timestamp"]
        ohlcv = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "date": pd.to_datetime(ts, unit="s"),
            "open": ohlcv["open"],
            "high": ohlcv["high"],
            "low": ohlcv["low"],
            "close": ohlcv["close"],
            "volume": ohlcv["volume"],
        })
        df = df.dropna(subset=["close"])
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].round(2)
        df["volume"] = df["volume"].fillna(0).astype(int)
        return df
    except Exception as e:
        logger.debug(f"Yahoo chart API failed for {symbol}: {e}")
        return pd.DataFrame()


def yahoo_quote(symbol: str) -> dict | None:
    """Fetch latest quote from Yahoo chart API meta + last candle.

    Returns dict with: ltp, open, high, low, close (prev), volume, change, pct_change.
    Returns None on failure.
    """
    url = _CHART_URL.format(symbol=symbol)
    params = {"range": "5d", "interval": "1d", "includePrePost": "false"}
    try:
        r = cffi_requests.get(url, params=params, impersonate="chrome", timeout=_TIMEOUT)
        data = r.json()
        chart_result = data["chart"]["result"][0]
        meta = chart_result["meta"]
        ohlcv = chart_result["indicators"]["quote"][0]

        price = meta.get("regularMarketPrice", 0)
        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose", 0)

        # Last candle OHLCV
        open_price = ohlcv["open"][-1] if ohlcv.get("open") else 0
        high = ohlcv["high"][-1] if ohlcv.get("high") else 0
        low = ohlcv["low"][-1] if ohlcv.get("low") else 0
        volume = ohlcv["volume"][-1] if ohlcv.get("volume") else 0

        change = price - prev_close if price and prev_close else 0
        pct_change = (change / prev_close * 100) if prev_close else 0

        return {
            "ltp": round(price, 2) if price else 0,
            "open": round(open_price, 2) if open_price else 0,
            "high": round(high, 2) if high else 0,
            "low": round(low, 2) if low else 0,
            "close": round(prev_close, 2) if prev_close else 0,
            "volume": int(volume) if volume else 0,
            "change": round(change, 2),
            "pct_change": round(pct_change, 2),
        }
    except Exception as e:
        logger.debug(f"Yahoo quote API failed for {symbol}: {e}")
        return None


def yahoo_history(symbol: str, period: str = "5d") -> pd.DataFrame:
    """Convenience wrapper: daily OHLCV with standard column names (Close, Open, High, Low, Volume).

    Returns DataFrame indexed by date — matches the format that yf.Ticker.history() returns,
    so it's a drop-in replacement.
    """
    df = yahoo_chart(symbol, period=period, interval="1d")
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    df = df.set_index("Date")
    return df


def yahoo_fundamentals(symbol: str) -> dict | None:
    """Fetch fundamental data from Yahoo quoteSummary API.

    Returns dict with: pe, pb, roe, de, rev_growth, earn_growth, div_yield,
                       market_cap, promoter_holding, sector, industry.
    Values that are ratios (roe, rev_growth, etc.) are returned as raw decimals.
    Returns None on failure.
    """
    url = _SUMMARY_URL.format(symbol=symbol)
    params = {"modules": "defaultKeyStatistics,financialData,summaryProfile,summaryDetail"}
    try:
        r = cffi_requests.get(url, params=params, impersonate="chrome", timeout=_TIMEOUT)
        data = r.json()
        result = data["quoteSummary"]["result"][0]

        key_stats = result.get("defaultKeyStatistics", {})
        financial = result.get("financialData", {})
        profile = result.get("summaryProfile", {})
        detail = result.get("summaryDetail", {})

        def _raw(d, key):
            v = d.get(key, {})
            return v.get("raw") if isinstance(v, dict) else v

        return {
            "pe": _raw(detail, "trailingPE") or _raw(key_stats, "forwardPE"),
            "pb": _raw(key_stats, "priceToBook"),
            "roe": _raw(financial, "returnOnEquity"),
            "de": _raw(key_stats, "debtToEquity"),
            "rev_growth": _raw(financial, "revenueGrowth"),
            "earn_growth": _raw(financial, "earningsGrowth"),
            "div_yield": _raw(detail, "dividendYield"),
            "market_cap": _raw(detail, "marketCap"),
            "promoter_holding": _raw(key_stats, "heldPercentInsiders"),
            "regularMarketPrice": _raw(detail, "regularMarketPrice"),
            "sector": profile.get("sector", ""),
            "industry": profile.get("industry", ""),
        }
    except Exception as e:
        logger.debug(f"Yahoo fundamentals API failed for {symbol}: {e}")
        return None
