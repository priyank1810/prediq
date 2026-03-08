"""Direct Yahoo Finance API access via curl_cffi.

Bypasses yfinance library which gets blocked on cloud IPs (Lightsail, Render, etc.).
Uses curl_cffi with Chrome impersonation to avoid Yahoo bot detection.
"""
import logging
import re
import time
import threading
import pandas as pd
from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
_TIMEOUT = 15

# --- Crumb/session management for quoteSummary API ---
_crumb_lock = threading.Lock()
_crumb = None
_crumb_session = None
_crumb_expires = 0
_CRUMB_TTL = 1800  # 30 minutes


def _get_crumb_session():
    """Get a session with valid crumb for Yahoo quoteSummary API."""
    global _crumb, _crumb_session, _crumb_expires
    with _crumb_lock:
        if _crumb and _crumb_session and time.time() < _crumb_expires:
            return _crumb_session, _crumb
        try:
            session = cffi_requests.Session(impersonate="chrome")
            r = session.get("https://finance.yahoo.com/quote/AAPL/", timeout=_TIMEOUT)
            match = re.search(r'"crumb":"([^"]+)"', r.text)
            if match:
                _crumb = match.group(1).replace("\\u002F", "/")
                _crumb_session = session
                _crumb_expires = time.time() + _CRUMB_TTL
                logger.debug(f"Yahoo crumb refreshed")
                return _crumb_session, _crumb
        except Exception as e:
            logger.debug(f"Failed to get Yahoo crumb: {e}")
        return None, None


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


def yahoo_quote(symbol: str):
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


def yahoo_fundamentals(symbol: str) -> "dict | None":
    """Fetch fundamental data from Yahoo quoteSummary API.

    Returns dict with: pe, pb, roe, de, rev_growth, earn_growth, div_yield,
                       market_cap, promoter_holding, sector, industry,
                       plus income_statement, balance_sheet for financials.
    Values that are ratios (roe, rev_growth, etc.) are returned as raw decimals.
    Returns None on failure.
    """
    url = _SUMMARY_URL.format(symbol=symbol)
    modules = (
        "defaultKeyStatistics,financialData,summaryProfile,summaryDetail,"
        "incomeStatementHistory,incomeStatementHistoryQuarterly,"
        "balanceSheetHistory,balanceSheetHistoryQuarterly,"
        "earningsHistory,earnings"
    )
    try:
        session, crumb = _get_crumb_session()
        if session and crumb:
            params = {"modules": modules, "crumb": crumb}
            r = session.get(url, params=params, timeout=_TIMEOUT)
        else:
            params = {"modules": modules}
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

        def _fmt(d, key):
            v = d.get(key, {})
            return v.get("fmt") if isinstance(v, dict) else None

        # Parse income statements (annual)
        income_annual = []
        for stmt in result.get("incomeStatementHistory", {}).get("incomeStatementHistory", []):
            income_annual.append({
                "date": _fmt(stmt, "endDate") or "",
                "revenue": _raw(stmt, "totalRevenue"),
                "net_income": _raw(stmt, "netIncome"),
                "operating_income": _raw(stmt, "operatingIncome"),
                "gross_profit": _raw(stmt, "grossProfit"),
                "ebit": _raw(stmt, "ebit"),
            })

        # Parse income statements (quarterly)
        income_quarterly = []
        for stmt in result.get("incomeStatementHistoryQuarterly", {}).get("incomeStatementHistory", []):
            income_quarterly.append({
                "date": _fmt(stmt, "endDate") or "",
                "revenue": _raw(stmt, "totalRevenue"),
                "net_income": _raw(stmt, "netIncome"),
                "operating_income": _raw(stmt, "operatingIncome"),
                "gross_profit": _raw(stmt, "grossProfit"),
                "ebit": _raw(stmt, "ebit"),
            })

        # Parse balance sheet (annual)
        balance_annual = []
        for stmt in result.get("balanceSheetHistory", {}).get("balanceSheetStatements", []):
            balance_annual.append({
                "date": _fmt(stmt, "endDate") or "",
                "total_assets": _raw(stmt, "totalAssets"),
                "total_liabilities": _raw(stmt, "totalLiab"),
                "total_equity": _raw(stmt, "totalStockholderEquity"),
                "cash": _raw(stmt, "cash"),
                "total_debt": _raw(stmt, "longTermDebt"),
            })

        # Parse quarterly earnings
        earnings_quarterly = []
        for e in result.get("earningsHistory", {}).get("history", []):
            earnings_quarterly.append({
                "date": _fmt(e, "quarter") or "",
                "actual_eps": _raw(e, "epsActual"),
                "estimate_eps": _raw(e, "epsEstimate"),
                "surprise_pct": _raw(e, "surprisePercent"),
            })

        # Revenue/earnings chart data from earnings module
        earnings_chart = result.get("earnings", {})
        yearly_financials = []
        for item in earnings_chart.get("financialsChart", {}).get("yearly", []):
            yearly_financials.append({
                "year": item.get("date"),
                "revenue": _raw(item, "revenue"),
                "earnings": _raw(item, "earnings"),
            })

        return {
            "pe": _raw(detail, "trailingPE") or _raw(key_stats, "forwardPE"),
            "forward_pe": _raw(key_stats, "forwardPE"),
            "pb": _raw(key_stats, "priceToBook"),
            "roe": _raw(financial, "returnOnEquity"),
            "roa": _raw(financial, "returnOnAssets"),
            "de": _raw(key_stats, "debtToEquity"),
            "rev_growth": _raw(financial, "revenueGrowth"),
            "earn_growth": _raw(financial, "earningsGrowth"),
            "div_yield": _raw(detail, "dividendYield"),
            "market_cap": _raw(detail, "marketCap"),
            "promoter_holding": _raw(key_stats, "heldPercentInsiders"),
            "regularMarketPrice": _raw(detail, "regularMarketPrice"),
            "sector": profile.get("sector", ""),
            "industry": profile.get("industry", ""),
            "total_revenue": _raw(financial, "totalRevenue"),
            "revenue_per_share": _raw(financial, "revenuePerShare"),
            "profit_margins": _raw(financial, "profitMargins"),
            "operating_margins": _raw(financial, "operatingMargins"),
            "gross_margins": _raw(financial, "grossMargins"),
            "ebitda": _raw(financial, "ebitda"),
            "total_cash": _raw(financial, "totalCash"),
            "total_debt": _raw(financial, "totalDebt"),
            "current_ratio": _raw(financial, "currentRatio"),
            "free_cashflow": _raw(financial, "freeCashflow"),
            "operating_cashflow": _raw(financial, "operatingCashflow"),
            "earnings_growth": _raw(financial, "earningsGrowth"),
            "book_value": _raw(key_stats, "bookValue"),
            "eps_trailing": _raw(key_stats, "trailingEps"),
            "eps_forward": _raw(key_stats, "forwardEps"),
            "peg_ratio": _raw(key_stats, "pegRatio"),
            "beta": _raw(key_stats, "beta"),
            "52_week_high": _raw(detail, "fiftyTwoWeekHigh"),
            "52_week_low": _raw(detail, "fiftyTwoWeekLow"),
            "50_day_avg": _raw(detail, "fiftyDayAverage"),
            "200_day_avg": _raw(detail, "twoHundredDayAverage"),
            "income_annual": income_annual,
            "income_quarterly": income_quarterly,
            "balance_annual": balance_annual,
            "earnings_quarterly": earnings_quarterly,
            "yearly_financials": yearly_financials,
        }
    except Exception as e:
        logger.debug(f"Yahoo fundamentals API failed for {symbol}: {e}")
        return None
