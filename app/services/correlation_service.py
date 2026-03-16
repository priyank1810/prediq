"""Correlation analysis service for stocks and sectors."""

import logging
import numpy as np
import yfinance as yf
from typing import Optional
from app.utils.cache import cache
from app.utils.helpers import yfinance_symbol

logger = logging.getLogger(__name__)

CACHE_TTL_CORRELATION = 3600  # 1 hour

# NIFTY 50 universe for top-correlation lookups
NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
    "ASIANPAINT", "SUNPHARMA", "HCLTECH", "WIPRO", "ULTRACEMCO",
    "BAJAJFINSV", "ONGC", "NTPC", "TATAMOTORS", "POWERGRID",
    "M&M", "ADANIENT", "ADANIPORTS", "COALINDIA", "JSWSTEEL",
    "TATASTEEL", "TECHM", "HDFCLIFE", "SBILIFE", "INDUSINDBK",
    "NESTLEIND", "GRASIM", "DIVISLAB", "DRREDDY", "CIPLA",
    "EICHERMOT", "HEROMOTOCO", "BPCL", "TATACONSUM", "APOLLOHOSP",
    "BRITANNIA", "UPL", "HINDALCO", "BAJAJ-AUTO", "LTIM",
]

# Sector indices for sector correlation
SECTOR_INDICES = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL",
    "AUTO": "^CNXAUTO",
    "REALTY": "^CNXREALTY",
    "ENERGY": "^CNXENERGY",
}


def _fetch_close_prices(symbols: list[str], period: str = "6mo") -> Optional[dict]:
    """Fetch historical close prices for multiple symbols using yfinance.

    Returns dict mapping symbol -> list of close prices (aligned by date).
    """
    yf_symbols = []
    symbol_map = {}  # yf_symbol -> original symbol
    for s in symbols:
        yf_sym = yfinance_symbol(s) if not s.startswith("^") else s
        yf_symbols.append(yf_sym)
        symbol_map[yf_sym] = s

    try:
        data = yf.download(
            yf_symbols,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=True,
        )

        if data is None or data.empty:
            return None

        # Extract Close prices
        if len(yf_symbols) == 1:
            # Single symbol: data is a simple DataFrame
            close = data[["Close"]].copy()
            close.columns = [symbols[0]]
        else:
            # Multiple symbols: MultiIndex columns
            if "Close" in data.columns.get_level_values(0):
                close = data["Close"].copy()
            else:
                close = data.iloc[:, :len(yf_symbols)].copy()

            # Rename columns from yf symbols to original symbols
            rename_map = {}
            for col in close.columns:
                col_str = str(col)
                if col_str in symbol_map:
                    rename_map[col] = symbol_map[col_str]
            if rename_map:
                close = close.rename(columns=rename_map)

        # Drop rows where all values are NaN, then forward-fill remaining gaps
        close = close.dropna(how="all")
        close = close.ffill().bfill()

        # Need at least 20 data points for meaningful correlation
        if len(close) < 20:
            return None

        return close

    except Exception as e:
        logger.error("Failed to fetch close prices: %s", e)
        return None


def get_correlation_matrix(symbols: list[str], period: str = "6mo") -> Optional[dict]:
    """Compute Pearson correlation matrix for given symbols.

    Returns: {symbols: [...], matrix: [[...]]}
    """
    if not symbols or len(symbols) < 2:
        return None

    cache_key = f"corr_matrix:{','.join(sorted(symbols))}:{period}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    close = _fetch_close_prices(symbols, period)
    if close is None:
        return None

    # Compute daily returns
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return None

    # Only keep columns that are in our requested symbols and have data
    valid_symbols = [s for s in symbols if s in returns.columns and returns[s].notna().sum() > 20]
    if len(valid_symbols) < 2:
        return None

    returns = returns[valid_symbols]

    # Compute Pearson correlation
    corr = returns.corr().values

    # Replace NaN with 0
    corr = np.nan_to_num(corr, nan=0.0)

    result = {
        "symbols": valid_symbols,
        "matrix": [[round(float(corr[i][j]), 4) for j in range(len(valid_symbols))] for i in range(len(valid_symbols))],
    }

    cache.set(cache_key, result, CACHE_TTL_CORRELATION)
    return result


def get_sector_correlation(period: str = "6mo") -> Optional[dict]:
    """Compute correlation matrix for sector indices.

    Returns: {symbols: [...], matrix: [[...]]}
    """
    cache_key = f"corr_sector:{period}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    sector_names = list(SECTOR_INDICES.keys())
    yf_symbols = list(SECTOR_INDICES.values())

    close = _fetch_close_prices(yf_symbols, period)
    if close is None:
        return None

    # Rename columns from yf tickers to sector names
    rename_map = {v: k for k, v in SECTOR_INDICES.items()}
    # Handle both string and other column types
    final_rename = {}
    for col in close.columns:
        col_str = str(col)
        if col_str in rename_map:
            final_rename[col] = rename_map[col_str]
    if final_rename:
        close = close.rename(columns=final_rename)

    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return None

    valid_sectors = [s for s in sector_names if s in returns.columns and returns[s].notna().sum() > 20]
    if len(valid_sectors) < 2:
        return None

    returns = returns[valid_sectors]
    corr = returns.corr().values
    corr = np.nan_to_num(corr, nan=0.0)

    result = {
        "symbols": valid_sectors,
        "matrix": [[round(float(corr[i][j]), 4) for j in range(len(valid_sectors))] for i in range(len(valid_sectors))],
    }

    cache.set(cache_key, result, CACHE_TTL_CORRELATION)
    return result


def get_top_correlations(symbol: str, n: int = 10, period: str = "6mo") -> Optional[dict]:
    """Find the most and least correlated stocks from NIFTY50 universe.

    Returns: {symbol: str, period: str, most_correlated: [...], least_correlated: [...]}
    """
    cache_key = f"corr_top:{symbol}:{n}:{period}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # Build list: target symbol + NIFTY50 (excluding target itself)
    universe = [s for s in NIFTY50_SYMBOLS if s.upper() != symbol.upper()]
    all_symbols = [symbol.upper()] + universe

    close = _fetch_close_prices(all_symbols, period)
    if close is None:
        return None

    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return None

    target = symbol.upper()
    if target not in returns.columns:
        return None

    # Compute correlation of target with all others
    correlations = []
    for s in universe:
        if s in returns.columns and returns[s].notna().sum() > 20:
            corr_val = returns[target].corr(returns[s])
            if not np.isnan(corr_val):
                correlations.append({"symbol": s, "correlation": round(float(corr_val), 4)})

    if not correlations:
        return None

    # Sort by correlation (descending for most correlated)
    correlations.sort(key=lambda x: x["correlation"], reverse=True)

    half_n = max(n // 2, 1)
    most_correlated = correlations[:half_n]
    least_correlated = sorted(correlations, key=lambda x: x["correlation"])[:half_n]

    result = {
        "symbol": target,
        "period": period,
        "most_correlated": most_correlated,
        "least_correlated": least_correlated,
    }

    cache.set(cache_key, result, CACHE_TTL_CORRELATION)
    return result
