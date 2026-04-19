"""Multi-Timeframe Signal Dashboard API.

Returns technical signals across 1h, 4h, and 1D timeframes
for a given symbol, with per-timeframe indicators and an overall consensus.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.services.data_fetcher import data_fetcher
from app.services.indicator_service import indicator_service
from app.utils.cache import cache
from app.config import SIGNAL_DIRECTION_THRESHOLD, CACHE_TTL_MTF_1H, CACHE_TTL_MTF_DAILY

logger = logging.getLogger(__name__)

router = APIRouter()

# Cache TTLs per timeframe (seconds)
_CACHE_TTL = {
    "1h": CACHE_TTL_MTF_1H,
    "4h": 300,
    "1D": CACHE_TTL_MTF_DAILY,
}


def _direction_from_score(score: float) -> str:
    if score > SIGNAL_DIRECTION_THRESHOLD:
        return "BUY"
    elif score < -SIGNAL_DIRECTION_THRESHOLD:
        return "SELL"
    return "NEUTRAL"


def _strength_from_score(score: float) -> int:
    """Convert a -100..+100 score to a 0..100 strength value."""
    return min(100, max(0, int(abs(score))))


def _extract_indicators(details: dict) -> dict:
    """Extract the key indicators the dashboard needs from indicator details."""
    rsi = details.get("rsi")
    macd_hist = details.get("macd_histogram", 0)
    macd_line = details.get("macd_line", 0)
    macd_signal = details.get("macd_signal_line", 0)

    if macd_hist is not None and macd_hist > 0:
        macd_direction = "BULLISH"
    elif macd_hist is not None and macd_hist < 0:
        macd_direction = "BEARISH"
    else:
        macd_direction = "NEUTRAL"

    # SMA trend: compare current price to SMA-20 equivalent (ema21 as proxy)
    current_price = details.get("current_price", 0)
    ema21 = details.get("ema21", 0)
    if current_price and ema21:
        sma_trend = "ABOVE" if current_price > ema21 else "BELOW"
    else:
        sma_trend = "N/A"

    return {
        "rsi": round(rsi, 1) if rsi is not None else None,
        "macd_direction": macd_direction,
        "macd_histogram": round(macd_hist, 4) if macd_hist is not None else None,
        "macd_line": round(macd_line, 4) if macd_line is not None else None,
        "macd_signal": round(macd_signal, 4) if macd_signal is not None else None,
        "sma_trend": sma_trend,
        "ema9": details.get("ema9"),
        "ema21": ema21,
        "adx": details.get("adx"),
        "bb_position": details.get("bb_position"),
    }


def _resample_intraday(df: pd.DataFrame, rule: str) -> pd.DataFrame | None:
    """Resample a 15-min intraday DataFrame to a coarser timeframe."""
    if df is None or len(df) < 20:
        return None
    try:
        df_copy = df.copy()
        if "datetime_str" not in df_copy.columns:
            return None
        df_copy["_dt"] = pd.to_datetime(df_copy["datetime_str"], format="%Y-%m-%d %H:%M")
        df_copy = df_copy.set_index("_dt")
        resampled = df_copy.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        if "datetime_str" not in resampled.columns:
            resampled["datetime_str"] = resampled.index.strftime("%Y-%m-%d %H:%M")
        resampled = resampled.reset_index(drop=True)
        return resampled if len(resampled) >= 14 else None
    except Exception as e:
        logger.warning(f"Resample to {rule} failed: {e}")
        return None


def _compute_tf(symbol: str, label: str, df: pd.DataFrame) -> dict:
    """Compute signal data for a single timeframe DataFrame."""
    cache_key = f"mtf_dashboard:{symbol}:{label}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    if df is None or len(df) < 14:
        return {
            "timeframe": label,
            "signal": "N/A",
            "strength": 0,
            "score": 0,
            "indicators": {
                "rsi": None,
                "macd_direction": "N/A",
                "macd_histogram": None,
                "macd_line": None,
                "macd_signal": None,
                "sma_trend": "N/A",
                "ema9": None,
                "ema21": None,
                "adx": None,
                "bb_position": None,
            },
        }

    try:
        result = indicator_service.compute_intraday_indicators(df)
        score = result["score"]
        details = result.get("details", {})

        tf_data = {
            "timeframe": label,
            "signal": _direction_from_score(score),
            "strength": _strength_from_score(score),
            "score": round(score, 1),
            "indicators": _extract_indicators(details),
        }

        ttl = _CACHE_TTL.get(label, 120)
        cache.set(cache_key, tf_data, ttl)
        return tf_data
    except Exception as e:
        logger.warning(f"MTF dashboard {label} compute failed for {symbol}: {e}")
        return {
            "timeframe": label,
            "signal": "N/A",
            "strength": 0,
            "score": 0,
            "indicators": {
                "rsi": None,
                "macd_direction": "N/A",
                "macd_histogram": None,
                "macd_line": None,
                "macd_signal": None,
                "sma_trend": "N/A",
                "ema9": None,
                "ema21": None,
                "adx": None,
                "bb_position": None,
            },
        }


def _compute_consensus(timeframes: list[dict]) -> dict:
    """Compute majority consensus across all available timeframes."""
    valid = [tf for tf in timeframes if tf["signal"] in ("BUY", "SELL", "NEUTRAL")]
    if not valid:
        return {"signal": "N/A", "strength": 0, "agreement": 0, "total": 0}

    buy_count = sum(1 for tf in valid if tf["signal"] == "BUY")
    sell_count = sum(1 for tf in valid if tf["signal"] == "SELL")
    neutral_count = sum(1 for tf in valid if tf["signal"] == "NEUTRAL")
    total = len(valid)

    if buy_count > sell_count and buy_count > neutral_count:
        consensus_signal = "BUY"
        agreement = buy_count
    elif sell_count > buy_count and sell_count > neutral_count:
        consensus_signal = "SELL"
        agreement = sell_count
    else:
        consensus_signal = "NEUTRAL"
        agreement = neutral_count

    # Average strength of agreeing timeframes
    agreeing = [tf for tf in valid if tf["signal"] == consensus_signal]
    avg_strength = int(sum(tf["strength"] for tf in agreeing) / len(agreeing)) if agreeing else 0

    return {
        "signal": consensus_signal,
        "strength": avg_strength,
        "agreement": agreement,
        "total": total,
    }


@router.get("/{symbol}")
async def get_mtf_dashboard(symbol: str):
    """Return signals across 1h, 4h, and 1D timeframes for a symbol."""
    sym = symbol.upper()

    # Check top-level cache first
    top_cache_key = f"mtf_dashboard_full:{sym}"
    cached_full = cache.get(top_cache_key)
    if cached_full:
        return cached_full

    try:
        # Fetch intraday (5d of 15m candles) and daily data in parallel
        intraday_df, daily_df = await asyncio.gather(
            asyncio.to_thread(data_fetcher.get_intraday_data, sym, "5d", "15m"),
            asyncio.to_thread(data_fetcher.get_historical_data, sym, period="3mo"),
        )

        # Prepare dataframes for each timeframe
        # 1h: resample 15m -> 1h
        df_1h = await asyncio.to_thread(_resample_intraday, intraday_df, "1h")

        # 4h: resample 15m -> 4h
        df_4h = await asyncio.to_thread(_resample_intraday, intraday_df, "4h")

        # 1D: use daily historical data, ensure datetime_str column exists
        df_1d = None
        if daily_df is not None and not daily_df.empty:
            df_1d = daily_df.copy()
            if "datetime_str" not in df_1d.columns and "date" in df_1d.columns:
                df_1d["datetime_str"] = df_1d["date"].astype(str)

        # Compute each timeframe (can be parallelised but they're fast individually)
        tf_1h = await asyncio.to_thread(_compute_tf, sym, "1h", df_1h)
        tf_4h = await asyncio.to_thread(_compute_tf, sym, "4h", df_4h)
        tf_1d = await asyncio.to_thread(_compute_tf, sym, "1D", df_1d)

        timeframes = [tf_1h, tf_4h, tf_1d]
        consensus = _compute_consensus(timeframes)

        # Current price
        current_price = 0
        if intraday_df is not None and not intraday_df.empty:
            current_price = round(float(intraday_df["close"].iloc[-1]), 2)
        elif daily_df is not None and not daily_df.empty:
            current_price = round(float(daily_df["close"].iloc[-1]), 2)

        response = {
            "symbol": sym,
            "current_price": current_price,
            "timeframes": timeframes,
            "consensus": consensus,
        }

        cache.set(top_cache_key, response, 90)  # cache full response 90s
        return response

    except Exception as e:
        logger.error(f"MTF dashboard failed for {sym}: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-timeframe dashboard failed: {str(e)}")
