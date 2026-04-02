import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, List, Dict

from app.config import POPULAR_STOCKS
from app.services.data_fetcher import data_fetcher
from app.services.indicator_service import indicator_service
from app.utils.cache import cache

logger = logging.getLogger(__name__)

SCREENER_CACHE_TTL = 300  # 5 minutes


class ScreenerService:
    """Scan POPULAR_STOCKS against user-selected filter criteria."""

    def scan(self, filters: dict) -> List[dict]:
        """
        Run screener scan against POPULAR_STOCKS.

        filters may include:
          - rsi_oversold: bool  (RSI < 30)
          - rsi_overbought: bool  (RSI > 70)
          - macd_bullish: bool  (MACD bullish crossover)
          - macd_bearish: bool  (MACD bearish crossover)
          - above_sma_20: bool
          - above_sma_50: bool
          - above_sma_200: bool
          - below_sma_20: bool
          - below_sma_50: bool
          - below_sma_200: bool
          - volume_spike: bool  (volume > 2x average)
          - price_change_min: float  (min % change)
          - price_change_max: float  (max % change)

        Returns list of matching stocks with indicator values.
        """
        # Build a cache key from sorted filter items
        cache_key = f"screener:{_filters_key(filters)}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        results = []

        def _process_symbol(symbol: str) -> Optional[dict]:
            try:
                return self._evaluate_symbol(symbol, filters)
            except Exception as e:
                logger.debug(f"Screener: failed to evaluate {symbol}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_process_symbol, sym): sym for sym in POPULAR_STOCKS}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        # Sort by number of matched filters descending, then by absolute change
        results.sort(key=lambda r: (-len(r["matched_filters"]), -abs(r.get("change_pct", 0))))

        cache.set(cache_key, results, SCREENER_CACHE_TTL)
        return results

    def _evaluate_symbol(self, symbol: str, filters: dict) -> Optional[dict]:
        """Fetch data for a symbol and check against filters. Returns dict or None."""
        quote = data_fetcher.get_live_quote(symbol)
        if not quote or not quote.get("ltp"):
            return None

        # Fetch historical data for indicator computation
        hist_df = data_fetcher.get_historical_data(symbol, period="1y")
        if hist_df is None or hist_df.empty or len(hist_df) < 30:
            return None

        # Compute indicators
        indicators = indicator_service.compute_all(hist_df)

        ltp = float(quote.get("ltp", 0))
        change_pct = float(quote.get("pct_change", 0))
        volume = int(quote.get("volume", 0))
        avg_volume = int(quote.get("avg_volume", 0)) if quote.get("avg_volume") else 0

        # Extract latest indicator values
        rsi_values = indicators.get("rsi", {}).get("values", [])
        rsi = _last_valid(rsi_values)

        macd_line_values = indicators.get("macd_line", {}).get("values", [])
        macd_signal_values = indicators.get("macd_signal", {}).get("values", [])
        macd_line = _last_valid(macd_line_values)
        macd_signal = _last_valid(macd_signal_values)
        macd_prev = _last_valid(macd_line_values, offset=1)
        macd_signal_prev = _last_valid(macd_signal_values, offset=1)

        sma_20_values = indicators.get("sma_20", {}).get("values", [])
        sma_50_values = indicators.get("sma_50", {}).get("values", [])
        sma_20 = _last_valid(sma_20_values)
        sma_50 = _last_valid(sma_50_values)

        # Compute SMA 200 from historical close
        sma_200 = None
        if len(hist_df) >= 200:
            sma_200 = round(float(hist_df["close"].tail(200).mean()), 2)

        # Volume ratio
        volume_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else None

        # MACD crossover detection
        macd_crossover = "none"
        if (macd_prev is not None and macd_signal_prev is not None
                and macd_line is not None and macd_signal is not None):
            if macd_prev <= macd_signal_prev and macd_line > macd_signal:
                macd_crossover = "bullish"
            elif macd_prev >= macd_signal_prev and macd_line < macd_signal:
                macd_crossover = "bearish"

        # SMA status
        sma_status = {}
        if sma_20 is not None:
            sma_status["sma_20"] = "above" if ltp > sma_20 else "below"
        if sma_50 is not None:
            sma_status["sma_50"] = "above" if ltp > sma_50 else "below"
        if sma_200 is not None:
            sma_status["sma_200"] = "above" if ltp > sma_200 else "below"

        # Check filters
        matched = []

        if filters.get("rsi_oversold") and rsi is not None and rsi < 30:
            matched.append("RSI Oversold (<30)")
        if filters.get("rsi_overbought") and rsi is not None and rsi > 70:
            matched.append("RSI Overbought (>70)")
        if filters.get("macd_bullish") and macd_crossover == "bullish":
            matched.append("MACD Bullish Crossover")
        if filters.get("macd_bearish") and macd_crossover == "bearish":
            matched.append("MACD Bearish Crossover")
        if filters.get("above_sma_20") and sma_status.get("sma_20") == "above":
            matched.append("Above SMA 20")
        if filters.get("above_sma_50") and sma_status.get("sma_50") == "above":
            matched.append("Above SMA 50")
        if filters.get("above_sma_200") and sma_status.get("sma_200") == "above":
            matched.append("Above SMA 200")
        if filters.get("below_sma_20") and sma_status.get("sma_20") == "below":
            matched.append("Below SMA 20")
        if filters.get("below_sma_50") and sma_status.get("sma_50") == "below":
            matched.append("Below SMA 50")
        if filters.get("below_sma_200") and sma_status.get("sma_200") == "below":
            matched.append("Below SMA 200")
        if filters.get("volume_spike") and volume_ratio is not None and volume_ratio > 2.0:
            matched.append("Volume Spike (>2x)")

        price_min = filters.get("price_change_min")
        price_max = filters.get("price_change_max")
        if price_min is not None and change_pct >= price_min:
            if price_max is not None:
                if change_pct <= price_max:
                    matched.append(f"Price Change {price_min}% to {price_max}%")
            else:
                matched.append(f"Price Change >= {price_min}%")
        elif price_max is not None and change_pct <= price_max:
            matched.append(f"Price Change <= {price_max}%")

        # Only return if at least one filter matched
        if not matched:
            return None

        return {
            "symbol": symbol,
            "ltp": ltp,
            "change_pct": round(change_pct, 2),
            "rsi": rsi,
            "macd_signal": macd_crossover,
            "sma_status": sma_status,
            "volume_ratio": volume_ratio,
            "matched_filters": matched,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
        }


def _last_valid(values: list, offset: int = 0) -> Optional[float]:
    """Get the last non-None value from an indicator values list."""
    if not values:
        return None
    idx = len(values) - 1 - offset
    while idx >= 0:
        if values[idx] is not None:
            return values[idx]
        idx -= 1
    return None


def _filters_key(filters: dict) -> str:
    """Create a stable string key from filter dict."""
    items = sorted((k, v) for k, v in filters.items() if v is not None and v is not False)
    return "|".join(f"{k}={v}" for k, v in items)


screener_service = ScreenerService()
