import logging
from app.utils.cache import cache

logger = logging.getLogger(__name__)

CACHE_TTL_FUNDAMENTALS = 86400  # 24 hours (fundamentals change quarterly)


class FundamentalService:
    """Fetches fundamental data via yfinance .info and caches for 24 hours."""

    def get_fundamentals(self, symbol: str) -> dict:
        """Fetch fundamental ratios for a stock.

        Returns dict with: pe, pb, roe, de, rev_growth, earn_growth, div_yield,
                          market_cap, promoter_holding, sector
        """
        cache_key = f"fundamentals:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._fetch(symbol)
        cache.set(cache_key, data, CACHE_TTL_FUNDAMENTALS)
        return data

    def _fetch(self, symbol: str) -> dict:
        try:
            from app.utils.helpers import is_index, yfinance_symbol
            from app.utils.yahoo_api import yahoo_fundamentals

            # Indices don't have fundamental data (P/E, P/B, etc.)
            if is_index(symbol):
                return self._empty_result()

            ticker_symbol = yfinance_symbol(symbol)
            info = yahoo_fundamentals(ticker_symbol)

            if not info or not info.get("regularMarketPrice"):
                # Fallback to BSE
                ticker_symbol = yfinance_symbol(symbol, exchange="BSE")
                info = yahoo_fundamentals(ticker_symbol)

            if not info:
                return self._empty_result()

            result = {
                "pe": info.get("pe"),
                "pb": info.get("pb"),
                "roe": self._pct(info.get("roe")),
                "de": info.get("de"),
                "rev_growth": self._pct(info.get("rev_growth")),
                "earn_growth": self._pct(info.get("earn_growth")),
                "div_yield": self._pct(info.get("div_yield")),
                "market_cap": info.get("market_cap"),
                "promoter_holding": self._pct(info.get("promoter_holding")),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "symbol": symbol,
            }

            # Clean None values to 0 for numeric fields
            for key in ["pe", "pb", "roe", "de", "rev_growth", "earn_growth", "div_yield"]:
                if result[key] is None:
                    result[key] = 0

            # D/E from Yahoo is sometimes as percentage (e.g., 50 means 0.5)
            if result["de"] and result["de"] > 10:
                result["de"] = result["de"] / 100.0

            logger.info(f"Fetched fundamentals for {symbol}: PE={result['pe']}, PB={result['pb']}, ROE={result['roe']}")
            return result

        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
            return self._empty_result()

    @staticmethod
    def _pct(val):
        """Convert decimal ratio to percentage (e.g., 0.15 -> 15)."""
        if val is None:
            return None
        return round(float(val) * 100, 2)

    @staticmethod
    def _empty_result():
        return {
            "pe": 0, "pb": 0, "roe": 0, "de": 0,
            "rev_growth": 0, "earn_growth": 0, "div_yield": 0,
            "market_cap": 0, "promoter_holding": 0,
            "sector": "", "industry": "", "symbol": "",
        }


fundamental_service = FundamentalService()
