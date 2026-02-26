"""Market Movers service — Top Gainers & Losers from the broader NSE market.

Uses a 3-tier fallback:
  1. NSE NIFTY 200 index constituents (broadest coverage)
  2. NSE NIFTY 50 index constituents (narrower but still NSE-direct)
  3. yfinance bulk quotes for NIFTY 50 symbols (works everywhere)
"""

import logging
import time
from datetime import datetime

from app.config import CACHE_TTL_MARKET_MOVERS, NIFTY_50_SYMBOLS
from app.utils.cache import cache

logger = logging.getLogger(__name__)

NSE_BASE = "https://www.nseindia.com"
NSE_INDEX_API = f"{NSE_BASE}/api/equity-stockIndices"


class MarketMoversService:
    def __init__(self):
        self._session = None
        self._session_time = 0
        self._session_ttl = 240

    # ------------------------------------------------------------------
    # Session management (curl_cffi with Chrome TLS fingerprint)
    # ------------------------------------------------------------------

    def _create_session(self):
        """Create a fresh curl_cffi session and warm it up on NSE."""
        try:
            from curl_cffi import requests as cffi_requests
            session = cffi_requests.Session(impersonate="chrome110")
            resp = session.get(NSE_BASE, timeout=15)
            if resp.status_code == 200:
                self._session = session
                self._session_time = time.time()
                logger.debug("MarketMovers: NSE session established")
                return True
        except ImportError:
            logger.warning("MarketMovers: curl_cffi not installed")
        except Exception as e:
            logger.warning(f"MarketMovers: session init failed: {e}")
        return False

    def _ensure_session(self):
        now = time.time()
        if self._session and (now - self._session_time) < self._session_ttl:
            return True
        return self._create_session()

    # ------------------------------------------------------------------
    # NSE fetchers
    # ------------------------------------------------------------------

    def _fetch_nse_index(self, index_name: str) -> list[dict]:
        """Fetch constituents of an NSE index, returning a list of stock dicts."""
        if not self._ensure_session():
            return []

        try:
            resp = self._session.get(
                NSE_INDEX_API,
                params={"index": index_name},
                timeout=15,
            )

            if resp.status_code in (401, 403):
                # Session expired — retry once
                self._session_time = 0
                if not self._ensure_session():
                    return []
                resp = self._session.get(
                    NSE_INDEX_API,
                    params={"index": index_name},
                    timeout=15,
                )

            if resp.status_code != 200:
                logger.warning(f"MarketMovers: NSE returned {resp.status_code} for {index_name}")
                return []

            data = resp.json()
            stocks = []
            for item in data.get("data", []):
                symbol = item.get("symbol", "")
                pct = item.get("pChange")
                ltp = item.get("lastPrice")
                if symbol and pct is not None and ltp is not None:
                    stocks.append({
                        "symbol": symbol,
                        "ltp": float(ltp) if isinstance(ltp, (int, float)) else float(str(ltp).replace(",", "")),
                        "change": float(item.get("change", 0) or 0),
                        "pct_change": float(pct),
                        "open": float(item.get("open", 0) or 0),
                        "high": float(item.get("dayHigh", 0) or 0),
                        "low": float(item.get("dayLow", 0) or 0),
                        "prev_close": float(item.get("previousClose", 0) or 0),
                    })
            return stocks

        except Exception as e:
            logger.error(f"MarketMovers: NSE fetch failed for {index_name}: {e}")
            return []

    # ------------------------------------------------------------------
    # yfinance fallback
    # ------------------------------------------------------------------

    def _fetch_yfinance_fallback(self) -> list[dict]:
        """Use yfinance via existing data_fetcher for NIFTY 50 symbols."""
        try:
            from app.services.data_fetcher import data_fetcher
            quotes = data_fetcher.get_bulk_quotes(list(NIFTY_50_SYMBOLS))
            # Filter out empty/zero quotes
            return [q for q in quotes if q.get("ltp") and q["ltp"] > 0 and q.get("pct_change") is not None]
        except Exception as e:
            logger.error(f"MarketMovers: yfinance fallback failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_market_movers(self, count: int = 10) -> dict:
        """Return top gainers and losers with 3-tier fallback.

        Returns:
            {gainers: [...], losers: [...], source: str, total_stocks: int, timestamp: str}
        """
        cache_key = f"market_movers:{count}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        stocks = []
        source = ""

        # Tier 1: NIFTY 200
        stocks = self._fetch_nse_index("NIFTY 200")
        if len(stocks) >= 50:
            source = "NSE NIFTY 200"
        else:
            # Tier 2: NIFTY 50 (NSE direct)
            logger.info("MarketMovers: NIFTY 200 unavailable, trying NIFTY 50")
            stocks = self._fetch_nse_index("NIFTY 50")
            if len(stocks) >= 20:
                source = "NSE NIFTY 50"
            else:
                # Tier 3: yfinance
                logger.info("MarketMovers: NSE unavailable, falling back to yfinance")
                stocks = self._fetch_yfinance_fallback()
                source = "yfinance (NIFTY 50)"

        if not stocks:
            return {
                "gainers": [],
                "losers": [],
                "source": "unavailable",
                "total_stocks": 0,
                "timestamp": datetime.now().isoformat(),
            }

        # Sort by pct_change descending for gainers, ascending for losers
        sorted_stocks = sorted(stocks, key=lambda s: s.get("pct_change", 0), reverse=True)

        gainers = sorted_stocks[:count]
        losers = sorted_stocks[-count:][::-1]  # worst performers, most negative first

        result = {
            "gainers": gainers,
            "losers": losers,
            "source": source,
            "total_stocks": len(stocks),
            "timestamp": datetime.now().isoformat(),
        }

        cache.set(cache_key, result, CACHE_TTL_MARKET_MOVERS)
        return result


market_movers_service = MarketMoversService()
