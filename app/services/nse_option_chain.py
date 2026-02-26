"""NSE Option Chain data fetcher.

Fetches option chain data from NSE India's public API using curl_cffi
for browser-like TLS fingerprinting. Falls back to Angel One SmartAPI
when available.

NSE's option chain endpoint has aggressive Akamai bot protection.
Data availability may depend on market hours.
"""

import logging
import time
from datetime import datetime
from app.utils.cache import cache
from app.config import CACHE_TTL_OPTION_CHAIN

logger = logging.getLogger(__name__)

NSE_BASE_URL = "https://www.nseindia.com"
NSE_OPTION_CHAIN_EQUITY = f"{NSE_BASE_URL}/api/option-chain-equities"
NSE_OPTION_CHAIN_INDEX = f"{NSE_BASE_URL}/api/option-chain-indices"

# Symbols that use the index option chain endpoint
INDEX_OPTION_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}


class NSEOptionChainService:
    def __init__(self):
        self._session = None
        self._session_time = 0
        self._session_ttl = 240

    def _ensure_session(self):
        """Create a session with Chrome TLS fingerprint using curl_cffi."""
        now = time.time()
        if self._session and (now - self._session_time) < self._session_ttl:
            return

        try:
            from curl_cffi import requests as cffi_requests
            session = cffi_requests.Session(impersonate="chrome110")

            # Warm up session by visiting NSE
            resp = session.get(NSE_BASE_URL, timeout=15)
            if resp.status_code == 200:
                self._session = session
                self._session_time = now
                logger.debug("NSE session established via curl_cffi")
                return
        except ImportError:
            logger.warning("curl_cffi not installed, falling back to requests")
        except Exception as e:
            logger.warning(f"curl_cffi session failed: {e}")

        # Fallback to standard requests
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        try:
            session.get(f"{NSE_BASE_URL}/option-chain", timeout=15)
        except Exception:
            pass
        self._session = session
        self._session_time = now

    def _normalize_symbol(self, symbol: str) -> tuple:
        """Normalize symbol and determine if it's an index.
        Returns: (api_symbol, is_index)
        """
        upper = symbol.upper().strip()
        symbol_map = {
            "NIFTY 50": "NIFTY",
            "NIFTY BANK": "BANKNIFTY",
            "NIFTY FINANCIAL": "FINNIFTY",
            "NIFTY MIDCAP 100": "MIDCPNIFTY",
            "NIFTY NEXT 50": "NIFTYNXT50",
        }
        api_symbol = symbol_map.get(upper, upper)
        is_index = api_symbol in INDEX_OPTION_SYMBOLS
        return api_symbol, is_index

    def get_expiry_dates(self, symbol: str) -> list:
        """Get available expiry dates for a symbol."""
        data = self._fetch_raw(symbol)
        if not data:
            return []
        return data.get("records", {}).get("expiryDates", [])

    def get_option_chain(self, symbol: str, expiry: str = None) -> dict:
        """Fetch and parse option chain data."""
        cache_key = f"option_chain:{symbol}:{expiry or 'nearest'}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        data = self._fetch_raw(symbol)
        if not data or not data.get("records", {}).get("data"):
            raise ValueError(
                f"Option chain data not available for {symbol}. "
                "NSE may restrict access outside market hours (9:15 AM - 3:30 PM IST)."
            )

        result = self._parse_chain(data, symbol, expiry)
        cache.set(cache_key, result, CACHE_TTL_OPTION_CHAIN)
        return result

    def _fetch_raw(self, symbol: str) -> dict:
        """Fetch raw option chain JSON from NSE."""
        cache_key = f"nse_raw_chain:{symbol}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        self._ensure_session()
        if not self._session:
            return {}

        api_symbol, is_index = self._normalize_symbol(symbol)
        url = NSE_OPTION_CHAIN_INDEX if is_index else NSE_OPTION_CHAIN_EQUITY

        try:
            resp = self._session.get(
                url,
                params={"symbol": api_symbol},
                timeout=15,
            )

            if resp.status_code in (401, 403):
                self._session_time = 0
                self._ensure_session()
                resp = self._session.get(
                    url,
                    params={"symbol": api_symbol},
                    timeout=15,
                )

            if resp.status_code != 200:
                logger.warning(f"NSE returned {resp.status_code} for {symbol}")
                return {}

            data = resp.json() if hasattr(resp, 'json') and callable(resp.json) else {}
            if isinstance(data, dict) and data.get("records", {}).get("data"):
                cache.set(cache_key, data, CACHE_TTL_OPTION_CHAIN)
            return data

        except Exception as e:
            logger.error(f"NSE option chain fetch failed for {symbol}: {e}")
            return {}

    def _parse_chain(self, raw: dict, symbol: str, expiry: str = None) -> dict:
        """Parse raw NSE API response into structured option chain data."""
        records = raw.get("records", {})
        all_data = records.get("data", [])
        expiry_dates = records.get("expiryDates", [])
        underlying_value = records.get("underlyingValue", 0)

        if not expiry and expiry_dates:
            expiry = expiry_dates[0]

        filtered = [d for d in all_data if d.get("expiryDate") == expiry]

        chain_rows = []
        total_ce_oi = 0
        total_pe_oi = 0
        strikes_for_max_pain = []

        for item in filtered:
            strike = item.get("strikePrice", 0)
            ce = item.get("CE", {})
            pe = item.get("PE", {})

            ce_oi = ce.get("openInterest", 0) or 0
            pe_oi = pe.get("openInterest", 0) or 0
            ce_oi_change = ce.get("changeinOpenInterest", 0) or 0
            pe_oi_change = pe.get("changeinOpenInterest", 0) or 0

            total_ce_oi += ce_oi
            total_pe_oi += pe_oi

            row = {
                "strike_price": strike,
                "ce_oi": ce_oi,
                "ce_oi_change": ce_oi_change,
                "ce_volume": ce.get("totalTradedVolume", 0) or 0,
                "ce_iv": round(ce.get("impliedVolatility", 0) or 0, 2),
                "ce_ltp": round(ce.get("lastPrice", 0) or 0, 2),
                "ce_bid": round(ce.get("bidprice", 0) or 0, 2),
                "ce_ask": round(ce.get("askPrice", 0) or 0, 2),
                "ce_change": round(ce.get("change", 0) or 0, 2),
                "pe_oi": pe_oi,
                "pe_oi_change": pe_oi_change,
                "pe_volume": pe.get("totalTradedVolume", 0) or 0,
                "pe_iv": round(pe.get("impliedVolatility", 0) or 0, 2),
                "pe_ltp": round(pe.get("lastPrice", 0) or 0, 2),
                "pe_bid": round(pe.get("bidprice", 0) or 0, 2),
                "pe_ask": round(pe.get("askPrice", 0) or 0, 2),
                "pe_change": round(pe.get("change", 0) or 0, 2),
            }
            chain_rows.append(row)

            if ce_oi > 0 or pe_oi > 0:
                strikes_for_max_pain.append((strike, ce_oi, pe_oi))

        chain_rows.sort(key=lambda r: r["strike_price"])

        pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
        max_pain = self._calc_max_pain(strikes_for_max_pain) if strikes_for_max_pain else 0

        atm_strike = 0
        if chain_rows and underlying_value > 0:
            atm_strike = min(chain_rows, key=lambda r: abs(r["strike_price"] - underlying_value))["strike_price"]

        return {
            "symbol": symbol.upper(),
            "expiry": expiry,
            "expiry_dates": expiry_dates,
            "spot_price": round(underlying_value, 2),
            "atm_strike": atm_strike,
            "data": chain_rows,
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi,
            "pcr": pcr,
            "max_pain": max_pain,
            "timestamp": datetime.now().isoformat(),
        }

    def _calc_max_pain(self, strikes: list) -> float:
        """Calculate max pain strike — price at which total premium loss is minimized."""
        if not strikes:
            return 0

        min_pain = float("inf")
        max_pain_strike = 0

        all_strikes = [s[0] for s in strikes]
        for candidate in all_strikes:
            total_pain = 0
            for strike, ce_oi, pe_oi in strikes:
                call_pain = ce_oi * max(0, strike - candidate)
                put_pain = pe_oi * max(0, candidate - strike)
                total_pain += call_pain + put_pain

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = candidate

        return max_pain_strike


nse_option_chain = NSEOptionChainService()
