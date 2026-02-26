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
from app.utils.helpers import market_status
from app.config import CACHE_TTL_OPTION_CHAIN

logger = logging.getLogger(__name__)

NSE_BASE_URL = "https://www.nseindia.com"
NSE_OPTION_CHAIN_EQUITY = f"{NSE_BASE_URL}/api/option-chain-equities"
NSE_OPTION_CHAIN_INDEX = f"{NSE_BASE_URL}/api/option-chain-indices"

# Symbols that use the index option chain endpoint
INDEX_OPTION_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}

# Max retry attempts per fetch
_MAX_FETCH_RETRIES = 2
# Cooldown after consecutive failures (seconds)
_FAILURE_COOLDOWN = 60
_FAILURE_THRESHOLD = 3


class OptionChainError(Exception):
    """Structured error for option chain failures."""

    def __init__(self, message: str, error_type: str = "no_data"):
        """
        Args:
            message: Human-readable error message.
            error_type: One of "blocked", "market_closed", "no_data".
        """
        super().__init__(message)
        self.error_type = error_type


class NSEOptionChainService:
    def __init__(self):
        self._session = None
        self._session_time = 0
        self._session_ttl = 240
        self._consecutive_failures = 0
        self._last_failure_time = 0

    def _ensure_session(self) -> bool:
        """Create a session with Chrome TLS fingerprint using curl_cffi.

        Returns True if a usable session exists, False otherwise.
        """
        now = time.time()

        # Cooldown: if we've failed too many times recently, back off
        if (self._consecutive_failures >= _FAILURE_THRESHOLD
                and (now - self._last_failure_time) < _FAILURE_COOLDOWN):
            logger.debug(
                f"NSE session in cooldown ({self._consecutive_failures} consecutive failures). "
                f"Retrying in {int(_FAILURE_COOLDOWN - (now - self._last_failure_time))}s"
            )
            return False

        if self._session and (now - self._session_time) < self._session_ttl:
            return True

        try:
            from curl_cffi import requests as cffi_requests
            session = cffi_requests.Session(impersonate="chrome110")

            # Warm up session by visiting NSE
            resp = session.get(NSE_BASE_URL, timeout=15)
            if resp.status_code == 200:
                self._session = session
                self._session_time = now
                logger.debug("NSE session established via curl_cffi")
                return True
            else:
                logger.warning(f"NSE homepage returned {resp.status_code}")
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
        return True

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
        """Fetch and parse option chain data.

        Raises OptionChainError with context-aware messages.
        """
        cache_key = f"option_chain:{symbol}:{expiry or 'nearest'}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        data = self._fetch_raw(symbol)
        if not data or not data.get("records", {}).get("data"):
            self._raise_contextual_error(symbol)

        result = self._parse_chain(data, symbol, expiry)
        cache.set(cache_key, result, CACHE_TTL_OPTION_CHAIN)
        return result

    def _raise_contextual_error(self, symbol: str):
        """Raise an OptionChainError with market-hours and failure context."""
        status = market_status()

        if self._consecutive_failures >= _FAILURE_THRESHOLD:
            raise OptionChainError(
                f"NSE is blocking option chain requests for {symbol}. "
                "This typically happens on cloud servers (e.g. Render, Railway). "
                "NSE restricts access from data-centre IPs. "
                "Try again later or access from a local machine.",
                error_type="blocked",
            )

        if status in ("post_market", "closed_weekend"):
            raise OptionChainError(
                f"Option chain data for {symbol} is not available right now. "
                "Indian markets are closed. NSE option chain data is most reliable "
                "during market hours (Mon-Fri, 9:15 AM - 3:30 PM IST).",
                error_type="market_closed",
            )

        if status == "pre_market":
            raise OptionChainError(
                f"Option chain data for {symbol} is not available yet. "
                "Markets haven't opened. Data will be available after 9:15 AM IST.",
                error_type="market_closed",
            )

        raise OptionChainError(
            f"Option chain data not available for {symbol}. "
            "NSE may be temporarily unreachable. Please try again in a minute.",
            error_type="no_data",
        )

    def _fetch_raw(self, symbol: str) -> dict:
        """Fetch raw option chain JSON from NSE with retry."""
        cache_key = f"nse_raw_chain:{symbol}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        if not self._ensure_session():
            return {}
        if not self._session:
            return {}

        api_symbol, is_index = self._normalize_symbol(symbol)
        url = NSE_OPTION_CHAIN_INDEX if is_index else NSE_OPTION_CHAIN_EQUITY

        for attempt in range(_MAX_FETCH_RETRIES):
            try:
                resp = self._session.get(
                    url,
                    params={"symbol": api_symbol},
                    timeout=15,
                )

                if resp.status_code in (401, 403):
                    logger.info(f"NSE returned {resp.status_code} for {symbol}, refreshing session (attempt {attempt + 1})")
                    self._session_time = 0
                    if not self._ensure_session():
                        continue
                    resp = self._session.get(
                        url,
                        params={"symbol": api_symbol},
                        timeout=15,
                    )

                if resp.status_code != 200:
                    logger.warning(f"NSE returned {resp.status_code} for {symbol} (attempt {attempt + 1})")
                    if attempt < _MAX_FETCH_RETRIES - 1:
                        time.sleep(1)
                    continue

                data = resp.json() if hasattr(resp, 'json') and callable(resp.json) else {}
                if isinstance(data, dict) and data.get("records", {}).get("data"):
                    cache.set(cache_key, data, CACHE_TTL_OPTION_CHAIN)
                    # Reset failure tracking on success
                    self._consecutive_failures = 0
                    return data

                logger.warning(f"NSE returned empty data for {symbol} (attempt {attempt + 1})")

            except Exception as e:
                logger.error(f"NSE option chain fetch failed for {symbol} (attempt {attempt + 1}): {e}")

            if attempt < _MAX_FETCH_RETRIES - 1:
                time.sleep(1)

        # All retries exhausted — track failure
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        logger.warning(
            f"NSE option chain: all {_MAX_FETCH_RETRIES} attempts failed for {symbol}. "
            f"Consecutive failures: {self._consecutive_failures}"
        )
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
