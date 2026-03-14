"""Options chain data service.

Fetches option chain data from NSE for equities and indices,
with caching and fallback to empty structured responses.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

NSE_BASE = "https://www.nseindia.com"
NSE_OC_EQUITIES = NSE_BASE + "/api/option-chain-equities?symbol={symbol}"
NSE_OC_INDICES = NSE_BASE + "/api/option-chain-indices?symbol={symbol}"
CACHE_TTL = 180  # 3 minutes

# Indices use a different endpoint on NSE
_INDEX_SYMBOLS = {"NIFTY", "NIFTY 50", "BANKNIFTY", "NIFTY BANK", "FINNIFTY", "MIDCPNIFTY"}


def _empty_chain_response(symbol: str) -> dict:
    """Return an empty but well-structured options chain response."""
    return {
        "symbol": symbol,
        "underlying": 0,
        "expiry_dates": [],
        "selected_expiry": None,
        "strikes": [],
        "calls": [],
        "puts": [],
        "totals": {
            "call_oi": 0,
            "put_oi": 0,
            "call_volume": 0,
            "put_volume": 0,
            "call_oi_change": 0,
            "put_oi_change": 0,
            "pcr": 0,
        },
        "available": False,
    }


class OptionsService:
    """Fetches and parses NSE option chain data."""

    def __init__(self):
        self._session = None
        self._session_time = 0
        self._chain_cache = {}  # type: dict[str, tuple[float, dict]]

    def _create_session(self):
        """Create a curl_cffi session with NSE cookies."""
        try:
            from curl_cffi import requests as cffi_requests
            session = cffi_requests.Session(impersonate="chrome110")
            resp = session.get(NSE_BASE, timeout=15)
            if resp.status_code == 200:
                self._session = session
                self._session_time = time.time()
                return True
        except ImportError:
            logger.warning("OptionsService: curl_cffi not installed")
        except Exception as e:
            logger.warning(f"OptionsService: session init failed: {e}")
        return False

    def _ensure_session(self):
        if self._session and (time.time() - self._session_time) < 240:
            return True
        return self._create_session()

    def get_chain(self, symbol: str, expiry: Optional[str] = None) -> dict:
        """Fetch full option chain for a symbol.

        Args:
            symbol: NSE symbol (e.g. RELIANCE, NIFTY).
            expiry: Optional expiry date string (DD-Mon-YYYY). If None, uses nearest.

        Returns:
            Structured dict with calls, puts, strikes, totals, etc.
        """
        cache_key = f"{symbol}:{expiry or 'nearest'}"
        if cache_key in self._chain_cache:
            ts, data = self._chain_cache[cache_key]
            if time.time() - ts < CACHE_TTL:
                return data

        if not self._ensure_session():
            return _empty_chain_response(symbol)

        try:
            sym_upper = symbol.upper().replace(" ", "")
            if sym_upper in {"NIFTY", "NIFTY50"}:
                url = NSE_OC_INDICES.format(symbol="NIFTY")
            elif sym_upper in {"BANKNIFTY", "NIFTYBANK"}:
                url = NSE_OC_INDICES.format(symbol="BANKNIFTY")
            elif sym_upper in _INDEX_SYMBOLS:
                url = NSE_OC_INDICES.format(symbol=sym_upper)
            else:
                url = NSE_OC_EQUITIES.format(symbol=symbol.upper())

            resp = self._session.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Options chain fetch failed for {symbol}: HTTP {resp.status_code}")
                return _empty_chain_response(symbol)

            raw = resp.json()
            result = self._parse_chain(symbol, raw, expiry)
            self._chain_cache[cache_key] = (time.time(), result)
            return result

        except Exception as e:
            logger.warning(f"Options chain failed for {symbol}: {e}")
            return _empty_chain_response(symbol)

    def _parse_chain(self, symbol: str, raw: dict, selected_expiry: Optional[str]) -> dict:
        """Parse NSE option chain JSON into frontend-friendly structure."""
        records = raw.get("records", {})
        filtered = raw.get("filtered", {})
        oc_data = records.get("data", [])
        underlying = records.get("underlyingValue", 0)
        expiry_dates = records.get("expiryDates", [])

        # Determine which expiry to use
        if selected_expiry and selected_expiry in expiry_dates:
            active_expiry = selected_expiry
        elif expiry_dates:
            active_expiry = expiry_dates[0]  # nearest expiry
        else:
            active_expiry = None

        # Filter rows by expiry
        rows = []
        for row in oc_data:
            if active_expiry and row.get("expiryDate") != active_expiry:
                continue
            rows.append(row)

        # Build calls, puts, strikes
        strikes = []
        calls = []
        puts = []
        total_call_oi = 0
        total_put_oi = 0
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi_change = 0
        total_put_oi_change = 0

        for row in rows:
            strike = row.get("strikePrice", 0)
            strikes.append(strike)

            ce = row.get("CE", {})
            pe = row.get("PE", {})

            call_entry = {
                "strike": strike,
                "ltp": ce.get("lastPrice", 0) or 0,
                "oi": ce.get("openInterest", 0) or 0,
                "oi_change": ce.get("changeinOpenInterest", 0) or 0,
                "volume": ce.get("totalTradedVolume", 0) or 0,
                "iv": ce.get("impliedVolatility", 0) or 0,
                "change": ce.get("change", 0) or 0,
                "pct_change": ce.get("pChange", 0) or 0,
                "bid": ce.get("bidprice", 0) or 0,
                "ask": ce.get("askprice", 0) or 0,
            }
            put_entry = {
                "strike": strike,
                "ltp": pe.get("lastPrice", 0) or 0,
                "oi": pe.get("openInterest", 0) or 0,
                "oi_change": pe.get("changeinOpenInterest", 0) or 0,
                "volume": pe.get("totalTradedVolume", 0) or 0,
                "iv": pe.get("impliedVolatility", 0) or 0,
                "change": pe.get("change", 0) or 0,
                "pct_change": pe.get("pChange", 0) or 0,
                "bid": pe.get("bidprice", 0) or 0,
                "ask": pe.get("askprice", 0) or 0,
            }

            calls.append(call_entry)
            puts.append(put_entry)

            total_call_oi += call_entry["oi"]
            total_put_oi += put_entry["oi"]
            total_call_volume += call_entry["volume"]
            total_put_volume += put_entry["volume"]
            total_call_oi_change += call_entry["oi_change"]
            total_put_oi_change += put_entry["oi_change"]

        pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0

        return {
            "symbol": symbol.upper(),
            "underlying": underlying,
            "expiry_dates": expiry_dates,
            "selected_expiry": active_expiry,
            "strikes": sorted(set(strikes)),
            "calls": calls,
            "puts": puts,
            "totals": {
                "call_oi": total_call_oi,
                "put_oi": total_put_oi,
                "call_volume": total_call_volume,
                "put_volume": total_put_volume,
                "call_oi_change": total_call_oi_change,
                "put_oi_change": total_put_oi_change,
                "pcr": pcr,
            },
            "available": len(calls) > 0,
        }

    def get_max_pain(self, symbol: str, expiry: Optional[str] = None) -> dict:
        """Calculate max pain from option chain data.

        Uses the O(n) prefix-sum algorithm from oi_service.py.
        """
        chain = self.get_chain(symbol, expiry)
        if not chain["available"]:
            return {
                "symbol": symbol.upper(),
                "max_pain": 0,
                "underlying": 0,
                "pcr": 0,
                "available": False,
            }

        # Build strikes_data as (strike, ce_oi, pe_oi) sorted ascending
        strike_map = {}  # type: dict[float, list[int]]
        for c in chain["calls"]:
            strike_map.setdefault(c["strike"], [0, 0])[0] = c["oi"]
        for p in chain["puts"]:
            strike_map.setdefault(p["strike"], [0, 0])[1] = p["oi"]

        strikes_data = sorted(
            [(s, ois[0], ois[1]) for s, ois in strike_map.items()],
            key=lambda x: x[0],
        )

        # O(n) max pain using prefix sums (same as oi_service.py)
        if not strikes_data:
            return {
                "symbol": symbol.upper(),
                "max_pain": chain["underlying"],
                "underlying": chain["underlying"],
                "pcr": chain["totals"]["pcr"],
                "available": True,
            }

        n = len(strikes_data)
        strikes = [sd[0] for sd in strikes_data]
        ce_ois = [sd[1] for sd in strikes_data]
        pe_ois = [sd[2] for sd in strikes_data]

        sum_pe_below = 0
        sum_pe_str_below = 0
        sum_ce_above = sum(ce_ois)
        sum_ce_str_above = sum(ce_ois[i] * strikes[i] for i in range(n))

        best_pain = -1
        best_strike = strikes[0]
        pain_by_strike = []

        for k in range(n):
            sum_ce_above -= ce_ois[k]
            sum_ce_str_above -= ce_ois[k] * strikes[k]

            pain = (strikes[k] * sum_pe_below - sum_pe_str_below +
                    sum_ce_str_above - strikes[k] * sum_ce_above)

            pain_by_strike.append({"strike": strikes[k], "pain": pain})

            if pain > best_pain:
                best_pain = pain
                best_strike = strikes[k]

            sum_pe_below += pe_ois[k]
            sum_pe_str_below += pe_ois[k] * strikes[k]

        return {
            "symbol": symbol.upper(),
            "max_pain": best_strike,
            "underlying": chain["underlying"],
            "pcr": chain["totals"]["pcr"],
            "selected_expiry": chain["selected_expiry"],
            "pain_by_strike": pain_by_strike,
            "available": True,
        }


options_service = OptionsService()
