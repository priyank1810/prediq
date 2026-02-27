"""Angel One SmartAPI data provider for real-time market data.

Uses Angel One SmartAPI as the primary data source for live quotes.
Falls back gracefully when credentials are not configured.

Setup:
    1. Sign up at https://smartapi.angelone.in
    2. Fill in .env with ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional

import pyotp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Angel One exchange codes
EXCHANGE_NSE = "NSE"
EXCHANGE_BSE = "BSE"

# SmartAPI market data modes
MODE_LTP = "LTP"
MODE_FULL = "FULL"

# Token cache: symbol -> {"token": str, "exchange": str, "expires": float}
_token_cache: dict[str, dict] = {}
_token_cache_lock = Lock()
TOKEN_CACHE_TTL = 86400  # 24 hours — tokens don't change often

# Angel One candle interval constants
INTERVAL_MAP = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
}

# Period string to number of days
PERIOD_TO_DAYS = {
    "1d": 1,
    "5d": 5,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}

# Map of well-known Angel One symbol tokens for major indices/stocks
# These avoid needing a searchScrip call for the most common symbols
KNOWN_TOKENS = {
    # Indices
    "NIFTY 50": {"token": "99926000", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty 50"},
    "NIFTY BANK": {"token": "99926009", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty Bank"},
    "SENSEX": {"token": "99919000", "exchange": EXCHANGE_BSE, "trading_symbol": "SENSEX"},
    "NIFTY IT": {"token": "99926004", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty IT"},
    "NIFTY FINANCIAL": {"token": "99926037", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty Fin Service"},
    "INDIA VIX": {"token": "99926043", "exchange": EXCHANGE_NSE, "trading_symbol": "India VIX"},
    "NIFTY NEXT 50": {"token": "99926001", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty Next 50"},
    "NIFTY MIDCAP 100": {"token": "99926014", "exchange": EXCHANGE_NSE, "trading_symbol": "NIFTY MID SELECT"},
    "NIFTY AUTO": {"token": "99926002", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty Auto"},
    "NIFTY PHARMA": {"token": "99926011", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty Pharma"},
    "NIFTY METAL": {"token": "99926019", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty Metal"},
    "NIFTY ENERGY": {"token": "99926008", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty Energy"},
    "NIFTY FMCG": {"token": "99926018", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty FMCG"},
    "NIFTY REALTY": {"token": "99926012", "exchange": EXCHANGE_NSE, "trading_symbol": "Nifty Realty"},
    # Top stocks (NSE)
    "RELIANCE": {"token": "2885", "exchange": EXCHANGE_NSE, "trading_symbol": "RELIANCE-EQ"},
    "TCS": {"token": "11536", "exchange": EXCHANGE_NSE, "trading_symbol": "TCS-EQ"},
    "HDFCBANK": {"token": "1333", "exchange": EXCHANGE_NSE, "trading_symbol": "HDFCBANK-EQ"},
    "INFY": {"token": "1594", "exchange": EXCHANGE_NSE, "trading_symbol": "INFY-EQ"},
    "ICICIBANK": {"token": "4963", "exchange": EXCHANGE_NSE, "trading_symbol": "ICICIBANK-EQ"},
    "SBIN": {"token": "3045", "exchange": EXCHANGE_NSE, "trading_symbol": "SBIN-EQ"},
    "BHARTIARTL": {"token": "10604", "exchange": EXCHANGE_NSE, "trading_symbol": "BHARTIARTL-EQ"},
    "ITC": {"token": "1660", "exchange": EXCHANGE_NSE, "trading_symbol": "ITC-EQ"},
    "KOTAKBANK": {"token": "1922", "exchange": EXCHANGE_NSE, "trading_symbol": "KOTAKBANK-EQ"},
    "LT": {"token": "11483", "exchange": EXCHANGE_NSE, "trading_symbol": "LT-EQ"},
    "HINDUNILVR": {"token": "1394", "exchange": EXCHANGE_NSE, "trading_symbol": "HINDUNILVR-EQ"},
    "AXISBANK": {"token": "5900", "exchange": EXCHANGE_NSE, "trading_symbol": "AXISBANK-EQ"},
    "BAJFINANCE": {"token": "317", "exchange": EXCHANGE_NSE, "trading_symbol": "BAJFINANCE-EQ"},
    "MARUTI": {"token": "10999", "exchange": EXCHANGE_NSE, "trading_symbol": "MARUTI-EQ"},
    "TITAN": {"token": "3506", "exchange": EXCHANGE_NSE, "trading_symbol": "TITAN-EQ"},
    "ASIANPAINT": {"token": "236", "exchange": EXCHANGE_NSE, "trading_symbol": "ASIANPAINT-EQ"},
    "SUNPHARMA": {"token": "3351", "exchange": EXCHANGE_NSE, "trading_symbol": "SUNPHARMA-EQ"},
    "HCLTECH": {"token": "7229", "exchange": EXCHANGE_NSE, "trading_symbol": "HCLTECH-EQ"},
    "WIPRO": {"token": "3787", "exchange": EXCHANGE_NSE, "trading_symbol": "WIPRO-EQ"},
    "ULTRACEMCO": {"token": "11532", "exchange": EXCHANGE_NSE, "trading_symbol": "ULTRACEMCO-EQ"},
}


class AngelOneProvider:
    """Real-time data provider using Angel One SmartAPI."""

    # Minimum seconds between API calls to avoid TooManyRequests (AB1004)
    _RATE_LIMIT_INTERVAL = 0.5  # ~2 calls/sec to stay well under Angel One limits
    _RATE_LIMIT_RETRY_WAIT = 3.0  # wait before retry on 429
    _MAX_RETRIES = 3

    def __init__(self):
        self._client = None
        self._session_data = None
        self._login_time = None
        self._login_lock = Lock()
        self._available = False
        self._last_api_call = 0.0
        self._rate_lock = Lock()

        # Read credentials from env
        self.api_key = os.getenv("ANGEL_API_KEY", "").strip()
        self.client_id = os.getenv("ANGEL_CLIENT_ID", "").strip()
        self.password = os.getenv("ANGEL_PASSWORD", "").strip()
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET", "").strip()

        if all([self.api_key, self.client_id, self.password, self.totp_secret]):
            self._available = True
            logger.info("Angel One credentials found — real-time data enabled")
        else:
            logger.info("Angel One credentials not configured — using yfinance fallback")

    def _throttle(self):
        """Enforce minimum interval between API calls."""
        with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_api_call
            if elapsed < self._RATE_LIMIT_INTERVAL:
                time.sleep(self._RATE_LIMIT_INTERVAL - elapsed)
            self._last_api_call = time.time()

    @property
    def is_available(self) -> bool:
        return self._available

    def _ensure_session(self) -> bool:
        """Login or refresh session if needed. Returns True if session is active."""
        if not self._available:
            return False

        with self._login_lock:
            # Session valid for ~6 hours, refresh after 5
            if (self._client and self._login_time
                    and (time.time() - self._login_time) < 18000):
                return True

            try:
                from SmartApi import SmartConnect

                client = SmartConnect(api_key=self.api_key)
                totp = pyotp.TOTP(self.totp_secret).now()

                session_data = client.generateSession(
                    clientCode=self.client_id,
                    password=self.password,
                    totp=totp,
                )

                if not session_data or session_data.get("status") is False:
                    error_msg = session_data.get("message", "Unknown error") if session_data else "No response"
                    logger.error(f"Angel One login failed: {error_msg}")
                    return False

                self._client = client
                self._session_data = session_data
                self._login_time = time.time()
                logger.info(f"Angel One session established for {self.client_id}")
                return True

            except Exception as e:
                logger.error(f"Angel One login error: {e}")
                return False

    def _lookup_token(self, symbol: str) -> dict | None:
        """Get symbol token info. Returns {"token": str, "exchange": str} or None."""
        upper = symbol.upper()

        # Check known tokens first
        if upper in KNOWN_TOKENS:
            return KNOWN_TOKENS[upper]

        # Check cache
        with _token_cache_lock:
            cached = _token_cache.get(upper)
            if cached and cached["expires"] > time.time():
                return cached

        # Search via API
        if not self._ensure_session():
            return None

        try:
            result = self._client.searchScrip(exchange=EXCHANGE_NSE, searchscrip=symbol)
            if result and result.get("data"):
                items = result["data"]
                # Prefer exact match or EQ series
                best = None
                for item in items:
                    ts = item.get("tradingsymbol", "")
                    if ts.upper() == f"{upper}-EQ" or ts.upper() == upper:
                        best = item
                        break
                if not best:
                    best = items[0]

                token_info = {
                    "token": best["symboltoken"],
                    "exchange": EXCHANGE_NSE,
                    "trading_symbol": best.get("tradingsymbol", symbol),
                    "expires": time.time() + TOKEN_CACHE_TTL,
                }
                with _token_cache_lock:
                    _token_cache[upper] = token_info
                return token_info
        except Exception as e:
            logger.warning(f"Token lookup failed for {symbol}: {e}")

        return None

    def get_live_quote(self, symbol: str) -> dict | None:
        """Fetch real-time quote from Angel One.

        Returns standardized quote dict compatible with data_fetcher format,
        or None if unavailable.
        """
        if not self._ensure_session():
            return None

        token_info = self._lookup_token(symbol)
        if not token_info:
            return None

        try:
            exchange = token_info["exchange"]
            token = token_info["token"]

            self._throttle()
            ltp_data = self._client.ltpData(
                exchange=exchange,
                tradingsymbol=token_info.get("trading_symbol", symbol),
                symboltoken=token,
            )

            if not ltp_data or not ltp_data.get("data"):
                return None

            data = ltp_data["data"]
            ltp = float(data.get("ltp", 0))
            open_price = float(data.get("open", 0))
            high = float(data.get("high", 0))
            low = float(data.get("low", 0))
            close = float(data.get("close", 0))  # prev close

            # ltpData typically returns rupees, but may return paise for
            # some endpoints. Index tokens (starting with '999') always
            # return actual points — never divide those.
            is_index = token.startswith("999")
            if not is_index and ltp > 100000:
                ltp /= 100
                open_price /= 100
                high /= 100
                low /= 100
                close /= 100

            change = round(ltp - close, 2) if close else 0
            pct_change = round((change / close) * 100, 2) if close else 0

            return {
                "symbol": symbol,
                "ltp": round(ltp, 2),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": int(data.get("volume", 0) or 0),
                "change": change,
                "pct_change": pct_change,
                "timestamp": datetime.now().isoformat(),
                "source": "angel_one",
            }

        except Exception as e:
            logger.warning(f"Angel One quote failed for {symbol}: {e}")
            return None

    def get_multiple_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Fetch quotes for multiple symbols. Returns {symbol: quote_dict}."""
        results = {}

        if not self._ensure_session():
            return results

        # Group by exchange
        nse_tokens = []
        bse_tokens = []

        for symbol in symbols:
            token_info = self._lookup_token(symbol)
            if token_info:
                entry = {"symbol": symbol, "token": token_info["token"]}
                if token_info["exchange"] == EXCHANGE_BSE:
                    bse_tokens.append(entry)
                else:
                    nse_tokens.append(entry)

        # Fetch in batches using market data API
        for exchange, tokens in [(EXCHANGE_NSE, nse_tokens), (EXCHANGE_BSE, bse_tokens)]:
            if not tokens:
                continue
            try:
                token_list = [t["token"] for t in tokens]
                symbol_map = {t["token"]: t["symbol"] for t in tokens}

                # SmartAPI accepts up to 50 tokens per request
                for i in range(0, len(token_list), 50):
                    batch = token_list[i:i + 50]
                    self._throttle()
                    response = self._client.getMarketData(
                        mode=MODE_FULL,
                        exchangeTokens={exchange: batch},
                    )

                    if response and response.get("data") and response["data"].get("fetched"):
                        for item in response["data"]["fetched"]:
                            token = item.get("symbolToken", "")
                            sym = symbol_map.get(token)
                            if not sym:
                                continue

                            # Index tokens (starting with '999') return values in
                            # actual points; stock tokens return paise (÷100).
                            is_index = token.startswith("999")
                            divisor = 1 if is_index else 100

                            ltp = float(item.get("ltp", 0)) / divisor
                            close = float(item.get("close", 0)) / divisor
                            change = round(ltp - close, 2) if close else 0
                            pct_change = round((change / close) * 100, 2) if close else 0

                            results[sym] = {
                                "symbol": sym,
                                "ltp": round(ltp, 2),
                                "open": round(float(item.get("open", 0)) / divisor, 2),
                                "high": round(float(item.get("high", 0)) / divisor, 2),
                                "low": round(float(item.get("low", 0)) / divisor, 2),
                                "close": round(close, 2),
                                "volume": int(item.get("tradeVolume", 0) or 0),
                                "change": change,
                                "pct_change": pct_change,
                                "timestamp": datetime.now().isoformat(),
                                "source": "angel_one",
                            }

            except Exception as e:
                logger.warning(f"Angel One batch quote failed for {exchange}: {e}")

        return results

    def get_historical_candles(self, symbol: str, period: str = "1y") -> list[dict] | None:
        """Fetch daily OHLCV candles from Angel One.

        Returns list of dicts {date, open, high, low, close, volume} or None.
        """
        if not self._ensure_session():
            return None

        token_info = self._lookup_token(symbol)
        if not token_info:
            return None

        days = PERIOD_TO_DAYS.get(period, 365)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        # Use current time if before market close, otherwise 15:30
        to_time = min(to_date, to_date.replace(hour=15, minute=30, second=0, microsecond=0))
        params = {
            "exchange": token_info["exchange"],
            "symboltoken": token_info["token"],
            "interval": "ONE_DAY",
            "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
            "todate": to_time.strftime("%Y-%m-%d %H:%M"),
        }

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                self._throttle()
                response = self._client.getCandleData(params)

                if response and response.get("errorcode") == "AB1004":
                    if attempt < self._MAX_RETRIES:
                        logger.debug(f"Rate limited on {symbol} historical, retry {attempt + 1}")
                        time.sleep(self._RATE_LIMIT_RETRY_WAIT * (attempt + 1))
                        continue
                    logger.warning(f"Angel One historical rate limited for {symbol} after retries")
                    return None

                if not response or response.get("status") is False or not response.get("data"):
                    logger.warning(f"Angel One historical empty for {symbol}: {response}")
                    return None

                candles = response["data"]
                result = []
                for c in candles:
                    result.append({
                        "date": c[0][:10],
                        "open": round(float(c[1]), 2),
                        "high": round(float(c[2]), 2),
                        "low": round(float(c[3]), 2),
                        "close": round(float(c[4]), 2),
                        "volume": int(c[5]),
                    })

                logger.info(f"Got {len(result)} daily candles for {symbol} from Angel One")
                return result

            except Exception as e:
                logger.warning(f"Angel One historical failed for {symbol}: {e}")
                return None

        return None

    def get_intraday_candles(self, symbol: str, interval: str = "15m", period: str = "5d") -> list[dict] | None:
        """Fetch intraday OHLCV candles from Angel One.

        Returns list of dicts {datetime, open, high, low, close, volume, datetime_str} or None.
        """
        if not self._ensure_session():
            return None

        token_info = self._lookup_token(symbol)
        if not token_info:
            return None

        angel_interval = INTERVAL_MAP.get(interval, "FIFTEEN_MINUTE")
        days = PERIOD_TO_DAYS.get(period, 5)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        # Use current time if before market close, otherwise 15:30
        to_time = min(to_date, to_date.replace(hour=15, minute=30, second=0, microsecond=0))
        params = {
            "exchange": token_info["exchange"],
            "symboltoken": token_info["token"],
            "interval": angel_interval,
            "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
            "todate": to_time.strftime("%Y-%m-%d %H:%M"),
        }

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                self._throttle()
                response = self._client.getCandleData(params)

                if response and response.get("errorcode") == "AB1004":
                    if attempt < self._MAX_RETRIES:
                        logger.debug(f"Rate limited on {symbol} intraday, retry {attempt + 1}")
                        time.sleep(self._RATE_LIMIT_RETRY_WAIT * (attempt + 1))
                        continue
                    logger.warning(f"Angel One intraday rate limited for {symbol} after retries")
                    return None

                if not response or response.get("status") is False or not response.get("data"):
                    logger.warning(f"Angel One intraday empty for {symbol}: {response}")
                    return None

                candles = response["data"]
                result = []
                for c in candles:
                    ts_str = c[0]
                    dt = datetime.fromisoformat(ts_str)
                    result.append({
                        "datetime": dt,
                        "open": round(float(c[1]), 2),
                        "high": round(float(c[2]), 2),
                        "low": round(float(c[3]), 2),
                        "close": round(float(c[4]), 2),
                        "volume": int(c[5]),
                        "datetime_str": dt.strftime("%Y-%m-%d %H:%M"),
                    })

                logger.info(f"Got {len(result)} intraday ({interval}) candles for {symbol} from Angel One")
                return result

            except Exception as e:
                logger.warning(f"Angel One intraday failed for {symbol}: {e}")
                return None

        return None

    def get_feed_token(self) -> str | None:
        """Get feed token for WebSocket connection."""
        if not self._ensure_session():
            return None
        if self._session_data and self._session_data.get("data"):
            return self._session_data["data"].get("feedToken")
        return None

    def get_auth_token(self) -> str | None:
        """Get JWT auth token for WebSocket."""
        if not self._ensure_session():
            return None
        if self._session_data and self._session_data.get("data"):
            return self._session_data["data"].get("jwtToken")
        return None

    def get_exchange_type(self, symbol: str) -> int:
        """Get Angel One exchange type code. NSE_CM=1, BSE_CM=3."""
        token_info = self._lookup_token(symbol)
        if token_info and token_info["exchange"] == EXCHANGE_BSE:
            return 3
        return 1  # NSE_CM

    def logout(self):
        """Logout and cleanup session."""
        if self._client:
            try:
                self._client.terminateSession(self.client_id)
            except Exception:
                pass
            self._client = None
            self._session_data = None
            self._login_time = None


# Singleton instance
angel_provider = AngelOneProvider()
