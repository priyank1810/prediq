import io
import csv
import logging
import pandas as pd
from curl_cffi import requests as cffi_requests
from datetime import datetime, timedelta
from typing import Optional
from app.utils.cache import cache
from app.utils.helpers import yfinance_symbol, now_ist
from app.utils.yahoo_api import yahoo_chart, yahoo_quote
from app.utils.circuit_breaker import yahoo_breaker, angel_breaker, nse_breaker
from app.config import CACHE_TTL_QUOTE, CACHE_TTL_HISTORY, CACHE_TTL_STOCK_LIST, CACHE_TTL_INTRADAY, POPULAR_STOCKS, INDICES

NSE_EQUITY_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_ETF_CSV_URL = "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"

logger = logging.getLogger(__name__)

# Try importing NSE libraries (may not be available)
try:
    from nsetools import Nse
    nse_client = Nse()
    NSE_AVAILABLE = True
except Exception:
    nse_client = None
    NSE_AVAILABLE = False

# Try Angel One (real-time data)
try:
    from app.services.angel_provider import angel_provider
    ANGEL_AVAILABLE = angel_provider.is_available
except Exception:
    angel_provider = None
    ANGEL_AVAILABLE = False


class DataFetcher:
    def __init__(self):
        self._stock_codes = None

    def get_live_quote(self, symbol: str) -> dict:
        from app.utils.helpers import is_index
        cache_key = f"quote:{symbol}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        quote = None

        # 1. Try Angel One (real-time, no delay)
        if ANGEL_AVAILABLE and angel_breaker.allow_request():
            try:
                quote = angel_provider.get_live_quote(symbol)
                if quote:
                    angel_breaker.record_success()
                    logger.debug(f"Got real-time quote for {symbol} from Angel One")
            except Exception as e:
                angel_breaker.record_failure()
                logger.debug(f"Angel One failed for {symbol}: {e}")

        # 2. Try nsetools (doesn't support indices)
        if quote is None and not is_index(symbol) and NSE_AVAILABLE and nse_breaker.allow_request():
            try:
                quote = self._fetch_from_nsetools(symbol)
                nse_breaker.record_success()
            except Exception:
                nse_breaker.record_failure()

        # 3. Fallback to Yahoo Finance API (15-20 min delayed)
        if quote is None and yahoo_breaker.allow_request():
            try:
                quote = self._fetch_from_yfinance(symbol)
                yahoo_breaker.record_success()
            except Exception:
                yahoo_breaker.record_failure()

        # Enrich with avg_volume from cached historical data
        if quote and not quote.get("avg_volume"):
            try:
                hist_df = self.get_historical_data(symbol, period="1mo")
                if hist_df is not None and not hist_df.empty and "volume" in hist_df.columns:
                    avg_vol = float(hist_df["volume"].tail(10).mean())
                    if avg_vol > 0:
                        quote["avg_volume"] = int(avg_vol)
            except Exception:
                pass

        if quote:
            cache.set(cache_key, quote, CACHE_TTL_QUOTE)
        return quote

    def get_bulk_quotes(self, symbols: list[str], skip_cache: bool = False) -> list[dict]:
        """Fetch quotes for multiple symbols, using Angel One batch API when available.

        skip_cache=True forces live fetch for all symbols (used by trade validation).
        """
        results = {}

        # Try Angel One batch first (single API call for all symbols)
        if ANGEL_AVAILABLE and angel_breaker.allow_request():
            try:
                angel_quotes = angel_provider.get_multiple_quotes(symbols)
                for sym, quote in angel_quotes.items():
                    results[sym] = quote
                    cache.set(f"quote:{sym}", quote, CACHE_TTL_QUOTE)
                angel_breaker.record_success()
            except Exception as e:
                angel_breaker.record_failure()
                logger.debug(f"Angel One batch failed: {e}")

        # Check cache for missing symbols before fetching (skip when fresh data required)
        missing = [s for s in symbols if s not in results]
        if not skip_cache:
            for sym in list(missing):
                cached = cache.get(f"quote:{sym}")
                if cached:
                    results[sym] = cached
                    missing.remove(sym)

        # Fetch remaining concurrently with 8s timeout per stock
        if missing:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def _fetch_one(sym):
                try:
                    return sym, self.get_live_quote(sym)
                except Exception:
                    return sym, None
            with ThreadPoolExecutor(max_workers=min(len(missing), 5)) as pool:
                futures = {pool.submit(_fetch_one, s): s for s in missing}
                for fut in as_completed(futures, timeout=8):
                    try:
                        sym, quote = fut.result(timeout=1)
                        if quote:
                            results[sym] = quote
                    except Exception:
                        pass

        return [results.get(s, {"symbol": s, "ltp": 0, "change": 0, "pct_change": 0}) for s in symbols]

    def get_intraday_data(self, symbol: str, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
        cache_key = f"intraday:{symbol}:{period}:{interval}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # 1. Try Angel One (real-time, no delay)
        df = self._intraday_from_angel(symbol, period, interval)

        # 2. Fallback to Yahoo chart API (direct curl_cffi)
        if (df is None or df.empty) and yahoo_breaker.allow_request():
            yf_symbol = yfinance_symbol(symbol)
            df = yahoo_chart(yf_symbol, period=period, interval=interval)
            if df is not None and not df.empty:
                df = df.rename(columns={"date": "datetime"})
                df["datetime_str"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M")

        # 3. Fallback to Google Finance
        if df is None or df.empty:
            df = self._intraday_from_google(symbol, period, interval)

        if df is None:
            df = pd.DataFrame()

        # 3. Append live candles from Angel One when available
        try:
            from app.services.live_candle_builder import candle_builder
            if candle_builder.has_data(symbol):
                live_df = candle_builder.get_intraday_df(symbol, interval=interval)
                if not live_df.empty:
                    if df.empty:
                        df = live_df
                    else:
                        last_hist_time = pd.to_datetime(df["datetime"]).max()
                        live_df["datetime"] = pd.to_datetime(live_df["datetime"])
                        new_candles = live_df[live_df["datetime"] > last_hist_time]
                        if not new_candles.empty:
                            df = pd.concat([df, new_candles], ignore_index=True)
        except Exception as e:
            logger.debug(f"Live candle append failed for {symbol}: {e}")

        if not df.empty:
            cache.set(cache_key, df, CACHE_TTL_INTRADAY)
        return df

    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        cache_key = f"history:{symbol}:{period}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # 1. Try Angel One (real-time, no delay)
        df = self._historical_from_angel(symbol, period)

        # 2. Fallback to yfinance
        if (df is None or df.empty) and yahoo_breaker.allow_request():
            try:
                df = self._historical_from_yfinance(symbol, period)
                if df is not None and not df.empty:
                    yahoo_breaker.record_success()
            except Exception:
                yahoo_breaker.record_failure()
                df = pd.DataFrame()

        # 3. Fallback to Google Finance
        if df is None or df.empty:
            df = self._historical_from_google(symbol, period)

        if df is None:
            df = pd.DataFrame()
        if not df.empty:
            cache.set(cache_key, df, CACHE_TTL_HISTORY)
        return df

    def search_stocks(self, query: str) -> list[dict]:
        query_upper = query.upper()
        results = []

        # Search indices first
        for name, ticker in INDICES.items():
            if query_upper in name.upper() or query_upper in ticker.upper():
                results.append({"symbol": name, "name": name, "type": "index"})
                if len(results) >= 20:
                    return results

        # Then search all NSE stocks + ETFs (symbol + company name)
        stock_list = self._get_nse_stock_list()
        for item in stock_list:
            if query_upper in item["symbol"].upper() or query_upper in item["name"].upper():
                item_type = item.get("type", "stock")
                results.append({"symbol": item["symbol"], "name": item["name"], "type": item_type})
                if len(results) >= 20:
                    break
        return results

    def get_popular_stocks(self) -> list[dict]:
        indices = [{"symbol": name, "name": name, "type": "index"} for name in list(INDICES.keys())[:5]]
        stocks = [{"symbol": s, "name": s, "type": "stock"} for s in POPULAR_STOCKS]
        return indices + stocks

    def get_stock_count(self) -> dict:
        """Return count of available stocks and indices."""
        stock_list = self._get_nse_stock_list()
        return {"stocks": len(stock_list), "indices": len(INDICES)}

    def _get_nse_stock_list(self) -> list[dict]:
        """Get full NSE stock list with symbols and company names.
        Fetches from official NSE CSV, falls back to nsetools, then POPULAR_STOCKS."""
        cached = cache.get("nse_stock_list")
        if cached:
            return cached

        stock_list = []

        # 1. Try official NSE CSV (most complete — 2200+ stocks)
        try:
            resp = cffi_requests.get(
                NSE_EQUITY_CSV_URL,
                impersonate="chrome",
                timeout=10,
            )
            if resp.status_code == 200:
                reader = csv.DictReader(io.StringIO(resp.text))
                for row in reader:
                    symbol = row.get("SYMBOL", "").strip()
                    name = row.get("NAME OF COMPANY", "").strip()
                    if symbol:
                        stock_list.append({"symbol": symbol, "name": name})
                logger.info(f"Loaded {len(stock_list)} stocks from NSE CSV")
        except Exception as e:
            logger.warning(f"NSE CSV fetch failed: {e}")

        # 1b. Also fetch ETFs from NSE
        try:
            resp = cffi_requests.get(
                NSE_ETF_CSV_URL,
                impersonate="chrome",
                timeout=10,
            )
            if resp.status_code == 200:
                reader = csv.DictReader(io.StringIO(resp.text))
                etf_count = 0
                seen_symbols = {s["symbol"] for s in stock_list}
                for row in reader:
                    symbol = row.get("Symbol", "").strip()
                    name = row.get("SecurityName", row.get("Underlying", "")).strip()
                    if symbol and symbol not in seen_symbols:
                        stock_list.append({"symbol": symbol, "name": name, "type": "etf"})
                        seen_symbols.add(symbol)
                        etf_count += 1
                logger.info(f"Loaded {etf_count} ETFs from NSE CSV")
        except Exception as e:
            logger.warning(f"NSE ETF CSV fetch failed: {e}")

        # 2. Fallback: nsetools
        if not stock_list and NSE_AVAILABLE:
            try:
                codes = nse_client.get_stock_codes()
                if isinstance(codes, dict):
                    stock_list = [{"symbol": k, "name": v} for k, v in codes.items() if k != "SYMBOL"]
                elif isinstance(codes, list):
                    stock_list = [{"symbol": c, "name": c} for c in codes if c != "SYMBOL"]
                logger.info(f"Loaded {len(stock_list)} stocks from nsetools")
            except Exception:
                pass

        # 3. Fallback: hardcoded popular stocks
        if not stock_list:
            stock_list = [{"symbol": s, "name": s} for s in POPULAR_STOCKS]
            logger.info("Using hardcoded POPULAR_STOCKS as fallback")

        cache.set("nse_stock_list", stock_list, CACHE_TTL_STOCK_LIST)
        return stock_list

    def _fetch_from_nsetools(self, symbol: str) -> dict:
        data = nse_client.get_quote(symbol)
        if not data:
            raise ValueError(f"No data for {symbol}")
        return {
            "symbol": symbol,
            "ltp": data.get("lastPrice", 0),
            "open": data.get("open", 0),
            "high": data.get("dayHigh", 0),
            "low": data.get("dayLow", 0),
            "close": data.get("previousClose", 0),
            "volume": data.get("totalTradedVolume", 0),
            "change": data.get("change", 0),
            "pct_change": data.get("pChange", 0),
            "timestamp": now_ist().isoformat(),
        }

    def _fetch_from_yfinance(self, symbol: str) -> dict:
        yf_symbol = yfinance_symbol(symbol)
        quote = yahoo_quote(yf_symbol)
        if not quote or quote.get("ltp", 0) == 0:
            raise ValueError(f"No data available for {yf_symbol}")
        quote["symbol"] = symbol
        quote["timestamp"] = now_ist().isoformat()
        return quote

    def _historical_from_angel(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV from Angel One. Returns DataFrame or None."""
        if not ANGEL_AVAILABLE:
            return None
        try:
            candles = angel_provider.get_historical_candles(symbol, period)
            if not candles:
                return None
            df = pd.DataFrame(candles)
            # Ensure exact same format as _historical_from_yfinance output
            cols = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]
            return df
        except Exception as e:
            logger.debug(f"Angel One historical fetch failed for {symbol}: {e}")
            return None

    def _intraday_from_angel(self, symbol: str, period: str = "5d", interval: str = "15m") -> Optional[pd.DataFrame]:
        """Fetch intraday candles from Angel One. Returns DataFrame or None."""
        if not ANGEL_AVAILABLE:
            return None
        try:
            candles = angel_provider.get_intraday_candles(symbol, interval, period)
            if not candles:
                return None
            df = pd.DataFrame(candles)
            # Ensure exact same format as yfinance intraday output
            cols = ["datetime", "open", "high", "low", "close", "volume", "datetime_str"]
            df = df[[c for c in cols if c in df.columns]]
            return df
        except Exception as e:
            logger.debug(f"Angel One intraday fetch failed for {symbol}: {e}")
            return None

    def _historical_from_yfinance(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        yf_symbol = yfinance_symbol(symbol)
        df = yahoo_chart(yf_symbol, period=period, interval="1d")
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df


    def _google_symbol(self, symbol: str) -> str:
        """Convert symbol to Google Finance format (NSE:SYMBOL)."""
        from app.utils.helpers import is_index
        if is_index(symbol):
            gmap = {
                "NIFTY 50": "NIFTY_50:INDEXNSE", "NIFTY BANK": "NIFTY_BANK:INDEXNSE",
                "SENSEX": "SENSEX:INDEXBOM", "INDIA VIX": "INDIAVIX:INDEXNSE",
            }
            return gmap.get(symbol, symbol)
        return f"{symbol}:NSE"

    def _intraday_from_google(self, symbol: str, period: str = "5d",
                               interval: str = "15m") -> Optional[pd.DataFrame]:
        """Fetch intraday data from Google Finance chart API."""
        try:
            gsym = self._google_symbol(symbol)
            # Google Finance interval: 900=15m, 120=2m, 600=10m, 1800=30m, 3600=1h
            interval_map = {"2m": 120, "5m": 300, "10m": 600, "15m": 900, "30m": 1800, "1h": 3600}
            period_map = {"1d": "1d", "5d": "5d", "1mo": "1M"}
            gi = interval_map.get(interval, 900)
            gp = period_map.get(period, "5d")

            url = f"https://www.google.com/finance/quote/{gsym}"
            resp = cffi_requests.get(url, timeout=10, impersonate="chrome")
            if resp.status_code != 200:
                return None

            # Extract JSON data from page (Google embeds chart data in HTML)
            import re, json as _json
            # Look for the chart data pattern
            match = re.search(r'data-last-price="([^"]+)"', resp.text)
            if not match:
                return None

            # Google Finance doesn't expose raw OHLCV via simple scraping.
            # Use the Yahoo chart as the actual data source — this method
            # serves as a connectivity check + fallback quote.
            logger.debug(f"Google Finance page loaded for {symbol}, but OHLCV requires Yahoo/Angel")
            return None
        except Exception as e:
            logger.debug(f"Google Finance intraday failed for {symbol}: {e}")
            return None

    def _historical_from_google(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch historical daily data from Google Finance via Trending Stocks API."""
        try:
            # Use an alternative: NSE India's own API for historical data
            nse_symbol = symbol.replace(" ", "%20")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/",
            }

            # First get a session cookie
            session = cffi_requests.Session(impersonate="chrome")
            session.get("https://www.nseindia.com/", timeout=10)

            # Map period to from/to dates
            from datetime import datetime, timedelta
            to_date = datetime.now()
            period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
            days = period_days.get(period, 365)
            from_date = to_date - timedelta(days=days)

            url = (
                f"https://www.nseindia.com/api/historical/cm/equity?"
                f"symbol={nse_symbol}&series=[%22EQ%22]"
                f"&from={from_date.strftime('%d-%m-%Y')}&to={to_date.strftime('%d-%m-%Y')}"
            )
            resp = session.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return None

            data = resp.json()
            records = data.get("data", [])
            if not records:
                return None

            rows = []
            for r in records:
                try:
                    rows.append({
                        "date": r.get("CH_TIMESTAMP", ""),
                        "open": float(r.get("CH_OPENING_PRICE", 0)),
                        "high": float(r.get("CH_TRADE_HIGH_PRICE", 0)),
                        "low": float(r.get("CH_TRADE_LOW_PRICE", 0)),
                        "close": float(r.get("CH_CLOSING_PRICE", 0)),
                        "volume": int(r.get("CH_TOT_TRADED_QTY", 0)),
                    })
                except (ValueError, TypeError):
                    continue

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df = df.sort_values("date").reset_index(drop=True)
            logger.info(f"NSE India historical data fetched for {symbol}: {len(df)} rows")
            return df

        except Exception as e:
            logger.debug(f"NSE India historical fetch failed for {symbol}: {e}")
            return None


data_fetcher = DataFetcher()
