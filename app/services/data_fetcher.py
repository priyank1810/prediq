import io
import csv
import logging
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Optional
from app.utils.cache import cache
from app.utils.helpers import yfinance_symbol
from app.config import CACHE_TTL_QUOTE, CACHE_TTL_HISTORY, CACHE_TTL_STOCK_LIST, CACHE_TTL_INTRADAY, POPULAR_STOCKS, INDICES

NSE_EQUITY_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

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
        if ANGEL_AVAILABLE:
            try:
                quote = angel_provider.get_live_quote(symbol)
                if quote:
                    logger.debug(f"Got real-time quote for {symbol} from Angel One")
            except Exception as e:
                logger.debug(f"Angel One failed for {symbol}: {e}")

        # 2. Try nsetools (doesn't support indices)
        if quote is None and not is_index(symbol) and NSE_AVAILABLE:
            try:
                quote = self._fetch_from_nsetools(symbol)
            except Exception:
                pass

        # 3. Fallback to yfinance (15-20 min delayed)
        if quote is None:
            quote = self._fetch_from_yfinance(symbol)

        if quote:
            cache.set(cache_key, quote, CACHE_TTL_QUOTE)
        return quote

    def get_bulk_quotes(self, symbols: list[str]) -> list[dict]:
        """Fetch quotes for multiple symbols, using Angel One batch API when available."""
        results = {}

        # Try Angel One batch first (single API call for all symbols)
        if ANGEL_AVAILABLE:
            try:
                angel_quotes = angel_provider.get_multiple_quotes(symbols)
                for sym, quote in angel_quotes.items():
                    results[sym] = quote
                    cache.set(f"quote:{sym}", quote, CACHE_TTL_QUOTE)
            except Exception as e:
                logger.debug(f"Angel One batch failed: {e}")

        # Fetch remaining symbols individually (yfinance fallback)
        for symbol in symbols:
            if symbol not in results:
                try:
                    quote = self.get_live_quote(symbol)
                    if quote:
                        results[symbol] = quote
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

        # 2. Fallback to yfinance
        if df is None or df.empty:
            yf_symbol = yfinance_symbol(symbol)
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            if df.empty:
                df = pd.DataFrame()
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()
                datetime_col = "Datetime" if "Datetime" in df.columns else "Date"
                df = df.rename(columns={
                    datetime_col: "datetime", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
                cols = ["datetime", "open", "high", "low", "close", "volume"]
                df = df[[c for c in cols if c in df.columns]]
                df["datetime_str"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df[col] = df[col].round(2)
                if "volume" in df.columns:
                    df["volume"] = df["volume"].astype(int)

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
        if df is None or df.empty:
            df = self._historical_from_yfinance(symbol, period)

        if df is not None and not df.empty:
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

        # Then search all NSE stocks (symbol + company name)
        stock_list = self._get_nse_stock_list()
        for item in stock_list:
            if query_upper in item["symbol"].upper() or query_upper in item["name"].upper():
                results.append({"symbol": item["symbol"], "name": item["name"], "type": "stock"})
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
            resp = requests.get(
                NSE_EQUITY_CSV_URL,
                headers={"User-Agent": "Mozilla/5.0"},
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
            "timestamp": datetime.now().isoformat(),
        }

    def _fetch_from_yfinance(self, symbol: str) -> dict:
        yf_symbol = yfinance_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)

        try:
            # fast_info can raise KeyError intermittently; fall back to history
            try:
                info = ticker.fast_info
                ltp = info.last_price
                prev_close = info.previous_close
            except (KeyError, AttributeError):
                ltp = None
                prev_close = None

            hist = ticker.history(period="1d")
            if hist.empty and ltp is None:
                raise ValueError(f"No data available for {yf_symbol}")

            if not hist.empty:
                open_price = float(hist["Open"].iloc[-1])
                high = float(hist["High"].iloc[-1])
                low = float(hist["Low"].iloc[-1])
                close = float(hist["Close"].iloc[-1])
                volume = int(hist["Volume"].iloc[-1])
                if ltp is None:
                    ltp = close
                if prev_close is None:
                    prev_close = float(hist["Close"].iloc[0]) if len(hist) > 1 else ltp
            else:
                open_price = high = low = volume = 0
                close = prev_close or 0

            change = ltp - prev_close if ltp and prev_close else 0
            pct_change = (change / prev_close * 100) if prev_close else 0

            return {
                "symbol": symbol,
                "ltp": round(ltp, 2) if ltp else 0,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(prev_close, 2) if prev_close else 0,
                "volume": volume,
                "change": round(change, 2),
                "pct_change": round(pct_change, 2),
                "timestamp": datetime.now().isoformat(),
            }
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to fetch quote for {symbol}: {e}")

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
        df = yf.download(yf_symbol, period=period, progress=False)
        if df.empty:
            return pd.DataFrame()

        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                "Low": "low", "Close": "close", "Volume": "volume"})
        # Keep only needed columns
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].round(2)
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(int)
        return df


data_fetcher = DataFetcher()
