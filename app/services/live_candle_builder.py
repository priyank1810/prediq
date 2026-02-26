"""Builds real-time OHLCV candles from Angel One tick data.

Accumulates ticks into 1-minute bars, then resamples to 5m/15m candles.
Provides DataFrames in the same format as yfinance for seamless integration
with the prediction pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


class LiveCandleBuilder:
    def __init__(self):
        # Raw ticks: {symbol: [(timestamp, ltp, volume, bid, ask), ...]}
        self._ticks = defaultdict(list)
        # Built 1-minute candles: {symbol: DataFrame}
        self._candles_1m = {}
        # Latest spread: {symbol: float}
        self._spreads = {}
        # Max ticks to keep in memory per symbol (1 day worth at ~1 tick/sec)
        self._max_ticks = 30000

    def on_tick(self, symbol: str, ltp: float, volume: int = 0, bid: float = 0.0, ask: float = 0.0):
        """Called from WebSocket price_streamer when Angel One provides a quote."""
        now = datetime.now()
        self._ticks[symbol].append((now, ltp, volume, bid, ask))

        # Track spread
        if bid > 0 and ask > 0:
            self._spreads[symbol] = ask - bid

        # Trim old ticks
        if len(self._ticks[symbol]) > self._max_ticks:
            self._ticks[symbol] = self._ticks[symbol][-self._max_ticks:]

        # Rebuild 1-minute candles periodically (every 60 ticks)
        if len(self._ticks[symbol]) % 60 == 0:
            self._build_1m_candles(symbol)

    def _build_1m_candles(self, symbol: str):
        """Convert raw ticks into 1-minute OHLCV candles."""
        ticks = self._ticks.get(symbol, [])
        if not ticks:
            return

        rows = []
        for ts, ltp, vol, bid, ask in ticks:
            rows.append({"datetime": ts, "price": ltp, "volume": vol})

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

        # Resample to 1-minute candles
        candles = df["price"].resample("1min").ohlc()
        candles.columns = ["open", "high", "low", "close"]
        candles["volume"] = df["volume"].resample("1min").sum()
        candles = candles.dropna(subset=["open"])

        self._candles_1m[symbol] = candles

    def get_intraday_df(self, symbol: str, interval: str = "15m") -> pd.DataFrame:
        """Get intraday candles as a DataFrame matching yfinance format.

        Returns empty DataFrame if insufficient data.
        """
        # Ensure latest candles are built
        self._build_1m_candles(symbol)

        candles = self._candles_1m.get(symbol)
        if candles is None or candles.empty:
            return pd.DataFrame()

        # Resample to requested interval
        resampled = candles["close"].resample(interval).ohlc() if interval != "1min" else candles[["open", "high", "low", "close"]].copy()

        if interval != "1min":
            resampled.columns = ["open", "high", "low", "close"]
            resampled["volume"] = candles["volume"].resample(interval).sum()
        else:
            resampled = candles.copy()

        resampled = resampled.dropna(subset=["open"])

        if resampled.empty:
            return pd.DataFrame()

        # Format like yfinance output
        result = resampled.reset_index()
        result = result.rename(columns={"datetime": "datetime"})
        result["datetime_str"] = result["datetime"].dt.strftime("%Y-%m-%d %H:%M")

        for col in ["open", "high", "low", "close"]:
            if col in result.columns:
                result[col] = result[col].round(2)
        if "volume" in result.columns:
            result["volume"] = result["volume"].astype(int)

        return result

    def get_spread(self, symbol: str) -> float:
        """Get latest bid-ask spread for a symbol."""
        return self._spreads.get(symbol, 0.0)

    def has_data(self, symbol: str) -> bool:
        """Check if we have live candle data for a symbol."""
        return symbol in self._ticks and len(self._ticks[symbol]) > 60

    def get_symbols(self) -> list:
        """Get list of symbols with live data."""
        return list(self._ticks.keys())


# Singleton instance
candle_builder = LiveCandleBuilder()
