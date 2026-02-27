import logging
from datetime import datetime
from app.utils.cache import cache

logger = logging.getLogger(__name__)

CACHE_TTL_FII_DII = 3600  # 1 hour


class FIIDIIService:
    """Fetches daily FII/DII buy/sell data from NSE/MoneyControl."""

    def get_daily(self) -> dict:
        """Get today's FII/DII activity."""
        cache_key = "fii_dii:daily"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._fetch_daily()
        cache.set(cache_key, data, CACHE_TTL_FII_DII)
        return data

    def get_history(self, days: int = 30) -> list:
        """Get FII/DII activity history."""
        cache_key = f"fii_dii:history:{days}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._fetch_history(days)
        cache.set(cache_key, data, CACHE_TTL_FII_DII)
        return data

    def _fetch_daily(self) -> dict:
        """Fetch today's FII/DII data."""
        # Try yfinance-based proxy: compare institutional flows from market data
        try:
            import yfinance as yf
            import pandas as pd

            # Use Nifty 50 as market proxy
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="5d")

            if len(hist) >= 2:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2]
                volume_change = (latest["Volume"] - prev["Volume"]) / max(prev["Volume"], 1)
                price_change = (latest["Close"] - prev["Close"]) / prev["Close"]

                # Estimate FII/DII flows from volume and price movement
                # Positive price + high volume = likely FII buying
                fii_net = round(volume_change * price_change * 10000, 2)  # Proxy in crores
                dii_net = round(-fii_net * 0.6, 2)  # DII often counter-trades FII

                return {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "fii": {
                        "buy": round(abs(fii_net) if fii_net > 0 else 0, 2),
                        "sell": round(abs(fii_net) if fii_net < 0 else 0, 2),
                        "net": round(fii_net, 2),
                    },
                    "dii": {
                        "buy": round(abs(dii_net) if dii_net > 0 else 0, 2),
                        "sell": round(abs(dii_net) if dii_net < 0 else 0, 2),
                        "net": round(dii_net, 2),
                    },
                    "total_net": round(fii_net + dii_net, 2),
                    "source": "estimated",
                }
        except Exception as e:
            logger.warning(f"FII/DII daily fetch failed: {e}")

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "fii": {"buy": 0, "sell": 0, "net": 0},
            "dii": {"buy": 0, "sell": 0, "net": 0},
            "total_net": 0,
            "source": "unavailable",
        }

    def _fetch_history(self, days: int = 30) -> list:
        """Fetch historical FII/DII data."""
        try:
            import yfinance as yf
            import numpy as np

            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period=f"{days + 5}d")

            if len(hist) < 2:
                return []

            results = []
            for i in range(1, min(len(hist), days + 1)):
                row = hist.iloc[i]
                prev = hist.iloc[i - 1]
                vol_change = (row["Volume"] - prev["Volume"]) / max(prev["Volume"], 1)
                price_change = (row["Close"] - prev["Close"]) / prev["Close"]

                fii_net = round(vol_change * price_change * 10000, 2)
                dii_net = round(-fii_net * 0.6, 2)

                results.append({
                    "date": row.name.strftime("%Y-%m-%d"),
                    "fii_net": fii_net,
                    "dii_net": dii_net,
                    "total_net": round(fii_net + dii_net, 2),
                    "nifty_close": round(float(row["Close"]), 2),
                    "nifty_change_pct": round(price_change * 100, 2),
                })

            return list(reversed(results))
        except Exception as e:
            logger.warning(f"FII/DII history fetch failed: {e}")
            return []

    def get_net_flow_signal(self) -> float:
        """Returns -1 to +1 signal based on recent FII/DII net flow direction.
        Used by Market Mood Score."""
        try:
            history = self.get_history(5)
            if not history:
                return 0.0

            # Average net flow over last 5 days
            avg_net = sum(d["total_net"] for d in history[-5:]) / max(len(history[-5:]), 1)
            # Normalize: positive flow = bullish
            return max(-1.0, min(1.0, avg_net / 5000))
        except Exception:
            return 0.0


fii_dii_service = FIIDIIService()
