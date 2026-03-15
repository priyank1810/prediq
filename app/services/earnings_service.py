import logging
from datetime import datetime, timedelta
from app.utils.cache import cache

logger = logging.getLogger(__name__)

class EarningsService:
    """Fetch upcoming earnings dates from Yahoo Finance."""

    def get_earnings(self, symbols: list[str]) -> list[dict]:
        """Get upcoming earnings for a list of symbols."""
        cache_key = f"earnings_{'_'.join(sorted(symbols[:10]))}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        results = []
        for symbol in symbols:
            try:
                data = self._fetch_earnings(symbol)
                if data:
                    results.append(data)
            except Exception as e:
                logger.debug(f"Earnings fetch failed for {symbol}: {e}")

        # Sort by earnings date (soonest first)
        results.sort(key=lambda x: x.get("earnings_date", "9999-12-31"))
        cache.set(cache_key, results, ttl=3600)  # Cache 1 hour
        return results

    def _fetch_earnings(self, symbol: str) -> dict | None:
        """Fetch next earnings date for a single symbol via yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            cal = ticker.calendar
            if cal is None or (hasattr(cal, 'empty') and cal.empty):
                return None

            # yfinance returns calendar as dict or DataFrame
            earnings_date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and len(ed) > 0:
                        earnings_date = ed[0]
                    elif hasattr(ed, 'isoformat'):
                        earnings_date = ed
            else:
                # DataFrame case
                if "Earnings Date" in cal.index:
                    vals = cal.loc["Earnings Date"]
                    if hasattr(vals, '__iter__'):
                        for v in vals:
                            if v and str(v) != 'NaT':
                                earnings_date = v
                                break
                    elif vals and str(vals) != 'NaT':
                        earnings_date = vals

            if earnings_date is None:
                return None

            # Convert to string
            if hasattr(earnings_date, 'strftime'):
                date_str = earnings_date.strftime("%Y-%m-%d")
            else:
                date_str = str(earnings_date)[:10]

            # Calculate days until earnings
            try:
                ed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                days_until = (ed_date - datetime.now().date()).days
            except Exception:
                days_until = None

            return {
                "symbol": symbol,
                "earnings_date": date_str,
                "days_until": days_until,
            }
        except Exception as e:
            logger.debug(f"yfinance earnings failed for {symbol}: {e}")
            return None


earnings_service = EarningsService()
