import logging
from app.utils.cache import cache

logger = logging.getLogger(__name__)

CACHE_TTL_FUNDAMENTALS = 86400  # 24 hours (fundamentals change quarterly)


class FundamentalService:
    """Fetches fundamental data via Yahoo Finance and caches for 24 hours."""

    def get_fundamentals(self, symbol: str) -> dict:
        cache_key = f"fundamentals:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._fetch(symbol)
        cache.set(cache_key, data, CACHE_TTL_FUNDAMENTALS)
        return data

    def _fetch(self, symbol: str) -> dict:
        try:
            from app.utils.helpers import is_index, yfinance_symbol
            from app.utils.yahoo_api import yahoo_fundamentals

            if is_index(symbol):
                return self._empty_result()

            ticker_symbol = yfinance_symbol(symbol)
            info = yahoo_fundamentals(ticker_symbol)

            if not info or not info.get("regularMarketPrice"):
                ticker_symbol = yfinance_symbol(symbol, exchange="BSE")
                info = yahoo_fundamentals(ticker_symbol)

            if not info:
                return self._empty_result()

            result = {
                # Key ratios
                "pe": info.get("pe"),
                "forward_pe": info.get("forward_pe"),
                "pb": info.get("pb"),
                "roe": self._pct(info.get("roe")),
                "roa": self._pct(info.get("roa")),
                "de": info.get("de"),
                "rev_growth": self._pct(info.get("rev_growth")),
                "earn_growth": self._pct(info.get("earn_growth")),
                "div_yield": self._pct(info.get("div_yield")),
                "peg_ratio": info.get("peg_ratio"),
                "beta": info.get("beta"),
                # Margins
                "profit_margin": self._pct(info.get("profit_margins")),
                "operating_margin": self._pct(info.get("operating_margins")),
                "gross_margin": self._pct(info.get("gross_margins")),
                # Valuations
                "market_cap": info.get("market_cap"),
                "book_value": info.get("book_value"),
                "eps_trailing": info.get("eps_trailing"),
                "eps_forward": info.get("eps_forward"),
                # Revenue & Profit
                "total_revenue": info.get("total_revenue"),
                "ebitda": info.get("ebitda"),
                "free_cashflow": info.get("free_cashflow"),
                "operating_cashflow": info.get("operating_cashflow"),
                "total_cash": info.get("total_cash"),
                "total_debt": info.get("total_debt"),
                "current_ratio": info.get("current_ratio"),
                # Price levels
                "52_week_high": info.get("52_week_high"),
                "52_week_low": info.get("52_week_low"),
                "50_day_avg": info.get("50_day_avg"),
                "200_day_avg": info.get("200_day_avg"),
                # Holdings
                "promoter_holding": self._pct(info.get("promoter_holding")),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "symbol": symbol,
                # Financial statements
                "income_annual": info.get("income_annual", []),
                "income_quarterly": info.get("income_quarterly", []),
                "balance_annual": info.get("balance_annual", []),
                "earnings_quarterly": info.get("earnings_quarterly", []),
                "yearly_financials": info.get("yearly_financials", []),
            }

            # Clean None values to 0 for numeric fields
            for key in ["pe", "forward_pe", "pb", "roe", "roa", "de",
                        "rev_growth", "earn_growth", "div_yield",
                        "profit_margin", "operating_margin", "gross_margin"]:
                if result[key] is None:
                    result[key] = 0

            # D/E from Yahoo is sometimes as percentage
            if result["de"] and result["de"] > 10:
                result["de"] = result["de"] / 100.0

            logger.info(f"Fetched fundamentals for {symbol}: PE={result['pe']}, PB={result['pb']}")
            return result

        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
            return self._empty_result()

    @staticmethod
    def _pct(val):
        if val is None:
            return None
        return round(float(val) * 100, 2)

    @staticmethod
    def _empty_result():
        return {
            "pe": 0, "forward_pe": 0, "pb": 0, "roe": 0, "roa": 0, "de": 0,
            "rev_growth": 0, "earn_growth": 0, "div_yield": 0,
            "peg_ratio": None, "beta": None,
            "profit_margin": 0, "operating_margin": 0, "gross_margin": 0,
            "market_cap": 0, "book_value": None,
            "eps_trailing": None, "eps_forward": None,
            "total_revenue": None, "ebitda": None,
            "free_cashflow": None, "operating_cashflow": None,
            "total_cash": None, "total_debt": None, "current_ratio": None,
            "52_week_high": None, "52_week_low": None,
            "50_day_avg": None, "200_day_avg": None,
            "promoter_holding": 0,
            "sector": "", "industry": "", "symbol": "",
            "income_annual": [], "income_quarterly": [],
            "balance_annual": [], "earnings_quarterly": [],
            "yearly_financials": [],
        }


fundamental_service = FundamentalService()
