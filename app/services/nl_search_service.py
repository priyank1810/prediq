"""Natural language search for stocks.

Parses queries like:
- "oversold banking stocks with high volume"
- "stocks above SMA 200 with bullish MACD"
- "undervalued stocks with strong growth"
- "top gainers today"
- "stocks near 52 week low"
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Keyword → screener filter mappings
FILTER_PATTERNS = [
    # RSI
    (r"\boversold\b", {"rsi_oversold": True}),
    (r"\boverbought\b", {"rsi_overbought": True}),
    # MACD
    (r"\bbullish\s*macd\b|\bmacd\s*bullish\b|\bmacd\s*cross(over)?\b", {"macd_bullish": True}),
    (r"\bbearish\s*macd\b|\bmacd\s*bearish\b", {"macd_bearish": True}),
    # SMA
    (r"\babove\s*(sma\s*)?200\b|\b200\s*sma\b.*\babove\b", {"above_sma_200": True}),
    (r"\babove\s*(sma\s*)?50\b|\b50\s*sma\b.*\babove\b", {"above_sma_50": True}),
    (r"\babove\s*(sma\s*)?20\b|\b20\s*sma\b.*\babove\b", {"above_sma_20": True}),
    (r"\bbelow\s*(sma\s*)?200\b|\b200\s*sma\b.*\bbelow\b", {"below_sma_200": True}),
    (r"\bbelow\s*(sma\s*)?50\b|\b50\s*sma\b.*\bbelow\b", {"below_sma_50": True}),
    (r"\bbelow\s*(sma\s*)?20\b|\b20\s*sma\b.*\bbelow\b", {"below_sma_20": True}),
    # Volume
    (r"\bhigh\s*vol(ume)?\b|\bvol(ume)?\s*spike\b|\bheavy\s*vol(ume)?\b", {"volume_spike": True}),
    # Price change
    (r"\bgainer|gaining|up\s*(?:more\s*than\s*)?(\d+)%", None),  # handled specially
    (r"\bloser|losing|down\s*(?:more\s*than\s*)?(\d+)%", None),  # handled specially
]

# Sector keywords
SECTOR_KEYWORDS = {
    "bank": "Banking", "banking": "Banking", "finance": "Banking", "financial": "Banking",
    "it": "IT", "tech": "IT", "software": "IT", "technology": "IT",
    "pharma": "Pharma", "healthcare": "Pharma", "health": "Pharma",
    "auto": "Auto", "automobile": "Auto", "automotive": "Auto",
    "metal": "Metal", "steel": "Metal", "mining": "Metal",
    "energy": "Energy", "oil": "Energy", "gas": "Energy", "power": "Energy",
    "fmcg": "FMCG", "consumer": "FMCG",
    "realty": "Realty", "real estate": "Realty", "property": "Realty",
    "infra": "Infra", "infrastructure": "Infra", "construction": "Infra",
    "telecom": "Telecom", "communication": "Telecom",
    "cement": "Cement",
    "chemical": "Chemical", "chemicals": "Chemical",
}

# Special query patterns
SPECIAL_QUERIES = {
    r"\btop\s*gainer|biggest\s*gainer|most\s*up\b": "top_gainers",
    r"\btop\s*loser|biggest\s*loser|most\s*down\b": "top_losers",
    r"\b52\s*week\s*low|yearly\s*low|year\s*low\b": "near_52w_low",
    r"\b52\s*week\s*high|yearly\s*high|year\s*high\b": "near_52w_high",
    r"\bundervalued|cheap|low\s*pe|value\s*stock\b": "undervalued",
    r"\bgrowth\s*stock|high\s*growth|fast\s*grow\b": "high_growth",
    r"\bdividend|high\s*yield|income\b": "high_dividend",
    r"\bmomentum|trending|breakout\b": "momentum",
}


class NLSearchService:
    """Parse natural language queries into screener filters and execute them."""

    def search(self, query: str) -> dict:
        """Parse a natural language query and return matching stocks."""
        query_lower = query.lower().strip()

        # Check for special queries first
        for pattern, query_type in SPECIAL_QUERIES.items():
            if re.search(pattern, query_lower):
                return self._handle_special_query(query_type, query_lower)

        # Parse filters from query
        filters = {}
        for pattern, filter_map in FILTER_PATTERNS:
            if filter_map and re.search(pattern, query_lower):
                filters.update(filter_map)

        # Parse price change thresholds
        gainer_match = re.search(r"\b(?:up|gain\w*)\s*(?:more\s*than\s*)?(\d+)\s*%", query_lower)
        if gainer_match:
            filters["price_change_min"] = float(gainer_match.group(1))
        loser_match = re.search(r"\b(?:down|los\w*)\s*(?:more\s*than\s*)?(\d+)\s*%", query_lower)
        if loser_match:
            filters["price_change_max"] = -float(loser_match.group(1))

        # Detect sector filter
        sector = self._detect_sector(query_lower)

        # If no filters detected, try keyword-based heuristics
        if not filters:
            filters = self._infer_filters(query_lower)

        if not filters:
            return {
                "query": query,
                "interpreted_as": "No recognizable filters found",
                "filters": {},
                "results": [],
                "suggestions": [
                    "Try: 'oversold stocks with high volume'",
                    "Try: 'banking stocks above SMA 200'",
                    "Try: 'top gainers today'",
                    "Try: 'undervalued stocks with strong growth'",
                ],
            }

        # Run screener
        from app.services.screener_service import screener_service
        results = screener_service.scan(filters)

        # Filter by sector if specified
        if sector:
            from app.config import SECTOR_MAP
            sector_symbols = set(SECTOR_MAP.get(sector, []))
            if sector_symbols:
                results = [r for r in results if r["symbol"] in sector_symbols]

        # Build interpretation string
        interpretation = self._build_interpretation(filters, sector)

        return {
            "query": query,
            "interpreted_as": interpretation,
            "filters": filters,
            "sector": sector,
            "result_count": len(results),
            "results": results[:20],  # Limit to 20
        }

    def _detect_sector(self, query: str) -> Optional[str]:
        """Detect sector from query keywords."""
        for keyword, sector in SECTOR_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", query):
                return sector
        return None

    def _infer_filters(self, query: str) -> dict:
        """Infer filters from less explicit queries."""
        filters = {}

        # "bullish" without specific indicator → bullish MACD + above SMA 20
        if re.search(r"\bbullish\b", query) and not any(
            re.search(p, query) for p, _ in FILTER_PATTERNS if _ and "macd" in str(_)
        ):
            filters["macd_bullish"] = True
            filters["above_sma_20"] = True

        # "bearish" → bearish MACD + below SMA 20
        if re.search(r"\bbearish\b", query):
            filters["macd_bearish"] = True
            filters["below_sma_20"] = True

        # "strong" → above SMA 50 + volume
        if re.search(r"\bstrong\b", query):
            filters["above_sma_50"] = True
            filters["volume_spike"] = True

        # "weak" → below SMA 50
        if re.search(r"\bweak\b", query):
            filters["below_sma_50"] = True

        return filters

    def _handle_special_query(self, query_type: str, query: str) -> dict:
        """Handle special query types that don't map to screener filters."""
        sector = self._detect_sector(query)

        if query_type == "top_gainers":
            return self._get_top_movers("gainers", sector)
        elif query_type == "top_losers":
            return self._get_top_movers("losers", sector)
        elif query_type in ("near_52w_low", "near_52w_high"):
            return self._get_near_52w(query_type, sector)
        elif query_type == "undervalued":
            return self._get_fundamental_screen("undervalued", sector)
        elif query_type == "high_growth":
            return self._get_fundamental_screen("high_growth", sector)
        elif query_type == "high_dividend":
            return self._get_fundamental_screen("high_dividend", sector)
        elif query_type == "momentum":
            from app.services.screener_service import screener_service
            results = screener_service.scan({
                "above_sma_20": True, "above_sma_50": True, "volume_spike": True
            })
            if sector:
                from app.config import SECTOR_MAP
                syms = set(SECTOR_MAP.get(sector, []))
                results = [r for r in results if r["symbol"] in syms]
            return {
                "query": query, "interpreted_as": "Momentum stocks (above SMA 20 & 50, high volume)",
                "filters": {"above_sma_20": True, "above_sma_50": True, "volume_spike": True},
                "sector": sector, "result_count": len(results), "results": results[:20],
            }

        return {"query": query, "interpreted_as": query_type, "results": []}

    def _get_top_movers(self, direction: str, sector: Optional[str]) -> dict:
        """Get top gaining or losing stocks."""
        from app.services.data_fetcher import data_fetcher
        from app.config import POPULAR_STOCKS, SECTOR_MAP

        symbols = list(POPULAR_STOCKS)
        if sector:
            sector_syms = set(SECTOR_MAP.get(sector, []))
            symbols = [s for s in symbols if s in sector_syms] or symbols

        quotes = data_fetcher.get_bulk_quotes(symbols)
        valid = [q for q in quotes if q.get("pct_change") is not None]

        if direction == "gainers":
            valid.sort(key=lambda q: q.get("pct_change", 0), reverse=True)
            label = "Top gainers"
        else:
            valid.sort(key=lambda q: q.get("pct_change", 0))
            label = "Top losers"

        results = []
        for q in valid[:15]:
            results.append({
                "symbol": q.get("symbol", ""),
                "ltp": q.get("ltp", 0),
                "change_pct": q.get("pct_change", 0),
                "matched_filters": [direction],
            })

        return {
            "query": label,
            "interpreted_as": f"{label}" + (f" in {sector}" if sector else ""),
            "result_count": len(results),
            "results": results,
        }

    def _get_near_52w(self, query_type: str, sector: Optional[str]) -> dict:
        """Get stocks near 52-week high or low."""
        from app.services.data_fetcher import data_fetcher
        from app.config import POPULAR_STOCKS, SECTOR_MAP
        from concurrent.futures import ThreadPoolExecutor

        symbols = list(POPULAR_STOCKS)
        if sector:
            sector_syms = set(SECTOR_MAP.get(sector, []))
            symbols = [s for s in symbols if s in sector_syms] or symbols

        quotes = data_fetcher.get_bulk_quotes(symbols)
        quote_map = {q["symbol"]: q for q in quotes if q.get("symbol")}

        # Fetch fundamentals to get 52-week data
        def _get_52w(sym):
            try:
                from app.services.fundamental_service import fundamental_service
                fund = fundamental_service.get_fundamentals(sym)
                if fund:
                    return sym, fund.get("52_week_high"), fund.get("52_week_low")
            except Exception:
                pass
            return sym, None, None

        results = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_get_52w, s) for s in symbols[:30]]
            for f in futures:
                sym, high_52, low_52 = f.result()
                q = quote_map.get(sym, {})
                ltp = q.get("ltp", 0)
                if not ltp:
                    continue

                if query_type == "near_52w_low" and low_52 and low_52 > 0:
                    pct_from_low = (ltp - low_52) / low_52 * 100
                    if pct_from_low < 10:  # Within 10% of 52-week low
                        results.append({
                            "symbol": sym, "ltp": ltp,
                            "change_pct": q.get("pct_change", 0),
                            "matched_filters": [f"{pct_from_low:.1f}% from 52W low (₹{low_52:,.0f})"],
                            "_sort": pct_from_low,
                        })
                elif query_type == "near_52w_high" and high_52 and high_52 > 0:
                    pct_from_high = (high_52 - ltp) / high_52 * 100
                    if pct_from_high < 10:
                        results.append({
                            "symbol": sym, "ltp": ltp,
                            "change_pct": q.get("pct_change", 0),
                            "matched_filters": [f"{pct_from_high:.1f}% from 52W high (₹{high_52:,.0f})"],
                            "_sort": pct_from_high,
                        })

        results.sort(key=lambda r: r.get("_sort", 999))
        for r in results:
            r.pop("_sort", None)

        label = "near 52-week low" if query_type == "near_52w_low" else "near 52-week high"
        return {
            "query": f"Stocks {label}",
            "interpreted_as": f"Stocks {label}" + (f" in {sector}" if sector else ""),
            "result_count": len(results),
            "results": results[:15],
        }

    def _get_fundamental_screen(self, screen_type: str, sector: Optional[str]) -> dict:
        """Screen stocks by fundamental criteria."""
        from app.services.data_fetcher import data_fetcher
        from app.config import POPULAR_STOCKS, SECTOR_MAP
        from concurrent.futures import ThreadPoolExecutor

        symbols = list(POPULAR_STOCKS)
        if sector:
            sector_syms = set(SECTOR_MAP.get(sector, []))
            symbols = [s for s in symbols if s in sector_syms] or symbols

        quotes = data_fetcher.get_bulk_quotes(symbols)
        quote_map = {q["symbol"]: q for q in quotes if q.get("symbol")}

        def _screen(sym):
            try:
                from app.services.fundamental_service import fundamental_service
                fund = fundamental_service.get_fundamentals(sym)
                if not fund or not fund.get("pe"):
                    return None

                q = quote_map.get(sym, {})
                result = {
                    "symbol": sym, "ltp": q.get("ltp", 0),
                    "change_pct": q.get("pct_change", 0),
                }

                if screen_type == "undervalued":
                    pe = fund.get("pe", 0)
                    pb = fund.get("pb", 0)
                    roe = fund.get("roe", 0)
                    if pe and 0 < pe < 20 and roe and roe > 10:
                        result["matched_filters"] = [f"P/E: {pe:.1f}", f"ROE: {roe:.1f}%"]
                        if pb:
                            result["matched_filters"].append(f"P/B: {pb:.1f}")
                        result["_sort"] = pe
                        return result

                elif screen_type == "high_growth":
                    rev_g = fund.get("rev_growth", 0)
                    earn_g = fund.get("earn_growth", 0)
                    if rev_g and rev_g > 15:
                        result["matched_filters"] = [f"Rev Growth: {rev_g:.0f}%"]
                        if earn_g:
                            result["matched_filters"].append(f"Earn Growth: {earn_g:.0f}%")
                        result["_sort"] = -rev_g
                        return result

                elif screen_type == "high_dividend":
                    div = fund.get("div_yield", 0)
                    if div and div > 2:
                        result["matched_filters"] = [f"Div Yield: {div:.1f}%"]
                        result["_sort"] = -div
                        return result

            except Exception:
                pass
            return None

        results = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_screen, s) for s in symbols[:30]]
            for f in futures:
                r = f.result()
                if r:
                    results.append(r)

        results.sort(key=lambda r: r.get("_sort", 999))
        for r in results:
            r.pop("_sort", None)

        labels = {
            "undervalued": "Undervalued stocks (low P/E, high ROE)",
            "high_growth": "High growth stocks (revenue growth > 15%)",
            "high_dividend": "High dividend yield stocks (> 2%)",
        }

        return {
            "query": labels.get(screen_type, screen_type),
            "interpreted_as": labels.get(screen_type, screen_type) + (f" in {sector}" if sector else ""),
            "result_count": len(results),
            "results": results[:15],
        }

    @staticmethod
    def _build_interpretation(filters: dict, sector: Optional[str]) -> str:
        """Build human-readable interpretation of parsed filters."""
        parts = []

        filter_labels = {
            "rsi_oversold": "RSI oversold (< 30)",
            "rsi_overbought": "RSI overbought (> 70)",
            "macd_bullish": "Bullish MACD crossover",
            "macd_bearish": "Bearish MACD crossover",
            "above_sma_200": "Above 200-day SMA",
            "above_sma_50": "Above 50-day SMA",
            "above_sma_20": "Above 20-day SMA",
            "below_sma_200": "Below 200-day SMA",
            "below_sma_50": "Below 50-day SMA",
            "below_sma_20": "Below 20-day SMA",
            "volume_spike": "High volume (> 2x average)",
        }

        for key, val in filters.items():
            if key in filter_labels and val:
                parts.append(filter_labels[key])
            elif key == "price_change_min":
                parts.append(f"Up more than {val}%")
            elif key == "price_change_max":
                parts.append(f"Down more than {abs(val)}%")

        if sector:
            parts.append(f"Sector: {sector}")

        return " + ".join(parts) if parts else "General search"


nl_search_service = NLSearchService()
