import logging
from app.utils.cache import cache
from app.config import SECTOR_MAP

logger = logging.getLogger(__name__)

CACHE_TTL_SECTOR = 300  # 5 minutes


class SectorService:
    """Computes sector-level performance data for the heat map."""

    def get_heatmap(self) -> list:
        """Get sector performance data with average % change per sector."""
        cache_key = "sector:heatmap"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        data = self._compute_heatmap()
        cache.set(cache_key, data, CACHE_TTL_SECTOR)
        return data

    def _compute_heatmap(self) -> list:
        from app.services.data_fetcher import data_fetcher

        sectors = []
        all_symbols = []
        symbol_to_sector = {}

        for sector, symbols in SECTOR_MAP.items():
            for s in symbols:
                all_symbols.append(s)
                symbol_to_sector[s] = sector

        # Bulk fetch quotes
        try:
            quotes = data_fetcher.get_bulk_quotes(all_symbols)
        except Exception as e:
            logger.warning(f"Sector heatmap bulk fetch failed: {e}")
            return []

        quote_map = {q["symbol"]: q for q in quotes if q.get("symbol")}

        for sector, symbols in SECTOR_MAP.items():
            changes = []
            stocks = []
            for s in symbols:
                q = quote_map.get(s)
                if q and q.get("pct_change") is not None:
                    changes.append(q["pct_change"])
                    stocks.append({
                        "symbol": s,
                        "ltp": q.get("ltp", 0),
                        "pct_change": q["pct_change"],
                    })

            if changes:
                avg_change = sum(changes) / len(changes)
            else:
                avg_change = 0

            sectors.append({
                "sector": sector,
                "avg_change": round(avg_change, 2),
                "stock_count": len(stocks),
                "stocks": sorted(stocks, key=lambda x: x.get("pct_change", 0), reverse=True),
            })

        # Sort by average change descending
        sectors.sort(key=lambda x: x["avg_change"], reverse=True)
        return sectors


sector_service = SectorService()
