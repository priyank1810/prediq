import logging
from app.utils.cache import cache
from app.config import SECTOR_MAP, CACHE_TTL_SECTOR_STRENGTH, SECTOR_EVENT_MODIFIERS

logger = logging.getLogger(__name__)

CACHE_TTL_SECTOR = 300  # 5 minutes


class SectorService:
    """Computes sector-level performance data for the heat map and relative strength."""

    def get_sector_strength(self, symbol: str) -> dict:
        """Compute sector-relative strength for a stock."""
        default = {"available": False, "score": 0}

        # Find sector
        sector = None
        sector_symbols = []
        for s, syms in SECTOR_MAP.items():
            if symbol in syms:
                sector = s
                sector_symbols = syms
                break

        if not sector or len(sector_symbols) < 2:
            return default

        cache_key = f"sector_strength:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from app.services.data_fetcher import data_fetcher
            quotes = data_fetcher.get_bulk_quotes(sector_symbols)
            quote_map = {q["symbol"]: q for q in quotes if q.get("symbol")}

            stock_quote = quote_map.get(symbol)
            if not stock_quote or stock_quote.get("pct_change") is None:
                return default

            stock_change = stock_quote["pct_change"]

            # Compute sector average (excluding current stock)
            sector_changes = []
            for s in sector_symbols:
                q = quote_map.get(s)
                if q and q.get("pct_change") is not None and s != symbol:
                    sector_changes.append(q["pct_change"])

            if not sector_changes:
                return default

            sector_avg = sum(sector_changes) / len(sector_changes)
            relative_pct = stock_change - sector_avg
            score = max(-100, min(100, relative_pct * 50))

            result = {
                "available": True,
                "score": round(score, 2),
                "sector": sector,
                "stock_change": round(stock_change, 2),
                "sector_avg_change": round(sector_avg, 2),
                "relative_pct": round(relative_pct, 2),
            }
            cache.set(cache_key, result, CACHE_TTL_SECTOR_STRENGTH)
            return result
        except Exception as e:
            logger.warning(f"Sector strength failed for {symbol}: {e}")
            return default

    def get_sector_for_symbol(self, symbol: str):
        """Return the sector name for a symbol, or None if not mapped."""
        for sector, symbols in SECTOR_MAP.items():
            if symbol in symbols:
                return sector
        return None

    def get_sector_adjusted_scores(self, symbol: str, sentiment_score: float,
                                   global_score: float, global_data: dict) -> dict:
        """Adjust sentiment and global scores based on sector-event interaction.

        Returns dict with adjusted scores, sector, modifier, and active events.
        """
        result = {
            "sentiment_score": sentiment_score,
            "global_score": global_score,
            "sector": None,
            "modifier_applied": 1.0,
            "active_events": {},
        }

        try:
            sector = self.get_sector_for_symbol(symbol)
            if not sector:
                return result

            result["sector"] = sector

            active_events = global_data.get("active_events", {})
            news_magnitude = global_data.get("news_magnitude", 0)

            if not active_events or news_magnitude < 20:
                return result

            result["active_events"] = active_events

            # Weighted-average modifier across active categories (weight = headline count)
            total_weight = 0
            weighted_modifier = 0.0
            for cat, info in active_events.items():
                cat_modifiers = SECTOR_EVENT_MODIFIERS.get(cat, {})
                mod = cat_modifiers.get(sector, 1.0)
                count = info.get("count", 1)
                weighted_modifier += mod * count
                total_weight += count

            if total_weight == 0:
                return result

            modifier = weighted_modifier / total_weight
            result["modifier_applied"] = round(modifier, 2)

            # Adjust the news portion of global_score
            news_score = global_data.get("news_score", 0)
            price_score = global_data.get("price_score", 0)
            adjusted_news = news_score * modifier

            # Re-blend using same magnitude-based ratios as global_market_service
            if news_magnitude >= 60:
                adjusted_global = price_score * 0.4 + adjusted_news * 0.6
            elif news_magnitude >= 30:
                adjusted_global = price_score * 0.5 + adjusted_news * 0.5
            else:
                adjusted_global = price_score * 0.7 + adjusted_news * 0.3

            result["global_score"] = max(-100, min(100, round(adjusted_global, 2)))

            # Adjust sentiment with dampened modifier (50% effect)
            dampened = 1.0 + (modifier - 1.0) * 0.5
            result["sentiment_score"] = max(-100, min(100, round(sentiment_score * dampened, 2)))

        except Exception as e:
            logger.warning(f"Sector adjustment failed for {symbol}: {e}")

        return result

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
