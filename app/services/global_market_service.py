import logging
import yfinance as yf
from app.utils.cache import cache
from app.config import CACHE_TTL_GLOBAL, GLOBAL_MARKET_SYMBOLS

logger = logging.getLogger(__name__)

INFLUENCE_WEIGHTS = {
    "S&P 500": 1.5,
    "NASDAQ": 1.2,
    "Dow Jones": 1.0,
    "Nikkei 225": 0.8,
    "India VIX": 1.3,
    "USD/INR": 1.0,
    "Crude Oil": 0.9,
    "Gold": 0.5,
}

# These indicators are inverse-correlated with Indian market
INVERSE_INDICATORS = {
    "India VIX": 10,    # VIX up = bearish
    "USD/INR": 30,      # Rupee weakening = bearish
    "Crude Oil": 10,    # Oil up = bearish for oil-importing India
}


class GlobalMarketService:
    def get_global_signal(self) -> dict:
        cache_key = "global_market_signal"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        markets = []
        total_weighted_score = 0
        total_weight = 0

        for name, symbol in GLOBAL_MARKET_SYMBOLS.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                current_price = info.last_price
                prev_close = info.previous_close

                if current_price and prev_close and prev_close > 0:
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                else:
                    change_pct = 0

                direction = "up" if change_pct > 0 else ("down" if change_pct < 0 else "flat")
                markets.append({
                    "name": name, "symbol": symbol,
                    "price": round(current_price, 2) if current_price else 0,
                    "change_pct": round(change_pct, 2),
                    "direction": direction,
                })

                weight = INFLUENCE_WEIGHTS.get(name, 1.0)
                if name in INVERSE_INDICATORS:
                    market_score = -change_pct * INVERSE_INDICATORS[name]
                else:
                    market_score = change_pct * 20

                market_score = max(-100, min(100, market_score))
                total_weighted_score += market_score * weight
                total_weight += weight

            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({symbol}): {e}")
                continue

        composite = (total_weighted_score / total_weight) if total_weight > 0 else 0
        composite = max(-100, min(100, round(composite, 2)))

        result = {"score": composite, "markets": markets}
        cache.set(cache_key, result, CACHE_TTL_GLOBAL)
        return result


global_market_service = GlobalMarketService()
