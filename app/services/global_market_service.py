import re
import logging
from datetime import datetime, timedelta

import feedparser
import yfinance as yf

from app.utils.cache import cache
from app.config import (
    CACHE_TTL_GLOBAL, GLOBAL_MARKET_SYMBOLS,
    POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS,
)

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

# RSS feeds for global macro news that affects Indian markets
GLOBAL_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=global+markets+today&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=US+Fed+interest+rate+economy&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=crude+oil+price+gold+commodity&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=war+geopolitical+sanctions+conflict+market+impact&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=India+Pakistan+China+geopolitics+defence&hl=en-IN&gl=IN&ceid=IN:en",
    "https://www.moneycontrol.com/rss/internationalmarkets.xml",
    "https://www.moneycontrol.com/rss/commodities.xml",
    "https://www.moneycontrol.com/rss/latestnews.xml",
]

# Keywords that signal high-impact global events (war, rate hike, recession, etc.)
BIG_EVENT_KEYWORDS = [
    # War / geopolitical
    "war", "invasion", "airstrike", "missile", "military", "attack",
    "conflict", "escalation", "ceasefire", "nuclear", "defence",
    "sanctions", "embargo", "geopolitical", "tensions",
    "india pakistan", "india china", "israel", "iran", "russia ukraine",
    "nato", "border", "terrorism",
    # Trade / policy
    "tariff", "trade war", "rate hike", "rate cut", "fed",
    "federal reserve", "interest rate", "rbi", "monetary policy",
    # Economy
    "recession", "inflation", "cpi", "gdp", "stagflation",
    "crash", "crisis", "collapse", "default", "debt ceiling",
    # Commodities
    "oil shock", "opec", "crude surge", "crude crash",
    # Black swan
    "pandemic", "lockdown", "black swan", "circuit breaker", "capitulation",
]


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

        price_score = (total_weighted_score / total_weight) if total_weight > 0 else 0
        price_score = max(-100, min(100, round(price_score, 2)))

        # Fetch and score global news
        news_result = self._fetch_global_news()
        news_score = news_result["score"]
        news_magnitude = news_result["magnitude"]
        headlines = news_result["headlines"]

        # Blend: price data (70%) + news sentiment (30%)
        composite = price_score * 0.7 + news_score * 0.3
        composite = max(-100, min(100, round(composite, 2)))

        result = {
            "score": composite,
            "price_score": price_score,
            "news_score": news_score,
            "news_magnitude": news_magnitude,
            "markets": markets,
            "headlines": headlines,
        }
        cache.set(cache_key, result, CACHE_TTL_GLOBAL)
        return result

    def _fetch_global_news(self) -> dict:
        """Fetch global macro news and score sentiment.
        Returns score (-100 to +100), magnitude (0-100), and headline list."""
        cache_key = "global_news"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        all_headlines = []
        seen_titles = set()

        for feed_url in GLOBAL_NEWS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:15]:
                    title = entry.get("title", "").strip()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    pub_time = None
                    if entry.get("published_parsed"):
                        try:
                            pub_time = datetime(*entry.published_parsed[:6])
                            if datetime.now() - pub_time > timedelta(days=3):
                                continue
                        except Exception:
                            pass

                    all_headlines.append({
                        "title": title,
                        "link": entry.get("link", ""),
                        "pub_time": pub_time,
                    })
            except Exception:
                pass

        if not all_headlines:
            result = {"score": 0, "magnitude": 0, "headlines": []}
            cache.set(cache_key, result, CACHE_TTL_GLOBAL)
            return result

        # Score headlines
        scored = []
        total_score = 0
        big_event_count = 0

        for h in all_headlines[:25]:
            title_lower = h["title"].lower()
            words = re.findall(r'\b[a-z]+\b', title_lower)

            pos = sum(1 for w in words if w in POSITIVE_KEYWORDS)
            neg = sum(1 for w in words if w in NEGATIVE_KEYWORDS)

            if pos + neg == 0:
                headline_score = 0
            else:
                headline_score = (pos - neg) / (pos + neg) * 100

            # Check if this is a big event
            is_big = any(kw in title_lower for kw in BIG_EVENT_KEYWORDS)
            if is_big:
                big_event_count += 1
                headline_score *= 1.5  # amplify big event sentiment

            headline_score = max(-100, min(100, headline_score))
            total_score += headline_score

            label = "positive" if headline_score > 10 else ("negative" if headline_score < -10 else "neutral")
            scored.append({
                "title": h["title"],
                "link": h.get("link", ""),
                "sentiment": label,
                "score": round(headline_score, 1),
                "big_event": is_big,
            })

        avg_score = total_score / len(scored) if scored else 0
        avg_score = max(-100, min(100, round(avg_score, 2)))

        # Magnitude: how "loud" is global news right now (0-100)
        # Based on: number of big events + overall sentiment intensity
        intensity = abs(avg_score)
        magnitude = min(100, round(big_event_count * 20 + intensity * 0.5))

        result = {
            "score": avg_score,
            "magnitude": magnitude,
            "headlines": scored,
        }
        cache.set(cache_key, result, CACHE_TTL_GLOBAL)
        return result


global_market_service = GlobalMarketService()
