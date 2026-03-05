import re
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import feedparser
from curl_cffi import requests as cffi_requests

from app.utils.cache import cache
from app.utils.helpers import now_ist
from app.utils.yahoo_api import yahoo_quote
from app.config import (
    CACHE_TTL_GLOBAL, GLOBAL_MARKET_SYMBOLS,
    POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS,
)

logger = logging.getLogger(__name__)


def _fetch_rss(url: str) -> str:
    """Fetch RSS XML using curl_cffi to bypass cloud IP blocks."""
    try:
        resp = cffi_requests.get(url, impersonate="chrome", timeout=10)
        return resp.text
    except Exception as e:
        logger.warning(f"curl_cffi RSS fetch failed for {url}: {e}")
        return ""


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

# Big event keywords with inherent sentiment direction.
# Positive value = bullish for markets, negative = bearish.
# War/conflict/crisis = bearish, rate cut/easing = bullish.
BIG_EVENT_SCORE = {
    # War / geopolitical (strongly bearish)
    "war": -80, "invasion": -80, "airstrike": -90, "missile": -85,
    "military": -60, "attack": -75, "conflict": -70, "escalation": -70,
    "nuclear": -95, "terrorism": -80, "tensions": -50,
    "sanctions": -60, "embargo": -60,
    "india pakistan": -85, "india china": -70, "israel": -50,
    "iran": -50, "russia ukraine": -60, "nato": -40, "border": -40,
    # Bearish policy
    "tariff": -50, "trade war": -65, "rate hike": -55,
    "debt ceiling": -60,
    # Economy bearish
    "recession": -80, "inflation": -45, "stagflation": -75,
    "crash": -90, "crisis": -85, "collapse": -90, "default": -80,
    # Commodities shock (bearish for India — oil importer)
    "oil shock": -70, "crude surge": -55, "crude crash": -40,
    # Black swan
    "pandemic": -90, "lockdown": -70, "black swan": -90,
    "circuit breaker": -80, "capitulation": -75,
    # Bullish events
    "ceasefire": 60, "rate cut": 55, "easing": 50,
    "peace": 60, "deal": 30, "recovery": 40, "stimulus": 50,
    "fed": -30, "federal reserve": -30, "rbi": -20,
    "monetary policy": -20, "opec": -30,
    "geopolitical": -40, "defence": -30, "cpi": -25, "gdp": -15,
}

# Flat list for quick "is this a big event?" check
BIG_EVENT_KEYWORDS = list(BIG_EVENT_SCORE.keys())


class GlobalMarketService:
    def get_global_signal(self) -> dict:
        cache_key = "global_market_signal"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        markets = []
        total_weighted_score = 0
        total_weight = 0

        def _fetch_one(name, symbol):
            q = yahoo_quote(symbol)
            if not q:
                raise ValueError(f"No data for {symbol}")
            return name, symbol, q["ltp"], q["close"]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_fetch_one, name, sym): name
                for name, sym in GLOBAL_MARKET_SYMBOLS.items()
            }
            for future in futures:
                name = futures[future]
                try:
                    name, symbol, current_price, prev_close = future.result(timeout=10)

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
                    logger.warning(f"Failed to fetch {name}: {e}")
                    continue

        price_score = (total_weighted_score / total_weight) if total_weight > 0 else 0
        price_score = max(-100, min(100, round(price_score, 2)))

        # Fetch and score global news
        news_result = self._fetch_global_news()
        news_score = news_result["score"]
        news_magnitude = news_result["magnitude"]
        headlines = news_result["headlines"]

        # Blend: normally price-led, but news dominates on big events
        if news_magnitude >= 60:
            # Big event — news leads (60% news, 40% price)
            composite = price_score * 0.4 + news_score * 0.6
        elif news_magnitude >= 30:
            # Moderate event — balanced
            composite = price_score * 0.5 + news_score * 0.5
        else:
            # Quiet — price leads (70% price, 30% news)
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

        def _fetch_feed(feed_url):
            entries = []
            try:
                feed = feedparser.parse(_fetch_rss(feed_url))
                for entry in feed.entries[:15]:
                    title = entry.get("title", "").strip()
                    if not title:
                        continue
                    pub_time = None
                    if entry.get("published_parsed"):
                        try:
                            pub_time = datetime(*entry.published_parsed[:6])
                            if now_ist().replace(tzinfo=None) - pub_time > timedelta(days=3):
                                continue
                        except Exception:
                            pass
                    entries.append({
                        "title": title,
                        "link": entry.get("link", ""),
                        "pub_time": pub_time,
                    })
            except Exception:
                pass
            return entries

        with ThreadPoolExecutor(max_workers=4) as executor:
            feed_futures = [executor.submit(_fetch_feed, url) for url in GLOBAL_NEWS_FEEDS]
            for future in feed_futures:
                try:
                    for h in future.result(timeout=10):
                        if h["title"] not in seen_titles:
                            seen_titles.add(h["title"])
                            all_headlines.append(h)
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

            # 1. Stock-market keyword score
            pos = sum(1 for w in words if w in POSITIVE_KEYWORDS)
            neg = sum(1 for w in words if w in NEGATIVE_KEYWORDS)
            if pos + neg == 0:
                keyword_score = 0
            else:
                keyword_score = (pos - neg) / (pos + neg) * 100

            # 2. Big event score — war/crisis/rate hike carry their OWN sentiment
            is_big = False
            event_score = 0
            for kw, score in BIG_EVENT_SCORE.items():
                if kw in title_lower:
                    is_big = True
                    # Take the strongest event match
                    if abs(score) > abs(event_score):
                        event_score = score

            if is_big:
                big_event_count += 1
                # Big events dominate: use event score, but blend if keywords also present
                if keyword_score != 0:
                    headline_score = event_score * 0.7 + keyword_score * 0.3
                else:
                    headline_score = event_score
            else:
                headline_score = keyword_score

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
