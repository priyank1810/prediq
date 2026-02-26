import re
import logging
from datetime import datetime, timedelta
from urllib.parse import quote

import feedparser

from app.utils.cache import cache
from app.config import CACHE_TTL_SENTIMENT, NEWS_RSS_SOURCES, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS

logger = logging.getLogger(__name__)


class SentimentService:
    def get_sentiment(self, symbol: str) -> dict:
        cache_key = f"sentiment:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        headlines = self._fetch_headlines(symbol)
        result = self._score_headlines(headlines)
        cache.set(cache_key, result, CACHE_TTL_SENTIMENT)
        return result

    def _fetch_headlines(self, symbol: str) -> list[dict]:
        all_headlines = []
        seen_titles = set()

        for url_template in NEWS_RSS_SOURCES:
            try:
                url = url_template.format(symbol=quote(symbol))
                feed = feedparser.parse(url)

                for entry in feed.entries[:15]:
                    title = entry.get("title", "").strip()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)

                    published = entry.get("published", "")
                    link = entry.get("link", "")

                    if entry.get("published_parsed"):
                        try:
                            pub_time = datetime(*entry.published_parsed[:6])
                            if datetime.now() - pub_time > timedelta(hours=24):
                                continue
                        except Exception:
                            pass

                    all_headlines.append({
                        "title": title,
                        "published": published,
                        "link": link,
                    })
            except Exception as e:
                logger.warning(f"RSS fetch failed for {symbol}: {e}")
                continue

        return all_headlines[:20]

    def _score_headlines(self, headlines: list[dict]) -> dict:
        if not headlines:
            return {
                "score": 0, "headline_count": 0,
                "positive_count": 0, "negative_count": 0, "neutral_count": 0,
                "headlines": [],
            }

        scored = []
        total_score = 0
        pos_count = neg_count = neu_count = 0

        for h in headlines:
            words = re.findall(r'\b[a-z]+\b', h["title"].lower())
            pos_matches = sum(1 for w in words if w in POSITIVE_KEYWORDS)
            neg_matches = sum(1 for w in words if w in NEGATIVE_KEYWORDS)

            if pos_matches + neg_matches == 0:
                headline_score = 0
                label = "neutral"
                neu_count += 1
            else:
                raw = (pos_matches - neg_matches) / (pos_matches + neg_matches)
                headline_score = raw * 100
                if headline_score > 10:
                    label = "positive"
                    pos_count += 1
                elif headline_score < -10:
                    label = "negative"
                    neg_count += 1
                else:
                    label = "neutral"
                    neu_count += 1

            total_score += headline_score
            scored.append({
                "title": h["title"],
                "sentiment": label,
                "score": round(headline_score, 2),
                "published": h["published"],
                "link": h["link"],
            })

        avg = total_score / len(headlines)
        composite = max(-100, min(100, round(avg, 2)))

        return {
            "score": composite,
            "headline_count": len(headlines),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "headlines": scored,
        }


sentiment_service = SentimentService()
