import re
import math
import logging
from datetime import datetime, timedelta
from urllib.parse import quote

import feedparser

from app.utils.cache import cache
from app.config import CACHE_TTL_SENTIMENT, NEWS_RSS_SOURCES, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS

logger = logging.getLogger(__name__)


class SentimentService:
    def __init__(self):
        self._finbert = None

    def _get_finbert(self):
        """Lazy-load FinBERT scorer."""
        if self._finbert is None:
            try:
                from app.config import FINBERT_ENABLED
                if FINBERT_ENABLED:
                    from app.ai.finbert_model import finbert_scorer
                    self._finbert = finbert_scorer
            except Exception as e:
                logger.info(f"FinBERT not available, using keyword scoring: {e}")
                self._finbert = False  # Mark as unavailable
        return self._finbert if self._finbert else None

    def get_sentiment(self, symbol: str) -> dict:
        cache_key = f"sentiment:{symbol}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        headlines = self._fetch_headlines(symbol)
        result = self._score_headlines(headlines, symbol)
        cache.set(cache_key, result, CACHE_TTL_SENTIMENT)
        return result

    def _fetch_headlines(self, symbol: str) -> list[dict]:
        all_headlines = []
        seen_titles = set()

        # Original Google News RSS sources
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
                    pub_time = None

                    if entry.get("published_parsed"):
                        try:
                            pub_time = datetime(*entry.published_parsed[:6])
                            # Extended to 7 days for time-decay weighting
                            if datetime.now() - pub_time > timedelta(days=7):
                                continue
                        except Exception:
                            pass

                    all_headlines.append({
                        "title": title,
                        "published": published,
                        "link": link,
                        "pub_time": pub_time,
                        "source": "google_news",
                    })
            except Exception as e:
                logger.warning(f"RSS fetch failed for {symbol}: {e}")
                continue

        # MoneyControl RSS
        try:
            mc_url = f"https://www.moneycontrol.com/rss/{quote(symbol)}.xml"
            feed = feedparser.parse(mc_url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                pub_time = None
                if entry.get("published_parsed"):
                    try:
                        pub_time = datetime(*entry.published_parsed[:6])
                    except Exception:
                        pass
                all_headlines.append({
                    "title": title,
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                    "pub_time": pub_time,
                    "source": "moneycontrol",
                })
        except Exception:
            pass

        # Economic Times RSS
        try:
            et_url = "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
            feed = feedparser.parse(et_url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "").strip()
                if not title or title in seen_titles:
                    continue
                # Only include if symbol mentioned
                if symbol.lower() not in title.lower():
                    continue
                seen_titles.add(title)
                pub_time = None
                if entry.get("published_parsed"):
                    try:
                        pub_time = datetime(*entry.published_parsed[:6])
                    except Exception:
                        pass
                all_headlines.append({
                    "title": title,
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                    "pub_time": pub_time,
                    "source": "economic_times",
                })
        except Exception:
            pass

        # Finnhub API (if key configured)
        try:
            from app.config import FINNHUB_API_KEY
            if FINNHUB_API_KEY:
                import aiohttp
                import json
                from urllib.request import urlopen
                today = datetime.now().strftime("%Y-%m-%d")
                week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}.NS&from={week_ago}&to={today}&token={FINNHUB_API_KEY}"
                try:
                    resp = urlopen(url, timeout=5)
                    news = json.loads(resp.read())
                    for item in news[:10]:
                        title = item.get("headline", "").strip()
                        if not title or title in seen_titles:
                            continue
                        seen_titles.add(title)
                        pub_time = None
                        if item.get("datetime"):
                            try:
                                pub_time = datetime.fromtimestamp(item["datetime"])
                            except Exception:
                                pass
                        all_headlines.append({
                            "title": title,
                            "published": str(pub_time) if pub_time else "",
                            "link": item.get("url", ""),
                            "pub_time": pub_time,
                            "source": "finnhub",
                        })
                except Exception:
                    pass
        except Exception:
            pass

        return all_headlines[:30]

    def _time_decay_weight(self, pub_time, symbol: str, title: str) -> float:
        """Apply exponential time-decay weighting.
        - Company-specific (contains symbol): half-life 24h
        - Sector-related: half-life 72h
        - Market-wide: half-life 168h
        """
        if pub_time is None:
            return 1.0  # No time info, assume recent

        hours_ago = (datetime.now() - pub_time).total_seconds() / 3600
        if hours_ago < 0:
            hours_ago = 0

        # Determine headline type
        if symbol.lower() in title.lower():
            half_life = 24.0  # Company-specific
        elif any(kw in title.lower() for kw in ["sector", "industry", "nifty it", "nifty bank", "pharma"]):
            half_life = 72.0  # Sector-related
        else:
            half_life = 168.0  # Market-wide

        decay = math.exp(-0.693 * hours_ago / half_life)  # 0.693 = ln(2)
        return decay

    def _score_headlines(self, headlines: list[dict], symbol: str = "") -> dict:
        if not headlines:
            return {
                "score": 0, "headline_count": 0,
                "positive_count": 0, "negative_count": 0, "neutral_count": 0,
                "headlines": [],
            }

        # Try FinBERT first
        finbert = self._get_finbert()
        use_finbert = finbert is not None

        if use_finbert:
            try:
                titles = [h["title"] for h in headlines]
                finbert_scores = finbert.score_headlines(titles)
            except Exception as e:
                logger.warning(f"FinBERT scoring failed, falling back to keywords: {e}")
                use_finbert = False

        scored = []
        total_weighted_score = 0
        total_weight = 0
        pos_count = neg_count = neu_count = 0

        for idx, h in enumerate(headlines):
            # Get time-decay weight
            weight = self._time_decay_weight(h.get("pub_time"), symbol, h["title"])

            if use_finbert:
                # FinBERT score: -1 to +1, scale to -100 to +100
                headline_score = finbert_scores[idx] * 100
            else:
                # Keyword matching fallback
                words = re.findall(r'\b[a-z]+\b', h["title"].lower())
                pos_matches = sum(1 for w in words if w in POSITIVE_KEYWORDS)
                neg_matches = sum(1 for w in words if w in NEGATIVE_KEYWORDS)

                if pos_matches + neg_matches == 0:
                    headline_score = 0
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

            total_weighted_score += headline_score * weight
            total_weight += weight

            scored.append({
                "title": h["title"],
                "sentiment": label,
                "score": round(headline_score, 2),
                "published": h.get("published", ""),
                "link": h.get("link", ""),
                "source": h.get("source", ""),
                "decay_weight": round(weight, 3),
            })

        avg = total_weighted_score / total_weight if total_weight > 0 else 0
        composite = max(-100, min(100, round(avg, 2)))

        return {
            "score": composite,
            "headline_count": len(headlines),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "headlines": scored,
            "scoring_method": "finbert" if use_finbert else "keyword",
        }


sentiment_service = SentimentService()
