import logging
import numpy as np
from app.utils.cache import cache

logger = logging.getLogger(__name__)

CACHE_TTL_MOOD = 300  # 5 minutes


class MarketMoodService:
    """Computes Market Mood Score (0-100) from 4 components:
    - Aggregate sentiment across top stocks (25%)
    - Technical breadth: % of Nifty50 above 20-day SMA (25%)
    - India VIX inverse mapping (25%)
    - FII/DII net flow direction (25%)

    Labels: 0-20 Extreme Fear, 20-40 Fear, 40-60 Neutral, 60-80 Greed, 80-100 Extreme Greed
    """

    LABELS = {
        (0, 20): "Extreme Fear",
        (20, 40): "Fear",
        (40, 60): "Neutral",
        (60, 80): "Greed",
        (80, 100): "Extreme Greed",
    }

    def get_mood(self) -> dict:
        cache_key = "market_mood"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._compute()
        cache.set(cache_key, result, CACHE_TTL_MOOD)
        return result

    def _compute(self) -> dict:
        components = {}

        # 1. Aggregate sentiment (25%)
        sentiment_score = self._get_sentiment_component()
        components["sentiment"] = {"score": sentiment_score, "weight": 0.25}

        # 2. Technical breadth (25%)
        breadth_score = self._get_breadth_component()
        components["breadth"] = {"score": breadth_score, "weight": 0.25}

        # 3. India VIX inverse (25%)
        vix_score = self._get_vix_component()
        components["vix"] = {"score": vix_score, "weight": 0.25}

        # 4. FII/DII flow (25%)
        flow_score = self._get_flow_component()
        components["fii_dii"] = {"score": flow_score, "weight": 0.25}

        # Weighted composite
        mood_score = (
            sentiment_score * 0.25 +
            breadth_score * 0.25 +
            vix_score * 0.25 +
            flow_score * 0.25
        )
        mood_score = max(0, min(100, round(mood_score, 1)))

        label = "Neutral"
        for (lo, hi), lbl in self.LABELS.items():
            if lo <= mood_score < hi:
                label = lbl
                break
        if mood_score >= 100:
            label = "Extreme Greed"

        return {
            "score": mood_score,
            "label": label,
            "components": components,
        }

    def _get_sentiment_component(self) -> float:
        """Aggregate FinBERT/keyword sentiment across top 20 stocks -> 0-100 scale."""
        try:
            from app.services.sentiment_service import sentiment_service
            from app.config import POPULAR_STOCKS

            scores = []
            for symbol in POPULAR_STOCKS[:10]:  # Sample 10 stocks for speed
                try:
                    sent = sentiment_service.get_sentiment(symbol)
                    scores.append(sent.get("score", 0))
                except Exception:
                    pass

            if scores:
                # Sentiment score is -100 to +100, map to 0-100
                avg = np.mean(scores)
                return float(np.clip((avg + 100) / 2, 0, 100))
        except Exception as e:
            logger.warning(f"Sentiment component failed: {e}")
        return 50.0

    def _get_breadth_component(self) -> float:
        """% of Nifty50 above their 20-day SMA -> 0-100."""
        try:
            from app.services.data_fetcher import data_fetcher
            from app.config import NIFTY_50_SYMBOLS
            import ta

            above_sma = 0
            total = 0

            for symbol in NIFTY_50_SYMBOLS[:20]:  # Sample 20 for speed
                try:
                    df = data_fetcher.get_historical_data(symbol, period="3mo")
                    if df is not None and len(df) >= 25:
                        close = df["close"]
                        sma20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
                        if float(close.iloc[-1]) > float(sma20.iloc[-1]):
                            above_sma += 1
                        total += 1
                except Exception:
                    pass

            if total > 0:
                return float(above_sma / total * 100)
        except Exception as e:
            logger.warning(f"Breadth component failed: {e}")
        return 50.0

    def _get_vix_component(self) -> float:
        """India VIX inverse mapping -> 0-100 (low VIX = high greed)."""
        try:
            import yfinance as yf

            vix = yf.Ticker("^INDIAVIX")
            hist = vix.history(period="5d")
            if len(hist) > 0:
                current_vix = float(hist["Close"].iloc[-1])
                # VIX typically ranges 10-40 for India
                # Low VIX (10) = Extreme Greed (100), High VIX (40) = Extreme Fear (0)
                score = np.clip((40 - current_vix) / 30 * 100, 0, 100)
                return float(score)
        except Exception as e:
            logger.warning(f"VIX component failed: {e}")
        return 50.0

    def _get_flow_component(self) -> float:
        """FII/DII net flow direction -> 0-100."""
        try:
            from app.services.fii_dii_service import fii_dii_service

            signal = fii_dii_service.get_net_flow_signal()
            # signal is -1 to +1, map to 0-100
            return float(np.clip((signal + 1) / 2 * 100, 0, 100))
        except Exception as e:
            logger.warning(f"FII/DII flow component failed: {e}")
        return 50.0


market_mood_service = MarketMoodService()
