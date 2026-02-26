import logging
from app.services.data_fetcher import data_fetcher
from app.services.indicator_service import indicator_service
from app.services.sentiment_service import sentiment_service
from app.services.global_market_service import global_market_service
from app.utils.helpers import now_ist, is_market_open
from app.config import SIGNAL_WEIGHT_TECHNICAL, SIGNAL_WEIGHT_SENTIMENT, SIGNAL_WEIGHT_GLOBAL

logger = logging.getLogger(__name__)


class SignalService:
    def get_signal(self, symbol: str) -> dict:
        # 1. Intraday data
        try:
            intraday_df = data_fetcher.get_intraday_data(symbol, period="5d", interval="15m")
        except Exception as e:
            logger.error(f"Intraday data failed for {symbol}: {e}")
            intraday_df = None

        # 2. Technical score
        if intraday_df is not None and not intraday_df.empty:
            tech_result = indicator_service.compute_intraday_indicators(intraday_df)
            technical_score = tech_result["score"]
            tech_details = tech_result["details"]
        else:
            technical_score = 0
            tech_details = {}

        # 3. Sentiment score
        try:
            sent_result = sentiment_service.get_sentiment(symbol)
            sentiment_score = sent_result["score"]
        except Exception as e:
            logger.warning(f"Sentiment failed for {symbol}: {e}")
            sent_result = {"score": 0, "headline_count": 0, "positive_count": 0,
                           "negative_count": 0, "neutral_count": 0, "headlines": []}
            sentiment_score = 0

        # 4. Global market score
        try:
            global_result = global_market_service.get_global_signal()
            global_score = global_result["score"]
        except Exception as e:
            logger.warning(f"Global market failed: {e}")
            global_result = {"score": 0, "markets": []}
            global_score = 0

        # 5. Weighted composite
        composite = (
            SIGNAL_WEIGHT_TECHNICAL * technical_score +
            SIGNAL_WEIGHT_SENTIMENT * sentiment_score +
            SIGNAL_WEIGHT_GLOBAL * global_score
        )
        composite = max(-100, min(100, round(composite, 2)))

        # 6. Direction and confidence
        if composite > 5:
            direction = "BULLISH"
        elif composite < -5:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        confidence = min(100, round(abs(composite), 2))

        # 7. Intraday candles for chart
        candles = []
        if intraday_df is not None and not intraday_df.empty:
            for _, row in intraday_df.tail(52).iterrows():
                candles.append({
                    "time": row.get("datetime_str", ""),
                    "open": round(float(row["open"]), 2),
                    "high": round(float(row["high"]), 2),
                    "low": round(float(row["low"]), 2),
                    "close": round(float(row["close"]), 2),
                    "volume": int(row["volume"]),
                })

        return {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "composite_score": composite,
            "timestamp": now_ist().isoformat(),
            "market_open": is_market_open(),
            "technical": {
                "score": technical_score,
                "weight": SIGNAL_WEIGHT_TECHNICAL,
                "details": tech_details,
            },
            "sentiment": {
                "score": sentiment_score,
                "weight": SIGNAL_WEIGHT_SENTIMENT,
                "headline_count": sent_result.get("headline_count", 0),
                "positive_count": sent_result.get("positive_count", 0),
                "negative_count": sent_result.get("negative_count", 0),
                "neutral_count": sent_result.get("neutral_count", 0),
                "headlines": sent_result.get("headlines", []),
            },
            "global_market": {
                "score": global_score,
                "weight": SIGNAL_WEIGHT_GLOBAL,
                "markets": global_result.get("markets", []),
            },
            "intraday_candles": candles,
        }


signal_service = SignalService()
