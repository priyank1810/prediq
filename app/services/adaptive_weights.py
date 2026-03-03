import logging
import time
import numpy as np
from app.config import SIGNAL_WEIGHT_TECHNICAL, SIGNAL_WEIGHT_SENTIMENT, SIGNAL_WEIGHT_GLOBAL, SECTOR_MAP

logger = logging.getLogger(__name__)

ADAPTIVE_WEIGHTS_MIN_SIGNALS = 30
ADAPTIVE_WEIGHTS_CACHE_TTL = 600  # 10 minutes


class AdaptiveWeightService:
    def __init__(self):
        self._cache = {}  # key -> (timestamp, result)

    def _get_sector(self, symbol: str) -> str:
        for sector, symbols in SECTOR_MAP.items():
            if symbol in symbols:
                return sector
        return "default"

    def get_weights(self, symbol: str) -> dict:
        """Get adaptive weights based on historical signal accuracy.

        Returns dict with keys: adapted, weights, sample_size, component_accuracies
        """
        sector = self._get_sector(symbol)
        cache_key = f"{sector}"

        # Check cache
        if cache_key in self._cache:
            ts, result = self._cache[cache_key]
            if time.time() - ts < ADAPTIVE_WEIGHTS_CACHE_TTL:
                return result

        # Default fallback
        fallback = {
            "adapted": False,
            "weights": {
                "technical": SIGNAL_WEIGHT_TECHNICAL,
                "sentiment": SIGNAL_WEIGHT_SENTIMENT,
                "global": SIGNAL_WEIGHT_GLOBAL,
            },
            "sample_size": 0,
            "component_accuracies": {},
        }

        try:
            from app.database import SessionLocal
            from app.models import SignalLog

            db = SessionLocal()
            try:
                # Query last 100 signals with correctness validation
                query = db.query(SignalLog).filter(
                    SignalLog.was_correct.isnot(None)
                )
                if sector != "default":
                    query = query.filter(SignalLog.sector == sector)

                signals = query.order_by(SignalLog.created_at.desc()).limit(100).all()

                if len(signals) < ADAPTIVE_WEIGHTS_MIN_SIGNALS:
                    self._cache[cache_key] = (time.time(), fallback)
                    return fallback

                # Compute per-component accuracy
                tech_correct = 0
                sent_correct = 0
                glob_correct = 0
                total = len(signals)

                for sig in signals:
                    correct = sig.was_correct
                    # Check if each component agreed with the correct outcome
                    # If signal was correct, components with same sign as composite agreed
                    # If signal was incorrect, components with same sign were wrong
                    if correct:
                        if sig.technical_score > 0 and sig.composite_score > 0:
                            tech_correct += 1
                        elif sig.technical_score < 0 and sig.composite_score < 0:
                            tech_correct += 1
                        elif abs(sig.technical_score) < 5:
                            tech_correct += 0.5

                        if sig.sentiment_score > 0 and sig.composite_score > 0:
                            sent_correct += 1
                        elif sig.sentiment_score < 0 and sig.composite_score < 0:
                            sent_correct += 1
                        elif abs(sig.sentiment_score) < 5:
                            sent_correct += 0.5

                        if sig.global_score > 0 and sig.composite_score > 0:
                            glob_correct += 1
                        elif sig.global_score < 0 and sig.composite_score < 0:
                            glob_correct += 1
                        elif abs(sig.global_score) < 5:
                            glob_correct += 0.5

                tech_acc = tech_correct / total
                sent_acc = sent_correct / total
                glob_acc = glob_correct / total

                # Softmax-like weighting with temperature=2.0
                temp = 2.0
                accs = np.array([tech_acc, sent_acc, glob_acc])
                exp_accs = np.exp(accs / temp)
                weights = exp_accs / exp_accs.sum()

                # Apply floors
                w_tech = max(0.20, float(weights[0]))
                w_sent = max(0.10, float(weights[1]))
                w_glob = max(0.05, float(weights[2]))

                # Renormalize
                total_w = w_tech + w_sent + w_glob
                w_tech /= total_w
                w_sent /= total_w
                w_glob /= total_w

                result = {
                    "adapted": True,
                    "weights": {
                        "technical": round(w_tech, 4),
                        "sentiment": round(w_sent, 4),
                        "global": round(w_glob, 4),
                    },
                    "sample_size": total,
                    "component_accuracies": {
                        "technical": round(tech_acc, 4),
                        "sentiment": round(sent_acc, 4),
                        "global": round(glob_acc, 4),
                    },
                }
                self._cache[cache_key] = (time.time(), result)
                return result
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Adaptive weights computation failed: {e}")
            self._cache[cache_key] = (time.time(), fallback)
            return fallback


adaptive_weight_service = AdaptiveWeightService()
