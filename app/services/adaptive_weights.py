import logging
import math
import time
import numpy as np
from app.config import (
    SIGNAL_WEIGHT_TECHNICAL, SIGNAL_WEIGHT_SENTIMENT, SIGNAL_WEIGHT_GLOBAL,
    SECTOR_MAP, ADAPTIVE_WEIGHTS_DECAY_HALFLIFE_DAYS,
    ADAPTIVE_WEIGHTS_MIN_SIGNALS, ADAPTIVE_WEIGHTS_CACHE_TTL,
)

logger = logging.getLogger(__name__)


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

                # Compute per-component accuracy with exponential decay
                halflife_sec = ADAPTIVE_WEIGHTS_DECAY_HALFLIFE_DAYS * 86400
                ln2 = math.log(2)
                now = time.time()

                tech_weighted_correct = 0.0
                sent_weighted_correct = 0.0
                glob_weighted_correct = 0.0
                total_weight = 0.0

                for sig in signals:
                    # Compute decay weight based on signal age
                    try:
                        age_seconds = now - sig.created_at.timestamp()
                    except Exception:
                        age_seconds = 0  # Timezone-naive fallback
                    w = math.exp(-ln2 / halflife_sec * max(0, age_seconds))

                    correct = sig.was_correct
                    if correct:
                        if sig.technical_score > 0 and sig.composite_score > 0:
                            tech_weighted_correct += w
                        elif sig.technical_score < 0 and sig.composite_score < 0:
                            tech_weighted_correct += w
                        elif abs(sig.technical_score) < 5:
                            tech_weighted_correct += 0.5 * w

                        if sig.sentiment_score > 0 and sig.composite_score > 0:
                            sent_weighted_correct += w
                        elif sig.sentiment_score < 0 and sig.composite_score < 0:
                            sent_weighted_correct += w
                        elif abs(sig.sentiment_score) < 5:
                            sent_weighted_correct += 0.5 * w

                        if sig.global_score > 0 and sig.composite_score > 0:
                            glob_weighted_correct += w
                        elif sig.global_score < 0 and sig.composite_score < 0:
                            glob_weighted_correct += w
                        elif abs(sig.global_score) < 5:
                            glob_weighted_correct += 0.5 * w

                    total_weight += w

                # Guard: if effective sample too small, fall back to static
                effective_sample_size = total_weight
                if effective_sample_size < ADAPTIVE_WEIGHTS_MIN_SIGNALS / 2:
                    fallback["effective_sample_size"] = round(effective_sample_size, 1)
                    fallback["decay_halflife_days"] = ADAPTIVE_WEIGHTS_DECAY_HALFLIFE_DAYS
                    self._cache[cache_key] = (time.time(), fallback)
                    return fallback

                tech_acc = tech_weighted_correct / total_weight
                sent_acc = sent_weighted_correct / total_weight
                glob_acc = glob_weighted_correct / total_weight

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
                    "sample_size": len(signals),
                    "effective_sample_size": round(effective_sample_size, 1),
                    "decay_halflife_days": ADAPTIVE_WEIGHTS_DECAY_HALFLIFE_DAYS,
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
