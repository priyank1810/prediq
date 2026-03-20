"""Per-stock learning service.

Analyzes historical signal accuracy for each individual stock and builds
a learning profile that captures:
- Which signal components (technical, sentiment, global, fundamental) work best
- Optimal direction threshold based on the stock's volatility and past accuracy
- Which market regimes the stock responds best to
- Component accuracy trends over time

Profiles are stored as JSON in data/stock_profiles/ and loaded at signal time
to customize weights and thresholds per stock.
"""

import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.config import (
    SIGNAL_WEIGHT_TECHNICAL,
    SIGNAL_WEIGHT_SENTIMENT,
    SIGNAL_WEIGHT_FUNDAMENTAL,
    SIGNAL_WEIGHT_GLOBAL,
)

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "stock_profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# Minimum signals before a stock profile becomes active
MIN_SIGNALS_FOR_PROFILE = 10
# How many recent signals to analyze
MAX_SIGNALS_TO_ANALYZE = 200
# Exponential decay half-life for weighting recent signals more
DECAY_HALFLIFE_DAYS = 14
# Profile refresh interval (seconds)
PROFILE_CACHE_TTL = 1800  # 30 minutes


class StockLearner:
    """Learns per-stock signal characteristics from historical data."""

    def __init__(self):
        self._cache = {}  # symbol -> (timestamp, profile)

    def _profile_path(self, symbol: str) -> Path:
        safe = symbol.replace(" ", "_").replace("^", "")
        return PROFILES_DIR / f"{safe}.json"

    def get_profile(self, symbol: str) -> dict | None:
        """Get the learning profile for a stock. Returns None if insufficient data."""
        # Check memory cache
        if symbol in self._cache:
            ts, profile = self._cache[symbol]
            if time.time() - ts < PROFILE_CACHE_TTL:
                return profile

        # Check disk cache
        path = self._profile_path(symbol)
        if path.exists():
            try:
                profile = json.loads(path.read_text())
                # Check freshness (profiles older than 24h should be rebuilt)
                updated = profile.get("updated_at", "")
                if updated:
                    updated_dt = datetime.fromisoformat(updated)
                    if (datetime.utcnow() - updated_dt).total_seconds() < 86400:
                        self._cache[symbol] = (time.time(), profile)
                        return profile
            except Exception:
                pass

        # Build fresh profile
        profile = self._build_profile(symbol)
        if profile:
            self._cache[symbol] = (time.time(), profile)
        return profile

    def _build_profile(self, symbol: str) -> dict | None:
        """Analyze signal history and build a learning profile for the stock."""
        try:
            from app.database import SessionLocal
            from app.models import SignalLog

            db = SessionLocal()
            try:
                signals = (
                    db.query(SignalLog)
                    .filter(
                        SignalLog.symbol == symbol,
                        SignalLog.was_correct.isnot(None),
                    )
                    .order_by(SignalLog.created_at.desc())
                    .limit(MAX_SIGNALS_TO_ANALYZE)
                    .all()
                )

                if len(signals) < MIN_SIGNALS_FOR_PROFILE:
                    return None

                profile = self._analyze_signals(symbol, signals)

                # Save to disk
                try:
                    self._profile_path(symbol).write_text(
                        json.dumps(profile, indent=2, default=str)
                    )
                except Exception as e:
                    logger.debug(f"Failed to save profile for {symbol}: {e}")

                return profile
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to build profile for {symbol}: {e}")
            return None

    def _analyze_signals(self, symbol: str, signals: list) -> dict:
        """Core analysis: compute per-component accuracy and optimal thresholds."""
        now = time.time()
        halflife_sec = DECAY_HALFLIFE_DAYS * 86400
        ln2 = math.log(2)

        # Accumulators
        tech_correct = 0.0
        sent_correct = 0.0
        glob_correct = 0.0
        fund_correct = 0.0
        total_weight = 0.0

        # Direction threshold analysis
        correct_composites = []
        incorrect_composites = []

        # Regime analysis
        regime_stats = {}

        # Time-window accuracy (15min, 30min, 1hr)
        accuracy_15m = {"correct": 0, "total": 0}
        accuracy_30m = {"correct": 0, "total": 0}
        accuracy_1hr = {"correct": 0, "total": 0}

        # Trend: split signals into halves to detect improvement/degradation
        mid = len(signals) // 2
        recent_correct = 0
        recent_total = 0
        older_correct = 0
        older_total = 0

        for i, sig in enumerate(signals):
            try:
                age_seconds = now - sig.created_at.timestamp()
            except Exception:
                age_seconds = 0
            w = math.exp(-ln2 / halflife_sec * max(0, age_seconds))

            correct = sig.was_correct
            total_weight += w

            # Track composites for threshold optimization
            if correct:
                correct_composites.append(abs(sig.composite_score))
            else:
                incorrect_composites.append(abs(sig.composite_score))

            # Per-component directional agreement when signal was correct
            if correct:
                # Technical: did it agree with the final direction?
                if (sig.technical_score > 0 and sig.composite_score > 0) or \
                   (sig.technical_score < 0 and sig.composite_score < 0):
                    tech_correct += w
                elif abs(sig.technical_score) < 5:
                    tech_correct += 0.5 * w

                if (sig.sentiment_score > 0 and sig.composite_score > 0) or \
                   (sig.sentiment_score < 0 and sig.composite_score < 0):
                    sent_correct += w
                elif abs(sig.sentiment_score) < 5:
                    sent_correct += 0.5 * w

                if (sig.global_score > 0 and sig.composite_score > 0) or \
                   (sig.global_score < 0 and sig.composite_score < 0):
                    glob_correct += w
                elif abs(sig.global_score) < 5:
                    glob_correct += 0.5 * w

                # Fundamental: infer from residual (composite - tech - sent - global)
                fund_residual = sig.composite_score - sig.technical_score * 0.5 - \
                    sig.sentiment_score * 0.2 - sig.global_score * 0.1
                if (fund_residual > 0 and sig.composite_score > 0) or \
                   (fund_residual < 0 and sig.composite_score < 0):
                    fund_correct += w
                elif abs(fund_residual) < 5:
                    fund_correct += 0.5 * w

            # Regime tracking
            regime = sig.regime or "unknown"
            if regime not in regime_stats:
                regime_stats[regime] = {"correct": 0, "total": 0}
            regime_stats[regime]["total"] += 1
            if correct:
                regime_stats[regime]["correct"] += 1

            # Time-window accuracy
            if sig.was_correct is not None:
                accuracy_15m["total"] += 1
                if sig.was_correct:
                    accuracy_15m["correct"] += 1
            if getattr(sig, "was_correct_30min", None) is not None:
                accuracy_30m["total"] += 1
                if sig.was_correct_30min:
                    accuracy_30m["correct"] += 1
            if getattr(sig, "was_correct_1hr", None) is not None:
                accuracy_1hr["total"] += 1
                if sig.was_correct_1hr:
                    accuracy_1hr["correct"] += 1

            # Trend tracking
            if i < mid:
                recent_total += 1
                if correct:
                    recent_correct += 1
            else:
                older_total += 1
                if correct:
                    older_correct += 1

        # Compute component accuracies
        if total_weight < 1:
            return None

        tech_acc = tech_correct / total_weight
        sent_acc = sent_correct / total_weight
        glob_acc = glob_correct / total_weight
        fund_acc = fund_correct / total_weight

        # Compute optimal weights using softmax with temperature
        temp = 2.0
        accs = np.array([tech_acc, sent_acc, glob_acc, fund_acc])
        exp_accs = np.exp(accs / temp)
        raw_weights = exp_accs / exp_accs.sum()

        # Apply floors
        weights = {
            "technical": max(0.20, float(raw_weights[0])),
            "sentiment": max(0.08, float(raw_weights[1])),
            "global": max(0.05, float(raw_weights[2])),
            "fundamental": max(0.05, float(raw_weights[3])),
        }
        # Renormalize
        total_w = sum(weights.values())
        weights = {k: round(v / total_w, 4) for k, v in weights.items()}

        # Optimal direction threshold: find threshold that maximizes accuracy
        optimal_threshold = self._find_optimal_threshold(
            correct_composites, incorrect_composites
        )

        # Best regime
        best_regime = None
        best_regime_acc = 0
        for regime, stats in regime_stats.items():
            if stats["total"] >= 5:
                acc = stats["correct"] / stats["total"]
                if acc > best_regime_acc:
                    best_regime_acc = acc
                    best_regime = regime

        # Trend: is accuracy improving or degrading?
        recent_acc = recent_correct / max(recent_total, 1)
        older_acc = older_correct / max(older_total, 1)
        trend = "improving" if recent_acc > older_acc + 0.05 else \
                "degrading" if recent_acc < older_acc - 0.05 else "stable"

        # Best time window
        time_accuracies = {}
        if accuracy_15m["total"] > 0:
            time_accuracies["15min"] = round(accuracy_15m["correct"] / accuracy_15m["total"] * 100, 1)
        if accuracy_30m["total"] > 0:
            time_accuracies["30min"] = round(accuracy_30m["correct"] / accuracy_30m["total"] * 100, 1)
        if accuracy_1hr["total"] > 0:
            time_accuracies["1hr"] = round(accuracy_1hr["correct"] / accuracy_1hr["total"] * 100, 1)

        best_timeframe = max(time_accuracies, key=time_accuracies.get) if time_accuracies else "15min"

        return {
            "symbol": symbol,
            "sample_size": len(signals),
            "weights": weights,
            "component_accuracies": {
                "technical": round(tech_acc * 100, 1),
                "sentiment": round(sent_acc * 100, 1),
                "global": round(glob_acc * 100, 1),
                "fundamental": round(fund_acc * 100, 1),
            },
            "optimal_threshold": round(optimal_threshold, 1),
            "regime_stats": {
                k: {
                    "accuracy": round(v["correct"] / v["total"] * 100, 1),
                    "signals": v["total"],
                }
                for k, v in regime_stats.items()
                if v["total"] >= 3
            },
            "best_regime": best_regime,
            "time_window_accuracy": time_accuracies,
            "best_timeframe": best_timeframe,
            "trend": trend,
            "recent_accuracy": round(recent_acc * 100, 1),
            "overall_accuracy": round(
                sum(1 for s in signals if s.was_correct) / len(signals) * 100, 1
            ),
            "updated_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _find_optimal_threshold(correct_composites: list, incorrect_composites: list) -> float:
        """Find the direction threshold that maximizes signal accuracy.

        Tests thresholds from 5 to 25 and picks the one that gives
        the best ratio of correct signals above threshold vs incorrect ones.
        """
        if not correct_composites and not incorrect_composites:
            return 10.0

        best_threshold = 10.0
        best_score = 0

        for threshold in range(5, 26):
            correct_above = sum(1 for c in correct_composites if c >= threshold)
            incorrect_above = sum(1 for c in incorrect_composites if c >= threshold)
            total_above = correct_above + incorrect_above

            if total_above < 3:
                continue

            accuracy = correct_above / total_above
            # Penalize too-high thresholds (too few signals generated)
            coverage = total_above / max(len(correct_composites) + len(incorrect_composites), 1)
            # Score: accuracy * sqrt(coverage) — balance quality with quantity
            score = accuracy * math.sqrt(coverage)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return float(best_threshold)

    def rebuild_all_profiles(self) -> dict:
        """Rebuild profiles for all stocks with sufficient signal history.

        Called by the daily learning task after market close.
        Returns summary of what was learned.
        """
        from app.database import SessionLocal
        from app.models import SignalLog
        from sqlalchemy import func

        db = SessionLocal()
        try:
            # Find all symbols with enough validated signals
            symbols_with_signals = (
                db.query(SignalLog.symbol, func.count(SignalLog.id).label("cnt"))
                .filter(SignalLog.was_correct.isnot(None))
                .group_by(SignalLog.symbol)
                .having(func.count(SignalLog.id) >= MIN_SIGNALS_FOR_PROFILE)
                .all()
            )

            results = {
                "total_symbols": len(symbols_with_signals),
                "profiles_built": 0,
                "profiles_failed": 0,
                "improvements": [],
                "degradations": [],
            }

            for row in symbols_with_signals:
                symbol = row.symbol
                try:
                    # Load old profile for comparison
                    old_profile = None
                    path = self._profile_path(symbol)
                    if path.exists():
                        try:
                            old_profile = json.loads(path.read_text())
                        except Exception:
                            pass

                    # Clear cache to force rebuild
                    self._cache.pop(symbol, None)
                    profile = self._build_profile(symbol)

                    if profile:
                        results["profiles_built"] += 1

                        # Detect accuracy changes
                        if old_profile:
                            old_acc = old_profile.get("overall_accuracy", 0)
                            new_acc = profile.get("overall_accuracy", 0)
                            if new_acc > old_acc + 3:
                                results["improvements"].append({
                                    "symbol": symbol,
                                    "old_accuracy": old_acc,
                                    "new_accuracy": new_acc,
                                })
                            elif new_acc < old_acc - 3:
                                results["degradations"].append({
                                    "symbol": symbol,
                                    "old_accuracy": old_acc,
                                    "new_accuracy": new_acc,
                                })
                    else:
                        results["profiles_failed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to rebuild profile for {symbol}: {e}")
                    results["profiles_failed"] += 1

            logger.info(
                f"Stock learner: rebuilt {results['profiles_built']} profiles, "
                f"{len(results['improvements'])} improved, "
                f"{len(results['degradations'])} degraded"
            )
            return results
        finally:
            db.close()


# Singleton
stock_learner = StockLearner()
