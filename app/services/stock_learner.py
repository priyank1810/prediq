"""Per-stock learning service.

Analyzes AI prediction accuracy and trade signal outcomes for each stock
to build a learning profile. Uses PredictionLog (price predictions) and
TradeSignalLog (entry/target/SL outcomes) — NOT the old SignalLog.

Profiles are stored as JSON in data/stock_profiles/.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "stock_profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

MIN_DATA_FOR_PROFILE = 5
PROFILE_CACHE_TTL = 1800  # 30 minutes


class StockLearner:
    def __init__(self):
        self._cache = {}

    def _profile_path(self, symbol: str) -> Path:
        safe = symbol.replace(" ", "_").replace("^", "")
        return PROFILES_DIR / f"{safe}.json"

    def get_profile(self, symbol: str) -> dict | None:
        if symbol in self._cache:
            ts, profile = self._cache[symbol]
            if time.time() - ts < PROFILE_CACHE_TTL:
                return profile

        path = self._profile_path(symbol)
        if path.exists():
            try:
                profile = json.loads(path.read_text())
                updated = profile.get("updated_at", "")
                if updated:
                    updated_dt = datetime.fromisoformat(updated)
                    if (datetime.utcnow() - updated_dt).total_seconds() < 86400:
                        self._cache[symbol] = (time.time(), profile)
                        return profile
            except Exception:
                pass

        profile = self._build_profile(symbol)
        if profile:
            self._cache[symbol] = (time.time(), profile)
        return profile

    def _build_profile(self, symbol: str) -> dict | None:
        try:
            from app.database import SessionLocal
            db = SessionLocal()
            try:
                pred_data = self._analyze_predictions(db, symbol)
                trade_data = self._analyze_trades(db, symbol)

                if not pred_data and not trade_data:
                    return None

                profile = {
                    "symbol": symbol,
                    "predictions": pred_data,
                    "trades": trade_data,
                    "updated_at": datetime.utcnow().isoformat(),
                }

                # Compute overall summary
                profile["summary"] = self._compute_summary(pred_data, trade_data)

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

    def _analyze_predictions(self, db, symbol: str) -> dict | None:
        """Analyze AI price prediction accuracy from PredictionLog."""
        from app.models import PredictionLog

        logs = (
            db.query(PredictionLog)
            .filter(
                PredictionLog.symbol == symbol,
                PredictionLog.actual_price.isnot(None),
                PredictionLog.actual_price > 0,
            )
            .order_by(PredictionLog.prediction_date.desc())
            .limit(200)
            .all()
        )

        if len(logs) < MIN_DATA_FOR_PROFILE:
            return None

        # Per-model stats with horizon-aware thresholds
        def _get_threshold(log):
            days = (log.target_date - log.prediction_date).days if log.target_date and log.prediction_date else 1
            if days <= 0: return 0.3
            elif days <= 1: return 0.5
            elif days <= 5: return 1.0
            elif days <= 22: return 1.5
            else: return 2.0

        model_stats = {}
        for log in logs:
            m = log.model_type or "unknown"
            if m not in model_stats:
                model_stats[m] = {"mape_sum": 0, "accurate": 0, "within_5pct": 0, "total": 0}
            s = model_stats[m]
            s["total"] += 1
            mape = abs(log.predicted_price - log.actual_price) / log.actual_price * 100
            s["mape_sum"] += mape
            if mape <= _get_threshold(log):
                s["accurate"] += 1
            if mape <= 5:
                s["within_5pct"] += 1

        models = {}
        best_model = None
        best_mape = 999
        for model, s in model_stats.items():
            if s["total"] == 0:
                continue
            avg_mape = round(s["mape_sum"] / s["total"], 2)
            models[model] = {
                "total": s["total"],
                "avg_mape": avg_mape,
                "accuracy": round(s["accurate"] / s["total"] * 100, 1),
                "accuracy_5pct": round(s["within_5pct"] / s["total"] * 100, 1),
            }
            if avg_mape < best_mape:
                best_mape = avg_mape
                best_model = model

        # Trend: compare recent vs older predictions
        mid = len(logs) // 2
        recent_mapes = []
        older_mapes = []
        for i, log in enumerate(logs):
            mape = abs(log.predicted_price - log.actual_price) / log.actual_price * 100
            if i < mid:
                recent_mapes.append(mape)
            else:
                older_mapes.append(mape)

        recent_avg = sum(recent_mapes) / len(recent_mapes) if recent_mapes else 0
        older_avg = sum(older_mapes) / len(older_mapes) if older_mapes else 0
        trend = "improving" if recent_avg < older_avg - 0.5 else \
                "degrading" if recent_avg > older_avg + 0.5 else "stable"

        overall_mape = round(sum(m["mape_sum"] for m in model_stats.values()) /
                            max(sum(m["total"] for m in model_stats.values()), 1), 2)

        return {
            "total_predictions": len(logs),
            "overall_mape": overall_mape,
            "overall_accuracy": round(100 - overall_mape, 1),
            "models": models,
            "best_model": best_model,
            "trend": trend,
            "recent_mape": round(recent_avg, 2),
        }

    def _analyze_trades(self, db, symbol: str) -> dict | None:
        """Analyze trade signal outcomes from TradeSignalLog."""
        from app.models import TradeSignalLog

        trades = (
            db.query(TradeSignalLog)
            .filter(
                TradeSignalLog.symbol == symbol,
                TradeSignalLog.status != "open",
            )
            .order_by(TradeSignalLog.created_at.desc())
            .limit(100)
            .all()
        )

        if len(trades) < MIN_DATA_FOR_PROFILE:
            return None

        target_hits = sum(1 for t in trades if t.status == "target_hit")
        sl_hits = sum(1 for t in trades if t.status == "sl_hit")
        expired = sum(1 for t in trades if t.status == "expired")
        expired_profit = sum(1 for t in trades if t.status == "expired" and (t.outcome_pct or 0) > 0)

        wins = target_hits + expired_profit
        win_rate = round(wins / len(trades) * 100, 1) if trades else 0

        # Average P&L
        pnls = [t.outcome_pct for t in trades if t.outcome_pct is not None]
        avg_pnl = round(sum(pnls) / len(pnls), 2) if pnls else 0
        win_pnls = [p for p in pnls if p > 0]
        loss_pnls = [p for p in pnls if p < 0]
        avg_win = round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0
        avg_loss = round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0

        # By timeframe
        by_timeframe = {}
        for t in trades:
            tf = t.timeframe or "unknown"
            if tf not in by_timeframe:
                by_timeframe[tf] = {"wins": 0, "total": 0, "pnl_sum": 0}
            by_timeframe[tf]["total"] += 1
            if t.status == "target_hit" or (t.status == "expired" and (t.outcome_pct or 0) > 0):
                by_timeframe[tf]["wins"] += 1
            by_timeframe[tf]["pnl_sum"] += t.outcome_pct or 0

        tf_stats = {}
        best_tf = None
        best_tf_wr = 0
        for tf, s in by_timeframe.items():
            if s["total"] >= 3:
                wr = round(s["wins"] / s["total"] * 100, 1)
                tf_stats[tf] = {
                    "win_rate": wr,
                    "trades": s["total"],
                    "avg_pnl": round(s["pnl_sum"] / s["total"], 2),
                }
                if wr > best_tf_wr:
                    best_tf_wr = wr
                    best_tf = tf

        # Trend
        mid = len(trades) // 2
        recent_wins = sum(1 for t in trades[:mid]
                         if t.status == "target_hit" or (t.status == "expired" and (t.outcome_pct or 0) > 0))
        older_wins = sum(1 for t in trades[mid:]
                        if t.status == "target_hit" or (t.status == "expired" and (t.outcome_pct or 0) > 0))
        recent_wr = recent_wins / max(mid, 1) * 100
        older_wr = older_wins / max(len(trades) - mid, 1) * 100
        trend = "improving" if recent_wr > older_wr + 5 else \
                "degrading" if recent_wr < older_wr - 5 else "stable"

        return {
            "total_trades": len(trades),
            "win_rate": win_rate,
            "target_hits": target_hits,
            "sl_hits": sl_hits,
            "expired": expired,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "by_timeframe": tf_stats,
            "best_timeframe": best_tf,
            "trend": trend,
        }

    def _compute_summary(self, pred_data: dict | None, trade_data: dict | None) -> dict:
        """Compute a unified summary from prediction and trade data."""
        summary = {
            "has_predictions": pred_data is not None,
            "has_trades": trade_data is not None,
        }

        if pred_data:
            summary["prediction_accuracy"] = pred_data.get("overall_accuracy", 0)
            summary["prediction_mape"] = pred_data.get("overall_mape", 0)
            summary["prediction_trend"] = pred_data.get("trend", "stable")
            summary["best_model"] = pred_data.get("best_model")
            summary["total_predictions"] = pred_data.get("total_predictions", 0)

        if trade_data:
            summary["trade_win_rate"] = trade_data.get("win_rate", 0)
            summary["trade_avg_pnl"] = trade_data.get("avg_pnl", 0)
            summary["trade_trend"] = trade_data.get("trend", "stable")
            summary["best_timeframe"] = trade_data.get("best_timeframe")
            summary["total_trades"] = trade_data.get("total_trades", 0)

        # Overall trend
        trends = []
        if pred_data:
            trends.append(pred_data.get("trend", "stable"))
        if trade_data:
            trends.append(trade_data.get("trend", "stable"))

        if "improving" in trends and "degrading" not in trends:
            summary["overall_trend"] = "improving"
        elif "degrading" in trends and "improving" not in trends:
            summary["overall_trend"] = "degrading"
        else:
            summary["overall_trend"] = "stable"

        return summary

    def rebuild_all_profiles(self) -> dict:
        from app.database import SessionLocal
        from app.models import PredictionLog, TradeSignalLog
        from sqlalchemy import func

        db = SessionLocal()
        try:
            # Get symbols from both tables
            pred_symbols = set(
                row.symbol for row in
                db.query(PredictionLog.symbol)
                .filter(PredictionLog.actual_price.isnot(None))
                .group_by(PredictionLog.symbol)
                .having(func.count(PredictionLog.id) >= MIN_DATA_FOR_PROFILE)
                .all()
            )
            trade_symbols = set(
                row.symbol for row in
                db.query(TradeSignalLog.symbol)
                .filter(TradeSignalLog.status != "open")
                .group_by(TradeSignalLog.symbol)
                .having(func.count(TradeSignalLog.id) >= MIN_DATA_FOR_PROFILE)
                .all()
            )
            all_symbols = pred_symbols | trade_symbols

            results = {
                "total_symbols": len(all_symbols),
                "profiles_built": 0,
                "profiles_failed": 0,
                "improvements": [],
                "degradations": [],
            }

            for symbol in all_symbols:
                try:
                    old_profile = None
                    path = self._profile_path(symbol)
                    if path.exists():
                        try:
                            old_profile = json.loads(path.read_text())
                        except Exception:
                            pass

                    self._cache.pop(symbol, None)
                    profile = self._build_profile(symbol)

                    if profile:
                        results["profiles_built"] += 1
                        if old_profile and old_profile.get("summary"):
                            old_acc = old_profile["summary"].get("prediction_accuracy", 0)
                            new_acc = profile["summary"].get("prediction_accuracy", 0)
                            if new_acc > old_acc + 2:
                                results["improvements"].append({"symbol": symbol, "old": old_acc, "new": new_acc})
                            elif new_acc < old_acc - 2:
                                results["degradations"].append({"symbol": symbol, "old": old_acc, "new": new_acc})
                    else:
                        results["profiles_failed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to rebuild profile for {symbol}: {e}")
                    results["profiles_failed"] += 1

            logger.info(f"Stock learner: rebuilt {results['profiles_built']} profiles")
            return results
        finally:
            db.close()


stock_learner = StockLearner()
