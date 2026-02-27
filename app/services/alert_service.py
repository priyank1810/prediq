from __future__ import annotations

import json
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import PriceAlert, SmartAlert
from app.services.data_fetcher import data_fetcher

logger = logging.getLogger(__name__)


class AlertService:
    def get_alerts(self, db: Session, user_id: int = None) -> list[PriceAlert]:
        query = db.query(PriceAlert)
        if user_id is not None:
            query = query.filter(PriceAlert.user_id == user_id)
        return query.order_by(PriceAlert.created_at.desc()).all()

    def create_alert(self, db: Session, data: dict, user_id: int = None) -> PriceAlert:
        alert = PriceAlert(**data, user_id=user_id)
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return alert

    def delete_alert(self, db: Session, alert_id: int, user_id: int = None) -> bool:
        query = db.query(PriceAlert).filter(PriceAlert.id == alert_id)
        if user_id is not None:
            query = query.filter(PriceAlert.user_id == user_id)
        alert = query.first()
        if not alert:
            return False
        db.delete(alert)
        db.commit()
        return True

    def check_alerts(self, db: Session) -> list[dict]:
        """Check all active alerts across all users (background task)."""
        active_alerts = db.query(PriceAlert).filter(PriceAlert.is_triggered == False).all()
        triggered = []

        for alert in active_alerts:
            try:
                quote = data_fetcher.get_live_quote(alert.symbol)
                current_price = quote["ltp"]
                should_trigger = False

                if alert.condition == "above" and current_price >= alert.target_price:
                    should_trigger = True
                elif alert.condition == "below" and current_price <= alert.target_price:
                    should_trigger = True

                if should_trigger:
                    alert.is_triggered = True
                    alert.triggered_at = datetime.utcnow()
                    db.commit()
                    triggered.append({
                        "symbol": alert.symbol,
                        "condition": alert.condition,
                        "target_price": alert.target_price,
                        "current_price": current_price,
                    })
            except Exception:
                continue

        return triggered

    def check_alerts_batched(self, db: Session) -> list[dict]:
        """Check all active alerts using a single batch quote fetch."""
        active_alerts = db.query(PriceAlert).filter(PriceAlert.is_triggered == False).all()
        if not active_alerts:
            return []

        # Batch-fetch all unique symbols in one API call
        symbols = list({alert.symbol for alert in active_alerts})
        quotes = data_fetcher.get_bulk_quotes(symbols)
        price_map = {}
        for q in quotes:
            if q.get("ltp"):
                price_map[q["symbol"]] = float(q["ltp"])

        triggered = []
        for alert in active_alerts:
            current_price = price_map.get(alert.symbol)
            if current_price is None:
                continue

            should_trigger = False
            if alert.condition == "above" and current_price >= alert.target_price:
                should_trigger = True
            elif alert.condition == "below" and current_price <= alert.target_price:
                should_trigger = True

            if should_trigger:
                alert.is_triggered = True
                alert.triggered_at = datetime.utcnow()
                db.commit()
                triggered.append({
                    "symbol": alert.symbol,
                    "condition": alert.condition,
                    "target_price": alert.target_price,
                    "current_price": current_price,
                })

        return triggered

    # --- Smart Alerts ---

    def get_smart_alerts(self, db: Session, user_id: int = None) -> list[SmartAlert]:
        query = db.query(SmartAlert)
        if user_id is not None:
            query = query.filter(SmartAlert.user_id == user_id)
        return query.order_by(SmartAlert.created_at.desc()).all()

    def create_smart_alert(self, db: Session, data: dict, user_id: int = None) -> SmartAlert:
        alert = SmartAlert(**data, user_id=user_id)
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return alert

    def delete_smart_alert(self, db: Session, alert_id: int, user_id: int = None) -> bool:
        query = db.query(SmartAlert).filter(SmartAlert.id == alert_id)
        if user_id is not None:
            query = query.filter(SmartAlert.user_id == user_id)
        alert = query.first()
        if not alert:
            return False
        db.delete(alert)
        db.commit()
        return True

    def check_smart_alerts(self, db: Session) -> list[dict]:
        """Check all active smart alerts (background task)."""
        active = db.query(SmartAlert).filter(SmartAlert.is_triggered == False).all()
        triggered = []

        for alert in active:
            try:
                result = None
                if alert.alert_type == "prediction_change":
                    result = self._check_prediction_change(alert)
                elif alert.alert_type == "sentiment_spike":
                    result = self._check_sentiment_spike(alert)
                elif alert.alert_type == "mood_extreme":
                    result = self._check_mood_extreme(alert)
                elif alert.alert_type == "confidence_change":
                    result = self._check_confidence_change(alert)

                if result:
                    alert.is_triggered = True
                    alert.triggered_at = datetime.utcnow()
                    alert.trigger_data = json.dumps(result)
                    db.commit()
                    triggered.append({
                        "alert_id": alert.id,
                        "alert_type": alert.alert_type,
                        "symbol": alert.symbol,
                        **result,
                    })
            except Exception as e:
                logger.debug(f"Smart alert check failed for {alert.id}: {e}")
                continue

        return triggered

    def _check_prediction_change(self, alert: SmartAlert) -> dict | None:
        """Check if prediction direction changed."""
        if not alert.symbol:
            return None
        try:
            from app.services.prediction_service import prediction_service
            result = prediction_service.predict(alert.symbol, "1d")
            ensemble = result.get("ensemble", {})
            preds = ensemble.get("predictions", [])
            if not preds:
                return None
            current_price = data_fetcher.get_live_quote(alert.symbol).get("ltp", 0)
            if current_price <= 0:
                return None
            change_pct = (preds[-1] - current_price) / current_price * 100
            threshold = alert.threshold or 2.0
            if abs(change_pct) > threshold:
                direction = "BULLISH" if change_pct > 0 else "BEARISH"
                return {"direction": direction, "change_pct": round(change_pct, 2)}
        except Exception:
            pass
        return None

    def _check_sentiment_spike(self, alert: SmartAlert) -> dict | None:
        """Check if sentiment score swings beyond threshold."""
        if not alert.symbol:
            return None
        try:
            from app.services.sentiment_service import sentiment_service
            sent = sentiment_service.get_sentiment(alert.symbol)
            score = sent.get("score", 0)
            threshold = alert.threshold or 50.0
            if abs(score) > threshold:
                return {"sentiment_score": score, "threshold": threshold}
        except Exception:
            pass
        return None

    def _check_mood_extreme(self, alert: SmartAlert) -> dict | None:
        """Check if Market Mood crosses into Extreme Fear/Greed."""
        try:
            from app.services.market_mood_service import market_mood_service
            mood = market_mood_service.get_mood()
            score = mood.get("score", 50)
            label = mood.get("label", "")
            if "Extreme" in label:
                return {"mood_score": score, "label": label}
        except Exception:
            pass
        return None

    def _check_confidence_change(self, alert: SmartAlert) -> dict | None:
        """Check if prediction confidence shifts beyond threshold."""
        if not alert.symbol:
            return None
        try:
            from app.services.prediction_service import prediction_service
            result = prediction_service.predict(alert.symbol, "1d")
            for model_name in ["lstm", "xgboost"]:
                model_data = result.get(model_name)
                if model_data and model_data.get("confidence_score"):
                    conf = model_data["confidence_score"]
                    threshold = alert.threshold or 80.0
                    if conf > threshold:
                        return {"model": model_name, "confidence": conf, "threshold": threshold}
        except Exception:
            pass
        return None


alert_service = AlertService()
