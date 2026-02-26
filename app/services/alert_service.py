from datetime import datetime
from sqlalchemy.orm import Session
from app.models import PriceAlert
from app.services.data_fetcher import data_fetcher


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


alert_service = AlertService()
