from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import PriceAlertCreate
from app.services.alert_service import alert_service
from app.auth import get_current_active_user
from app.models import User

router = APIRouter()


@router.get("")
def list_alerts(db: Session = Depends(get_db), user: User = Depends(get_current_active_user)):
    alerts = alert_service.get_alerts(db, user_id=user.id)
    return [
        {
            "id": a.id,
            "symbol": a.symbol,
            "target_price": a.target_price,
            "condition": a.condition,
            "is_triggered": a.is_triggered,
            "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in alerts
    ]


@router.post("")
def create_alert(
    data: PriceAlertCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    alert = alert_service.create_alert(db, data.model_dump(), user_id=user.id)
    return {"id": alert.id, "message": "Alert created"}


@router.delete("/{alert_id}")
def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_active_user),
):
    success = alert_service.delete_alert(db, alert_id, user_id=user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert deleted"}
