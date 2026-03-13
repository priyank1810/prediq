from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import PriceAlertCreate, SmartAlertCreate
from app.services.alert_service import alert_service
from app.auth import get_current_active_user

router = APIRouter()


@router.get("")
def list_alerts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    alerts = alert_service.get_alerts(db, user_id=user.id, limit=limit, offset=offset)
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
    user=Depends(get_current_active_user),
):
    alert = alert_service.create_alert(db, data.model_dump(), user_id=user.id)
    return {"id": alert.id, "message": "Alert created"}


@router.delete("/{alert_id}")
def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    success = alert_service.delete_alert(db, alert_id, user_id=user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert deleted"}


# --- Smart Alerts ---

@router.get("/smart")
def list_smart_alerts(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    alerts = alert_service.get_smart_alerts(db, user_id=user.id, limit=limit, offset=offset)
    return [
        {
            "id": a.id,
            "symbol": a.symbol,
            "alert_type": a.alert_type,
            "threshold": a.threshold,
            "is_triggered": a.is_triggered,
            "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
            "trigger_data": a.trigger_data,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in alerts
    ]


@router.post("/smart")
def create_smart_alert(
    data: SmartAlertCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    valid_types = ["prediction_change", "sentiment_spike", "mood_extreme", "confidence_change"]
    if data.alert_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid alert type. Use one of: {valid_types}")

    alert = alert_service.create_smart_alert(db, data.model_dump(), user_id=user.id)
    return {"id": alert.id, "message": "Smart alert created"}


@router.delete("/smart/{alert_id}")
def delete_smart_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_active_user),
):
    success = alert_service.delete_smart_alert(db, alert_id, user_id=user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Smart alert not found")
    return {"message": "Smart alert deleted"}
