from fastapi import APIRouter, HTTPException, Depends
from app.schemas import PredictionRequest
from app.services.prediction_service import prediction_service
from app.auth import get_current_active_user
from app.rate_limit import rate_limit
from app.models import User

router = APIRouter()


@router.post("/{symbol}")
def predict_stock(
    symbol: str,
    request: PredictionRequest,
    user: User = Depends(get_current_active_user),
    _rate=rate_limit(max_calls=5, window_seconds=60),
):
    try:
        result = prediction_service.predict(
            symbol=symbol.upper(),
            horizon=request.horizon,
            models=request.models,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
