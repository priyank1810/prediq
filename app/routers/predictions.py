import asyncio
from fastapi import APIRouter, HTTPException
from app.schemas import PredictionRequest
from app.rate_limit import rate_limit

router = APIRouter()


@router.post("/{symbol}")
async def predict_stock(
    symbol: str,
    request: PredictionRequest,
    _rate=rate_limit(max_calls=5, window_seconds=60),
):
    try:
        from app.services.prediction_service import prediction_service
        result = await asyncio.to_thread(
            prediction_service.predict,
            symbol.upper(),
            request.horizon,
            request.models,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
