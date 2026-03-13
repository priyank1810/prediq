import asyncio
from fastapi import APIRouter, HTTPException
from app.schemas import PredictionRequest
from app.rate_limit import rate_limit
from app.utils.helpers import validate_symbol

router = APIRouter()

PREDICTION_TIMEOUT = 90  # seconds


@router.post("/{symbol}")
async def predict_stock(
    symbol: str,
    request: PredictionRequest,
    _rate=rate_limit(max_calls=5, window_seconds=60),
):
    try:
        sym = validate_symbol(symbol)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol!r}")
    try:
        from app.services.prediction_service import prediction_service
        result = await asyncio.wait_for(
            asyncio.to_thread(
                prediction_service.predict,
                sym,
                request.horizon,
                request.models,
            ),
            timeout=PREDICTION_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out. Try again or use a simpler model.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
