import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import PredictionRequest
from app.services.job_service import job_service
from app.rate_limit import rate_limit

router = APIRouter()


@router.post("/{symbol}")
async def predict_stock(
    symbol: str,
    request: PredictionRequest,
    _rate=rate_limit(max_calls=5, window_seconds=60),
):
    try:
        job_id = await asyncio.to_thread(
            job_service.enqueue,
            "prediction",
            {"symbol": symbol.upper(), "horizon": request.horizon, "models": request.models},
            10,
        )

        # Poll-wait: check every 0.5s for up to 30s
        for _ in range(60):
            await asyncio.sleep(0.5)
            status = await asyncio.to_thread(job_service.get_status, job_id)
            if not status:
                break
            if status["status"] == "completed":
                return status["result"]
            if status["status"] == "failed":
                raise HTTPException(status_code=500, detail=f"Prediction failed: {status.get('error', 'unknown')}")

        # Timeout — return 202 with job_id for polling
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "pending", "poll_url": f"/api/jobs/{job_id}"},
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
