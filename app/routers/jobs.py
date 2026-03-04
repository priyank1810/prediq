from fastapi import APIRouter, HTTPException
from app.services.job_service import job_service

router = APIRouter()


@router.get("/{job_id}")
def get_job_status(job_id: int):
    """Get job status and result when completed."""
    status = job_service.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status
