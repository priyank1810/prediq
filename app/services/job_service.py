"""Job queue service — shared interface for server and worker processes."""
from __future__ import annotations

import json
import uuid
from datetime import timedelta
from sqlalchemy import text
from app.database import SessionLocal
from app.models import JobQueue
from app.utils.helpers import now_ist


class JobService:

    def enqueue(self, job_type: str, params: dict = None, priority: int = 0) -> int:
        """Create a new pending job. Returns job_id."""
        db = SessionLocal()
        try:
            job = JobQueue(
                job_type=job_type,
                priority=priority,
                params=json.dumps(params) if params else None,
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            return job.id
        finally:
            db.close()

    def claim_next(self) -> dict | None:
        """Atomically claim the highest-priority pending job.
        Returns dict with job info or None if no pending jobs.
        Safe for single-worker — SQLite serializes writes via WAL + busy_timeout.
        For multi-worker setups, prefer claim_job() which uses a unique worker_id."""
        db = SessionLocal()
        try:
            job = (
                db.query(JobQueue)
                .filter(JobQueue.status == "pending")
                .order_by(JobQueue.priority.desc(), JobQueue.created_at.asc())
                .first()
            )
            if not job:
                return None
            job.status = "running"
            job.started_at = now_ist()
            db.commit()
            return {
                "id": job.id,
                "job_type": job.job_type,
                "priority": job.priority,
                "params": json.loads(job.params) if job.params else {},
            }
        finally:
            db.close()

    def claim_job(self, worker_id: str | None = None) -> dict | None:
        """Atomically claim the next pending job, preventing duplicate processing.

        Uses an UPDATE ... WHERE pattern so that only one worker can claim a
        given job even when multiple workers poll concurrently.  SQLite's
        implicit write lock serializes these UPDATEs; the SELECT that follows
        reads only the row this transaction touched.

        Args:
            worker_id: Optional identifier for the claiming worker (for debugging).

        Returns:
            dict with job info, or None if no pending jobs.
        """
        db = SessionLocal()
        try:
            # Step 1: find the best candidate (read, no lock needed)
            candidate = (
                db.query(JobQueue)
                .filter(JobQueue.status == "pending")
                .order_by(JobQueue.priority.desc(), JobQueue.created_at.asc())
                .first()
            )
            if not candidate:
                return None

            # Step 2: atomic conditional UPDATE — only succeeds if still pending
            now = now_ist()
            result = db.execute(
                text(
                    "UPDATE job_queue "
                    "SET status = 'running', started_at = :now "
                    "WHERE id = :job_id AND status = 'pending'"
                ),
                {"now": now, "job_id": candidate.id},
            )
            db.commit()

            if result.rowcount == 0:
                # Another worker claimed it between our SELECT and UPDATE
                return None

            # Refresh to get committed state
            db.refresh(candidate)
            return {
                "id": candidate.id,
                "job_type": candidate.job_type,
                "priority": candidate.priority,
                "params": json.loads(candidate.params) if candidate.params else {},
                "worker_id": worker_id,
            }
        finally:
            db.close()

    def complete(self, job_id: int, result: dict):
        """Mark a job as completed with its result."""
        db = SessionLocal()
        try:
            job = db.query(JobQueue).filter(JobQueue.id == job_id).first()
            if job:
                job.status = "completed"
                job.result = json.dumps(result)
                job.completed_at = now_ist()
                db.commit()
        finally:
            db.close()

    def fail(self, job_id: int, error: str):
        """Mark a job as failed with an error message."""
        db = SessionLocal()
        try:
            job = db.query(JobQueue).filter(JobQueue.id == job_id).first()
            if job:
                job.status = "failed"
                job.error = error
                job.completed_at = now_ist()
                db.commit()
        finally:
            db.close()

    def get_status(self, job_id: int) -> dict | None:
        """Get job status and result if completed."""
        db = SessionLocal()
        try:
            job = db.query(JobQueue).filter(JobQueue.id == job_id).first()
            if not job:
                return None
            data = {
                "id": job.id,
                "job_type": job.job_type,
                "status": job.status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
            }
            if job.status == "completed" and job.result:
                data["result"] = json.loads(job.result)
            if job.status == "failed":
                data["error"] = job.error
            return data
        finally:
            db.close()

    def has_pending(self, job_type: str) -> bool:
        """Check if there's already a pending or running job of this type."""
        db = SessionLocal()
        try:
            count = (
                db.query(JobQueue)
                .filter(
                    JobQueue.job_type == job_type,
                    JobQueue.status.in_(["pending", "running"]),
                )
                .count()
            )
            return count > 0
        finally:
            db.close()

    def get_completed_background_jobs(self) -> list[dict]:
        """Get completed background jobs (priority=0) that haven't been broadcast yet."""
        db = SessionLocal()
        try:
            jobs = (
                db.query(JobQueue)
                .filter(
                    JobQueue.status == "completed",
                    JobQueue.priority == 0,
                )
                .order_by(JobQueue.completed_at.asc())
                .limit(50)
                .all()
            )
            return [
                {
                    "id": j.id,
                    "job_type": j.job_type,
                    "result": json.loads(j.result) if j.result else {},
                }
                for j in jobs
            ]
        finally:
            db.close()

    def mark_broadcast(self, job_id: int):
        """Mark a background job as broadcast (final state)."""
        db = SessionLocal()
        try:
            job = db.query(JobQueue).filter(JobQueue.id == job_id).first()
            if job:
                job.status = "broadcast"
                db.commit()
        finally:
            db.close()

    def reset_stale(self, timeout_seconds: int = 120):
        """Reset running jobs that have been stuck longer than timeout back to pending."""
        db = SessionLocal()
        try:
            cutoff = now_ist().replace(tzinfo=None) - timedelta(seconds=timeout_seconds)
            stale = (
                db.query(JobQueue)
                .filter(
                    JobQueue.status == "running",
                    JobQueue.started_at < cutoff,
                )
                .all()
            )
            for job in stale:
                job.status = "pending"
                job.started_at = None
            if stale:
                db.commit()
            return len(stale)
        finally:
            db.close()

    def release_stale_jobs(self, timeout_minutes: int = 10) -> int:
        """Reset jobs stuck in 'running' for more than *timeout_minutes* back to 'pending'.

        This is the recommended stale-job recovery method for multi-worker
        deployments.  It uses a longer default timeout (10 min) than
        reset_stale() to avoid reclaiming jobs that are simply slow.

        Returns:
            Number of jobs released.
        """
        db = SessionLocal()
        try:
            cutoff = now_ist().replace(tzinfo=None) - timedelta(minutes=timeout_minutes)
            stale = (
                db.query(JobQueue)
                .filter(
                    JobQueue.status == "running",
                    JobQueue.started_at < cutoff,
                )
                .all()
            )
            for job in stale:
                job.status = "pending"
                job.started_at = None
            if stale:
                db.commit()
            return len(stale)
        finally:
            db.close()

    def cleanup_old(self, hours: int = 24):
        """Delete completed/failed/broadcast jobs older than the given hours."""
        db = SessionLocal()
        try:
            cutoff = now_ist().replace(tzinfo=None) - timedelta(hours=hours)
            deleted = (
                db.query(JobQueue)
                .filter(
                    JobQueue.status.in_(["completed", "failed", "broadcast"]),
                    JobQueue.created_at < cutoff,
                )
                .delete(synchronize_session=False)
            )
            db.commit()
            return deleted
        finally:
            db.close()


job_service = JobService()
