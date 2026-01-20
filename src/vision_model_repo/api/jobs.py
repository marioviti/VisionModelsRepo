"""
Async job processing system for long-running inference tasks.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents an async inference job."""
    job_id: str
    status: JobStatus
    created_at: datetime
    task_func: Callable
    task_args: tuple = field(default_factory=tuple)
    task_kwargs: dict = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None


class JobQueue:
    """
    Manages async job processing queue.
    """

    def __init__(self, max_concurrent_jobs: int = 2):
        self._jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._max_concurrent = max_concurrent_jobs
        self._workers_started = False
        self._worker_tasks = []

    async def start_workers(self):
        """Start background worker tasks."""
        if self._workers_started:
            return

        logger.info(f"Starting {self._max_concurrent} job workers")
        for i in range(self._max_concurrent):
            task = asyncio.create_task(self._worker(i))
            self._worker_tasks.append(task)

        self._workers_started = True

    async def stop_workers(self):
        """Stop background worker tasks."""
        logger.info("Stopping job workers")
        for task in self._worker_tasks:
            task.cancel()

        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        self._workers_started = False

    async def _worker(self, worker_id: int):
        """Background worker that processes jobs from the queue."""
        logger.info(f"Worker {worker_id} started")

        while True:
            try:
                job = await self._queue.get()
                logger.info(f"Worker {worker_id} processing job {job.job_id}")

                job.status = JobStatus.RUNNING
                start_time = time.time()

                try:
                    # Execute the task function
                    result = await asyncio.to_thread(
                        job.task_func,
                        *job.task_args,
                        **job.task_kwargs
                    )

                    job.status = JobStatus.COMPLETED
                    job.result = result
                    job.completed_at = datetime.now()
                    job.processing_time = time.time() - start_time

                    logger.info(f"Job {job.job_id} completed in {job.processing_time:.2f}s")

                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.completed_at = datetime.now()
                    job.processing_time = time.time() - start_time

                    logger.error(f"Job {job.job_id} failed: {e}")

                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    def submit_job(
        self,
        task_func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Submit a new job to the queue.

        Args:
            task_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            task_func=task_func,
            task_args=args,
            task_kwargs=kwargs
        )

        self._jobs[job_id] = job

        # Add to queue (non-blocking)
        asyncio.create_task(self._queue.put(job))

        logger.info(f"Job {job_id} submitted")
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status information.

        Returns:
            Dictionary with job status, or None if not found
        """
        job = self._jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "processing_time": job.processing_time,
            "result": job.result if job.status == JobStatus.COMPLETED else None,
            "error": job.error if job.status == JobStatus.FAILED else None
        }

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from the queue.

        Args:
            job_id: Job ID to delete

        Returns:
            True if deleted, False if not found
        """
        if job_id in self._jobs:
            del self._jobs[job_id]
            logger.info(f"Job {job_id} deleted")
            return True
        return False

    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all jobs."""
        return {
            job_id: self.get_job_status(job_id)
            for job_id in self._jobs.keys()
        }

    def cleanup_completed_jobs(self, max_age_seconds: int = 3600):
        """
        Remove completed/failed jobs older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds
        """
        now = datetime.now()
        to_delete = []

        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                if job.completed_at:
                    age = (now - job.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_delete.append(job_id)

        for job_id in to_delete:
            self.delete_job(job_id)

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old jobs")


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(max_concurrent_jobs=2)
    return _job_queue
