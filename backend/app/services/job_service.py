"""Lightweight background job service for document processing.

Provides a simple in-process job queue that tracks document processing
lifecycle without requiring Redis/Celery. Suitable for single-server
deployment. Can be replaced with Celery/RQ for distributed processing.

Job lifecycle:
    QUEUED → PROCESSING → EXTRACT → CHUNK → EMBED → READY
                                   → FAILED

Design decisions:
- Jobs are stored in-memory (dict) — survives the process, not restarts
- Thread-safe via Python's GIL (sufficient for single-server)
- Idempotent: re-processing an already-READY document is a no-op
- Duplicate prevention: concurrent processing of same document is blocked
- Status polling via GET /documents/{id}/status (existing endpoint)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Document processing job status."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    EXTRACTING = "EXTRACTING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    READY = "READY"
    FAILED = "FAILED"


@dataclass
class ProcessingJob:
    """A document processing job."""
    document_id: int
    user_id: int
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0.0 to 1.0
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    stage: str = ""  # Current processing stage description


class JobService:
    """In-process job queue for document processing.

    Thread-safe for single-server deployment.
    """

    def __init__(self):
        self._jobs: Dict[int, ProcessingJob] = {}
        self._lock = threading.Lock()

    def enqueue(self, document_id: int, user_id: int) -> ProcessingJob:
        """Enqueue a document for processing.

        If the document is already being processed or is READY,
        returns the existing job without creating a duplicate.

        Args:
            document_id: The document to process.
            user_id: The owner of the document.

        Returns:
            ProcessingJob instance.
        """
        with self._lock:
            if document_id in self._jobs:
                existing = self._jobs[document_id]
                # If already processing or queued, don't duplicate
                if existing.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                    logger.debug("Job already active for doc %d, status=%s", document_id, existing.status)
                    return existing
                # If READY, don't re-process
                if existing.status == JobStatus.READY:
                    logger.debug("Document %d already READY, skipping", document_id)
                    return existing

            job = ProcessingJob(
                document_id=document_id,
                user_id=user_id,
                status=JobStatus.QUEUED,
            )
            self._jobs[document_id] = job
            logger.info("Job enqueued for document %d", document_id)
            return job

    def start(self, document_id: int) -> Optional[ProcessingJob]:
        """Mark a job as actively processing.

        Returns None if no queued job exists for this document.
        """
        with self._lock:
            job = self._jobs.get(document_id)
            if not job or job.status != JobStatus.QUEUED:
                return None
            job.status = JobStatus.PROCESSING
            job.started_at = time.monotonic()
            job.progress = 0.0
            logger.info("Job started for document %d", document_id)
            return job

    def update_stage(self, document_id: int, stage: str, progress: float = 0.0):
        """Update the processing stage of a job."""
        with self._lock:
            job = self._jobs.get(document_id)
            if job:
                job.stage = stage
                job.progress = min(1.0, max(0.0, progress))

    def complete(self, document_id: int):
        """Mark a job as completed successfully."""
        with self._lock:
            job = self._jobs.get(document_id)
            if job:
                job.status = JobStatus.READY
                job.progress = 1.0
                job.completed_at = time.monotonic()
                job.stage = "complete"
                elapsed = (job.completed_at - job.started_at) if job.started_at else 0
                logger.info("Job completed for document %d (%.2fs)", document_id, elapsed)

    def fail(self, document_id: int, error: str):
        """Mark a job as failed."""
        with self._lock:
            job = self._jobs.get(document_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error
                job.completed_at = time.monotonic()
                logger.error("Job failed for document %d: %s", document_id, error)

    def get_status(self, document_id: int) -> Optional[ProcessingJob]:
        """Get the current status of a processing job."""
        with self._lock:
            return self._jobs.get(document_id)

    def is_active(self, document_id: int) -> bool:
        """Check if a document is currently being processed."""
        with self._lock:
            job = self._jobs.get(document_id)
            return job is not None and job.status in (JobStatus.QUEUED, JobStatus.PROCESSING)

    def cleanup(self, max_age_seconds: float = 3600):
        """Remove completed/failed jobs older than max_age_seconds."""
        now = time.monotonic()
        with self._lock:
            to_remove = []
            for doc_id, job in self._jobs.items():
                if job.status in (JobStatus.READY, JobStatus.FAILED):
                    if job.completed_at and (now - job.completed_at) > max_age_seconds:
                        to_remove.append(doc_id)
            for doc_id in to_remove:
                del self._jobs[doc_id]
            if to_remove:
                logger.debug("Cleaned up %d old jobs", len(to_remove))


# Singleton instance
job_service = JobService()
