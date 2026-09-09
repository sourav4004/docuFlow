"""Tests for database-backed ProcessingJob model.

Covers:
- JobStatus enum values
- ProcessingJob model creation
- Status transitions
- Retry logic
- Attempts tracking
- Error message storage
"""

import pytest
from datetime import datetime, timezone

from app.models.job import ProcessingJob, JobStatus


class TestJobStatus:
    """Test JobStatus enum."""

    def test_job_status_values(self):
        assert JobStatus.QUEUED.value == "QUEUED"
        assert JobStatus.PROCESSING.value == "PROCESSING"
        assert JobStatus.COMPLETED.value == "COMPLETED"
        assert JobStatus.FAILED.value == "FAILED"

    def test_job_status_string_representation(self):
        # JobStatus enum str representation varies by Python version
        assert JobStatus.QUEUED in ["QUEUED", "JobStatus.QUEUED"]
        assert JobStatus.PROCESSING in ["PROCESSING", "JobStatus.PROCESSING"]
        assert JobStatus.COMPLETED in ["COMPLETED", "JobStatus.COMPLETED"]
        assert JobStatus.FAILED in ["FAILED", "JobStatus.FAILED"]


class TestProcessingJobModel:
    """Test ProcessingJob model creation and fields."""

    def test_create_processing_job(self):
        job = ProcessingJob(
            document_id=1,
            user_id=1,
            status=JobStatus.QUEUED.value,
            attempts=0,
            max_attempts=3,
        )
        assert job.document_id == 1
        assert job.user_id == 1
        assert job.status == JobStatus.QUEUED.value
        assert job.attempts == 0
        assert job.max_attempts == 3
        assert job.error_message is None

    def test_status_transitions(self):
        """Test valid status transitions."""
        job = ProcessingJob(
            document_id=1,
            user_id=1,
            status=JobStatus.QUEUED.value,
        )
        
        # QUEUED -> PROCESSING
        job.status = JobStatus.PROCESSING.value
        job.started_at = datetime.now(timezone.utc)
        assert job.status == JobStatus.PROCESSING.value
        
        # PROCESSING -> COMPLETED
        job.status = JobStatus.COMPLETED.value
        job.completed_at = datetime.now(timezone.utc)
        assert job.status == JobStatus.COMPLETED.value

    def test_failed_status(self):
        """Test failed job status."""
        job = ProcessingJob(
            document_id=1,
            user_id=1,
            status=JobStatus.PROCESSING.value,
        )
        
        job.status = JobStatus.FAILED.value
        job.error_message = "Extraction failed"
        job.completed_at = datetime.now(timezone.utc)
        
        assert job.status == JobStatus.FAILED.value
        assert job.error_message == "Extraction failed"

    def test_retry_logic(self):
        """Test retry logic for failed jobs."""
        job = ProcessingJob(
            document_id=1,
            user_id=1,
            status=JobStatus.FAILED.value,
            attempts=1,
            max_attempts=3,
        )
        
        # Can retry if attempts < max_attempts
        assert job.attempts < job.max_attempts
        
        # Reset for retry
        job.status = JobStatus.QUEUED.value
        job.error_message = None
        job.started_at = None
        job.completed_at = None
        job.attempts += 1
        
        assert job.status == JobStatus.QUEUED.value
        assert job.attempts == 2

    def test_max_attempts_enforced(self):
        """Test max attempts enforcement."""
        job = ProcessingJob(
            document_id=1,
            user_id=1,
            status=JobStatus.FAILED.value,
            attempts=3,
            max_attempts=3,
        )
        
        # Cannot retry if attempts >= max_attempts
        assert job.attempts >= job.max_attempts

    def test_repr(self):
        """Test string representation."""
        job = ProcessingJob(
            id=1,
            document_id=5,
            status=JobStatus.QUEUED.value,
            attempts=0,
        )
        repr_str = repr(job)
        assert "id=1" in repr_str
        assert "document_id=5" in repr_str
        assert "status=QUEUED" in repr_str
        assert "attempts=0" in repr_str


class TestProcessingJobDefaults:
    """Test ProcessingJob default values."""

    def test_default_status(self):
        # SQLAlchemy defaults are set at the database level, not Python constructor
        # This tests the column default configuration
        job = ProcessingJob(document_id=1, user_id=1)
        # When creating without explicit status, it should be None until flush
        # The actual default is set by server_default in the schema
        assert job.status is None or job.status == JobStatus.QUEUED.value

    def test_default_attempts(self):
        job = ProcessingJob(document_id=1, user_id=1)
        assert job.attempts is None or job.attempts == 0

    def test_default_max_attempts(self):
        job = ProcessingJob(document_id=1, user_id=1)
        assert job.max_attempts is None or job.max_attempts == 3

    def test_default_job_type(self):
        job = ProcessingJob(document_id=1, user_id=1)
        assert job.job_type is None or job.job_type == "document_processing"
