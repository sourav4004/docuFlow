from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets
import os
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration settings."""

    # Environment
    environment: str = "development"  # "development", "testing", "production"
    debug: bool = False  # Enable debug mode (disable in production)

    # Database
    database_url: str = "postgresql://docuflow_user:docuflow_password@localhost:5432/docuflow_db"
    db_pool_size: int = 5  # Connection pool size
    db_max_overflow: int = 10  # Max overflow connections
    db_pool_timeout: int = 30  # Connection timeout in seconds

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # CORS
    cors_origins: str = "http://localhost:3000"

    # Authentication
    secret_key: str = ""  # REQUIRED in production - generate with secrets.token_urlsafe(32)
    session_cookie_name: str = "docuflow_session"
    session_max_age: int = 86400 * 7  # 7 days in seconds
    session_idle_timeout: int = 0  # seconds of inactivity before expiry; 0 disables (production should set, e.g. 86400)
    cookie_secure: bool = False  # Set to True in production (HTTPS)
    cookie_samesite: str = "lax"  # "strict", "lax", or "none"

    # Storage
    storage_backend: str = "local"  # "local", "s3"
    storage_dir: str = "storage/documents"
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB in bytes

    # S3 Storage (when storage_backend == "s3")
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_endpoint_url: str = ""  # For S3-compatible services

    # Embedding
    embedding_provider: str = "fake"  # "fake", "openai"
    embedding_model: str = "fake-384"  # Model name for the provider (used by fake provider)
    embedding_api_key: str = ""  # API key (from env only, never hard-coded)
    embedding_dimension: int = 384  # Vector dimension

    # OpenAI embedding configuration (used when embedding_provider == "openai")
    openai_api_key: str = ""  # OPENAI_API_KEY
    openai_embedding_model: str = "text-embedding-3-small"  # OPENAI_EMBEDDING_MODEL
    openai_embedding_timeout: float = 60.0  # Timeout for embedding API calls

    # Retrieval
    retrieval_top_k: int = 5  # Number of results returned to the caller
    retrieval_min_similarity: float = 0.3  # Minimum score threshold for results
    retrieval_max_context_chars: int = 8000  # Max characters in LLM context
    retrieval_max_chunks_in_context: int = 20  # Max number of chunks in context

    # Vector Backend
    vector_backend: str = "auto"  # "auto", "pgvector", "json" - vector search backend

    # Hybrid Search
    hybrid_search_enabled: bool = True  # Enable hybrid (vector + keyword) retrieval
    vector_search_weight: float = 0.7  # Weight for semantic/vector score (0.0-1.0)
    keyword_search_weight: float = 0.3  # Weight for lexical/keyword score (0.0-1.0)
    hybrid_top_k: int = 20  # Number of candidates fetched before scoring/dedup
    keyword_only_top_k: int = 20  # Max results from keyword search alone
    vector_only_top_k: int = 20  # Max results from vector search in hybrid
    enable_keyword_fallback: bool = True  # Use keyword search when vector fails
    enable_vector_fallback: bool = True  # Use vector search when keyword fails

    # Multi-Document Context
    group_context_by_document: bool = True  # Group retrieval context chunks by document

    # Conversation Memory
    max_history_messages: int = 10  # Max conversation history messages for RAG
    max_history_chars: int = 4000  # Max characters from history included in prompt
    max_context_chars: int = 8000  # Max characters from document context

    # Confidence / Grounding
    confidence_enabled: bool = True  # Enable confidence scoring on RAG responses

    # LLM
    llm_provider: str = "fake"  # "fake", "openai_compatible"
    llm_model: str = "fake-llm"  # Model name for the provider
    llm_api_key: str = ""  # API key (from env only, never hard-coded)
    llm_base_url: str = "https://api.openai.com/v1"  # API base URL
    llm_timeout: float = 60.0  # Request timeout in seconds

    # Rate limiting
    rate_limit_enabled: bool = True  # Set to False to disable rate limiting (e.g. in tests)

    # Caching
    cache_enabled: bool = True  # Enable in-process caching
    cache_ttl_seconds: int = 300  # Default cache TTL (5 minutes)

    # Job Processing
    job_max_attempts: int = 3  # Maximum retry attempts for failed jobs
    job_stale_timeout_seconds: int = 300  # Mark processing jobs stale after 5 minutes

    # Data Retention
    session_retention_days: int = 30  # Clean expired sessions older than this
    job_retention_days: int = 7  # Clean completed/failed jobs older than this

    # API Keys
    api_key_max_per_workspace: int = 50  # Maximum active API keys per workspace
    api_key_default_ttl_days: int = 365  # Default API key expiration in days

    # Webhooks
    webhook_max_attempts: int = 5  # Max webhook delivery attempts
    webhook_retry_base_seconds: int = 60  # Exponential backoff base
    webhook_signature_tolerance_seconds: int = 300  # Replay protection window

    # Exports
    export_max_rows: int = 10000  # Max records in a single export
    export_download_ttl_hours: int = 24  # Download token validity

    # Retention
    retention_default_days: int = 30  # Default retention period
    retention_cleanup_batch: int = 500  # Records deleted per cleanup pass

    # SSO
    sso_state_ttl_minutes: int = 10  # OIDC state/nonce validity

    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"


def validate_settings() -> None:
    """Validate critical settings at startup.
    
    Raises:
        ValueError: If required production settings are missing.
    """
    if settings.is_production:
        # Production requires explicit secret_key
        if not settings.secret_key or settings.secret_key == "":
            raise ValueError(
                "SECRET_KEY is required in production mode. "
                "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        # Production should use secure cookies
        if not settings.cookie_secure:
            logger.warning("COOKIE_SECURE is False in production - cookies may be insecure")
        # Production should not have debug enabled
        if settings.debug:
            logger.warning("DEBUG mode is enabled in production - this is insecure")
    else:
        # Development/test: generate default secret if empty
        if not settings.secret_key:
            settings.secret_key = secrets.token_urlsafe(32)
            logger.info("Generated temporary SECRET_KEY for development")


settings = Settings()
validate_settings()
