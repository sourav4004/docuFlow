from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets


class Settings(BaseSettings):
    """Application configuration settings."""

    # Database
    database_url: str = "postgresql://docuflow_user:docuflow_password@localhost:5432/docuflow_db"

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # CORS
    cors_origins: str = "http://localhost:3000"

    # Authentication
    secret_key: str = secrets.token_urlsafe(32)  # Generate secure default
    session_cookie_name: str = "docuflow_session"
    session_max_age: int = 86400 * 7  # 7 days in seconds
    cookie_secure: bool = False  # Set to True in production (HTTPS)
    cookie_samesite: str = "lax"  # "strict", "lax", or "none"

    # Storage
    storage_dir: str = "storage/documents"
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB in bytes

    # Embedding
    embedding_provider: str = "fake"  # "fake", "openai"
    embedding_model: str = "fake-384"  # Model name for the provider
    embedding_api_key: str = ""  # API key (from env only, never hard-coded)
    embedding_dimension: int = 384  # Vector dimension

    # Retrieval
    retrieval_top_k: int = 5  # Number of vector search candidates
    retrieval_min_similarity: float = 0.3  # Minimum cosine similarity threshold
    retrieval_max_context_chars: int = 8000  # Max characters in LLM context

    # LLM
    llm_provider: str = "fake"  # "fake", "openai_compatible"
    llm_model: str = "fake-llm"  # Model name for the provider
    llm_api_key: str = ""  # API key (from env only, never hard-coded)
    llm_base_url: str = "https://api.openai.com/v1"  # API base URL
    llm_timeout: float = 60.0  # Request timeout in seconds

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()
