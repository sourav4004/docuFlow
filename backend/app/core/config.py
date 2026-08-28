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
