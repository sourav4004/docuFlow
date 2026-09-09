from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router, v1_router
from .core.config import settings
from .core.errors import install_error_handlers
from .core.rate_limit import RateLimitMiddleware
from .core.security_headers import SecurityHeadersMiddleware
from .core.observability import ObservabilityMiddleware, setup_observability

# Configure observability
setup_observability()

app = FastAPI(
    title="DocuFlow API",
    description="AI-powered document operations platform",
    version="0.2.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Observability (correlation IDs, request timing)
app.add_middleware(ObservabilityMiddleware)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Security headers
app.add_middleware(SecurityHeadersMiddleware)

# Unified error model: structured JSON errors, no stack leaks (Phase 24)
install_error_handlers(app)

# Include API routes (unversioned + /api/v1 compatibility layer)
app.include_router(api_router)
app.include_router(v1_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "DocuFlow API",
        "version": "0.1.0",
        "status": "running"
    }
