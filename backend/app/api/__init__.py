from fastapi import APIRouter

from .health import router as health_router
from .auth import router as auth_router
from .protected import router as protected_router
from .documents import router as documents_router
from .rag import router as rag_router

api_router = APIRouter()

# Include all API routers
api_router.include_router(health_router, tags=["health"])
api_router.include_router(auth_router)
api_router.include_router(protected_router)
api_router.include_router(documents_router)
api_router.include_router(rag_router)

