from fastapi import APIRouter

from .health import router as health_router
from .auth import router as auth_router
from .protected import router as protected_router
from .documents import router as documents_router
from .rag import router as rag_router
from .conversations import router as conversations_router
from .collections import router as collections_router
from .workspaces import router as workspaces_router
from .invitations import router as invitations_router
from .notifications import router as notifications_router
from .comments import router as comments_router
from .ai_assistant import router as ai_router
from .agents import router as agents_router
from .workflows import router as workflows_router
from .organizations import router as organizations_router
from .api_keys import router as api_keys_router
from .webhooks import router as webhooks_router
from .usage import router as usage_router
from .exports import router as exports_router
from .security_center import router as security_center_router
from .integrations import router as integrations_router
from .ai_actions import router as ai_actions_router
from .knowledge import router as knowledge_router
from .copilot import router as copilot_router
from .search_intel import router as search_intel_router
from .deadlines import router as deadlines_router
from .feedback import router as feedback_router
from .reports import router as reports_router
from .research import router as research_router
from .executions import router as executions_router
from .events import router as events_router
from .knowledge_os import router as knowledge_os_router
from .memory import router as memory_router
from .reviews import router as reviews_router
from .automation2 import router as automation2_router
from .governance import router as governance_router
from .costs import router as costs_router
from .worker_ops import jobs_router, ops_router
from .backfill_api import router as backfill_router
from .pages_api import router as pages_router
from .policy_api import router as policy_router
from .distributed17 import router as distributed17_router
from .ingestion17 import router as ingestion17_router
from .knowledge17 import router as knowledge17_router
from .platform17 import router as platform17_router
from .ops18 import router as ops18_router
from .ops19 import router as ops19_router
from .ops20 import router as ops20_router
from .ops21 import router as ops21_router
from .ops22 import router as ops22_router
from .ops23 import router as ops23_router

api_router = APIRouter()

# Include all API routers
api_router.include_router(health_router, tags=["health"])
api_router.include_router(auth_router)
api_router.include_router(protected_router)
api_router.include_router(documents_router)
api_router.include_router(rag_router)
api_router.include_router(conversations_router)
api_router.include_router(collections_router)
api_router.include_router(workspaces_router)
api_router.include_router(invitations_router)
api_router.include_router(notifications_router)
api_router.include_router(comments_router)
api_router.include_router(ai_router)
api_router.include_router(agents_router)
api_router.include_router(workflows_router)
api_router.include_router(organizations_router)
api_router.include_router(api_keys_router)
api_router.include_router(webhooks_router)
api_router.include_router(usage_router)
api_router.include_router(exports_router)
api_router.include_router(security_center_router)
api_router.include_router(integrations_router)
api_router.include_router(ai_actions_router)
api_router.include_router(knowledge_router)
api_router.include_router(copilot_router)
api_router.include_router(search_intel_router)
api_router.include_router(deadlines_router)
api_router.include_router(feedback_router)
api_router.include_router(reports_router)
api_router.include_router(research_router)
api_router.include_router(executions_router)
api_router.include_router(events_router)
api_router.include_router(knowledge_os_router)
api_router.include_router(memory_router)
api_router.include_router(reviews_router)
api_router.include_router(automation2_router)
api_router.include_router(governance_router)
api_router.include_router(costs_router)
api_router.include_router(jobs_router)
api_router.include_router(ops18_router)
api_router.include_router(ops19_router)
api_router.include_router(ops20_router)
api_router.include_router(ops21_router)
api_router.include_router(ops22_router)
api_router.include_router(ops23_router)
api_router.include_router(ops_router)
api_router.include_router(backfill_router)
api_router.include_router(pages_router)
api_router.include_router(policy_router)
api_router.include_router(distributed17_router)
api_router.include_router(ingestion17_router)
api_router.include_router(knowledge17_router)
api_router.include_router(platform17_router)

# Versioned compatibility layer: /api/v1/... mirrors the current API without
# breaking existing (prefix-less) routes. Future v2 will live separately.
v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(api_router)

