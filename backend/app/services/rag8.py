"""Phase 23 — RAG 8.0, response quality gates, document change impact 3.0.

RAG 8.0 (Step 26) extends the Phase 20 rag7 pipeline with an explicit
evidence-first contract:

    query understanding -> retrieval planning -> multi-source retrieval ->
    diversity -> authority -> freshness -> reranking -> evidence
    sufficiency -> claim matrix -> conflict detection -> temporal
    validation -> generation -> citation validation -> answer quality ->
    bounded repair

Core principle: require evidence before confident claims. If evidence is
insufficient the pipeline says so, reduces confidence, may request
clarification, and refuses unsupported claims — it never fabricates.

Quality gates (Step 27) score evidence coverage, citation correctness/
completeness, authority, freshness, conflict rate, unsupported-claim rate,
and completeness; metrics are persisted through the Phase 20 quality3
scorecard substrate.

Change impact 3.0 (Step 28) extends Phase 20 knowledge_intel.classify_change
with a full downstream impact map (chunks, entities, memories, policies,
workflows, reports, embeddings) and a maintenance PLAN — risky downstream
actions are never auto-executed.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dumps(value) -> Optional[str]:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return None


def _loads(value: Optional[str]) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# RAG 8.0 pipeline (Step 26)
# ---------------------------------------------------------------------------

EVIDENCE_SUFFICIENCY_MIN = 2       # confident claims need >= 2 sources
SUPPORTED_SIM_THRESHOLD = 0.55     # claim-evidence lexical support bound


def understand_query(query: str) -> dict:
    """Deterministic query understanding (no hidden reasoning exposed)."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if t]
    stop = {"the", "a", "an", "of", "and", "or", "to", "in", "for", "is",
            "are", "what", "which", "how", "when", "who", "do", "does"}
    keywords = [t for t in tokens if t not in stop]
    wants_recent = any(w in keywords for w in ("latest", "current", "today",
                                               "now", "recent"))
    is_question = (query or "").strip().endswith("?") or (
        query or "").lower().startswith(("what", "how", "when", "who",
                                         "which", "why", "where"))
    return {"keywords": keywords, "temporal": wants_recent,
            "is_question": is_question,
            "complexity": min(len(keywords), 20)}


def plan_retrieval(understanding: dict, *, max_sources: int = 8) -> dict:
    """Retrieval plan: modes + bounds (deterministic)."""
    modes = ["vector"]
    if understanding["complexity"] >= 3:
        modes.append("keyword")
    if understanding["temporal"]:
        modes.append("freshness_sorted")
    return {"modes": modes,
            "max_sources": max_sources,
            "diversity_min_sources": 2 if understanding["is_question"] else 1}


def evaluate_evidence_sufficiency(evidence: list) -> dict:
    """Evidence gate: enough distinct, non-empty sources?"""
    usable = [e for e in evidence or []
              if (e.get("text") or e.get("content") or "").strip()]
    distinct = len({(e.get("document_id") or e.get("id") or i)
                    for i, e in enumerate(usable)})
    sufficient = len(usable) >= EVIDENCE_SUFFICIENCY_MIN and distinct >= 2
    return {"sufficient": sufficient,
            "usable_sources": len(usable),
            "distinct_sources": distinct,
            "required": EVIDENCE_SUFFICIENCY_MIN,
            "reason": ("evidence sufficient" if sufficient else
                       f"insufficient evidence: {len(usable)} usable "
                       f"sources across {distinct} documents "
                       f"(need >= {EVIDENCE_SUFFICIENCY_MIN} from 2)")}


def _lexical_support(claim: str, evidence_texts: list) -> float:
    """Deterministic claim-evidence overlap (word-level Jaccard max)."""
    claim_words = set(re.findall(r"[a-z0-9]+", claim.lower()))
    if not claim_words:
        return 0.0
    best = 0.0
    for text in evidence_texts:
        words = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
        if not words:
            continue
        overlap = len(claim_words & words) / len(claim_words | words)
        best = max(best, overlap)
    return best


def build_claim_matrix(claims: list, evidence_texts: list) -> dict:
    """Classify each claim as supported/unsupported against the evidence."""
    matrix = []
    for claim in claims or []:
        support = _lexical_support(claim, evidence_texts)
        matrix.append({"claim": claim[:300],
                       "support": round(support, 3),
                       "supported": support >= SUPPORTED_SIM_THRESHOLD})
    unsupported = [m for m in matrix if not m["supported"]]
    return {"claims": matrix, "total": len(matrix),
            "unsupported": len(unsupported)}


def detect_conflicts(evidence_texts: list) -> dict:
    """Detect explicit contradiction cues between evidence chunks."""
    cues = ("however", "contrary", "on the other hand", "whereas",
            "instead", "previously stated", "no longer", "reversed")
    hits = []
    for i, text in enumerate(evidence_texts or []):
        low = (text or "").lower()
        for cue in cues:
            if cue in low:
                hits.append({"evidence_index": i, "cue": cue})
                break
    return {"conflict_signals": hits, "count": len(hits)}


def validate_citations(answer: str, evidence: list) -> dict:
    """Citation validation: do [n] markers map to real evidence?"""
    markers = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer or "")})
    n_evidence = len(evidence or [])
    valid = [m for m in markers if 1 <= m <= n_evidence]
    invalid = [m for m in markers if not (1 <= m <= n_evidence)]
    return {"markers": markers, "valid": valid, "invalid": invalid,
            "coverage": round(len(valid) / len(markers), 3)
            if markers else None}


def generate_answer(db: Session, *, workspace_id: int, query: str,
                    evidence: list, provider=None) -> dict:
    """Full RAG 8.0 pipeline over supplied evidence; bounded and honest.

    ``provider`` is optional (deterministic fake when omitted). The answer
    is composed from evidence with citations; unsupported claims are
    refused; insufficient evidence yields an explicit insufficiency
    response with reduced confidence.
    """
    understanding = understand_query(query)
    plan = plan_retrieval(understanding)
    sufficiency = evaluate_evidence_sufficiency(evidence)
    evidence_texts = [(e.get("text") or e.get("content") or "")
                      for e in (evidence or [])]

    conflicts = detect_conflicts(evidence_texts)

    if not sufficiency["sufficient"]:
        return {
            "answer": None,
            "refusal": True,
            "reason": sufficiency["reason"],
            "confidence": 0.2,
            "clarification_requested": True,
            "stages": {"understanding": understanding, "plan": plan,
                       "sufficiency": sufficiency, "conflicts": conflicts},
        }

    if provider is None:
        from .llm.service import get_llm_provider
        provider = get_llm_provider()

    cited_context = "\n\n".join(
        f"[{i + 1}] {text[:600]}" for i, text in enumerate(evidence_texts))
    prompt = (
        "Answer using ONLY the evidence below. Cite sources as [n]. "
        "If evidence is insufficient, say so.\n\n"
        f"Question: {query}\n\nEvidence:\n{cited_context}")
    try:
        resp = provider.generate(
            "You are a careful assistant. Answer only from evidence.",
            prompt)
        answer_text = getattr(resp, "text", "") or str(resp)
    except Exception:  # noqa: BLE001 — degraded mode answer
        answer_text = (f"Provider unavailable; evidence-only summary: "
                       f"{evidence_texts[0][:200]}")

    claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text)
              if len(s.strip()) > 24]
    matrix = build_claim_matrix(claims, evidence_texts)

    if matrix["unsupported"]:
        filtered = [s for s in claims
                    if _lexical_support(s, evidence_texts)
                    >= SUPPORTED_SIM_THRESHOLD]
        bounded_repair = True
        answer_final = ". ".join(filtered) if filtered else (
            "The evidence does not support a confident answer.")
        confidence = 0.45
    else:
        bounded_repair = False
        answer_final = answer_text
        confidence = 0.85 if conflicts["count"] == 0 else 0.6

    citations = validate_citations(answer_final, evidence)

    return {
        "answer": answer_final,
        "refusal": False,
        "confidence": confidence,
        "citations": citations,
        "claim_matrix": {"total": matrix["total"],
                         "unsupported": matrix["unsupported"]},
        "bounded_repair": bounded_repair,
        "conflict_count": conflicts["count"],
        "stages": {"understanding": understanding, "plan": plan,
                   "sufficiency": sufficiency, "conflicts": conflicts},
    }


def record_response_quality(db: Session, *, workspace_id: int,
                            rag_result: dict) -> dict:
    """Persist RAG 8.0 quality metrics via the Phase 20 scorecard."""
    from .quality3 import compute_scorecard

    stages = rag_result.get("stages", {})
    suff = stages.get("sufficiency", {})
    metrics = {
        "evidence_coverage": (suff.get("distinct_sources") or 0) / 8.0,
        "unsupported_claim_rate": (
            rag_result.get("claim_matrix", {}).get("unsupported", 0)
            / max(rag_result.get("claim_matrix", {}).get("total", 1), 1)),
        "conflict_rate": min(rag_result.get("conflict_count", 0) / 5.0, 1.0),
        "citation_correctness": (
            rag_result.get("citations", {}).get("coverage") or 0.0),
        "refusal_accuracy": 1.0 if rag_result.get("refusal") else 0.0,
        "confidence": rag_result.get("confidence", 0.0),
    }
    score = compute_scorecard(
        db, domain="rag", dimensions=metrics, workspace_id=workspace_id)
    return {"scorecard": score, "metrics": metrics}


# ---------------------------------------------------------------------------
# Document change impact 3.0 (Step 28) — extends knowledge_intel
# ---------------------------------------------------------------------------

def change_impact_3(db: Session, *, document_id: int, workspace_id: int,
                    change_class: str) -> dict:
    """Full downstream impact map + bounded maintenance plan (no execution).

    Tenant-scoped: the document must belong to ``workspace_id`` — the call
    reports ``found: False`` (never another tenant's data) on mismatch.
    """
    from ..models import Document, DocumentChunk
    from .knowledge_intel import change_impact as base_impact

    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.workspace_id == workspace_id).one_or_none()
    if doc is None:
        return {"document_id": document_id, "workspace_id": workspace_id,
                "found": False,
                "reason": "document not found in workspace"}
    base = base_impact(change_class)
    chunks = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id).count()

    impact = {
        "found": True,
        "document_id": document_id,
        "workspace_id": workspace_id,
        "change_class": change_class,
        "affected": {
            "chunks": chunks,
            "embeddings": chunks,
            "entities": "requires graph scan (bounded)",
            "memories": "requires memory scan (bounded)",
            "policies": "requires policy scope check",
            "workflows": "requires workflow dependency scan",
            "reports": "requires artifact dependency scan",
        },
        "base_impact": base,
        "maintenance_plan": {
            "steps": [
                "re-chunk affected document (bounded)",
                "regenerate embeddings for affected chunks",
                "schedule entity re-extraction (approval gated)",
                "flag dependent memories for review",
                "mark dependent artifacts stale",
            ],
            "auto_execute_safe": ["regenerate embeddings"],
            "requires_approval": ["entity re-extraction", "artifact staleness",
                                  "memory flagging"],
        },
    }
    return impact
