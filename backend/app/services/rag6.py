"""Phase 18 RAG 6.0 — evidence-first retrieval with planner, ranking,
sufficiency, claim matrix, citation coverage/correctness, refusal, and
bounded repair.

Retrieved content is ALWAYS treated as data, never as instructions.
Insufficient evidence produces an explicit refusal instead of a guess.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

MAX_REPAIR_ROUNDS = 2
MAX_EVIDENCE = 8


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Retrieval planner
# ---------------------------------------------------------------------------

def retrieval_plan(query: str, filters: Optional[dict] = None) -> dict:
    """Deterministic strategy selection: keyword / vector / hybrid / graph /
    memory / metadata — never exposes reasoning, only the chosen plan."""
    q = (query or "").strip().lower()
    strategy = "hybrid"
    sources: list[str] = ["keyword", "vector"]
    if filters and filters.get("entity_id"):
        strategy = "graph"
        sources = ["graph", "keyword"]
    elif filters and filters.get("memory"):
        strategy = "memory"
        sources = ["memory", "keyword"]
    elif len(q.split()) <= 2:
        strategy = "hybrid"
    elif any(token in q for token in
             ("who", "when", "where", "what is", "define")):
        strategy = "hybrid"
    return {
        "strategy": strategy,
        "sources": sources,
        "scope": filters or {},
        "max_evidence": MAX_EVIDENCE,
        "explainable": True,
    }


# ---------------------------------------------------------------------------
# Evidence ranking
# ---------------------------------------------------------------------------

def rank_evidence(chunks: list[dict], weights: Optional[dict] = None) -> list[dict]:
    """Score evidence by relevance, authority, freshness, completeness and
    contradiction state. Deterministic tie-break by chunk id."""
    w = {"relevance": 0.45, "freshness": 0.25, "authority": 0.15,
         "completeness": 0.15}
    if weights:
        w.update({k: float(v) for k, v in weights.items()
                  if k in w and float(v) >= 0})
    scored = []
    for idx, chunk in enumerate(chunks):
        relevance = float(chunk.get("score", chunk.get("relevance", 0.0)) or 0.0)
        freshness = 1.0
        updated = chunk.get("updated_at") or chunk.get("created_at")
        if updated is not None:
            age_days = max(0.0, (_utcnow() - _dt(updated)).total_seconds()
                           / 86400.0)
            freshness = max(0.0, 1.0 - age_days / 1095.0)  # 3-year decay
        authority = float(chunk.get("authority", 0.5) or 0.5)
        completeness = float(chunk.get("completeness", 0.5) or 0.5)
        conflict_penalty = 0.85 if chunk.get("has_conflict") else 1.0
        total = ((w["relevance"] * relevance
                  + w["freshness"] * freshness
                  + w["authority"] * authority
                  + w["completeness"] * completeness)
                 * conflict_penalty)
        scored.append({
            "chunk_id": chunk.get("chunk_id", chunk.get("id", idx)),
            "document_id": chunk.get("document_id"),
            "text": chunk.get("text", "")[:2000],
            "page": chunk.get("page"),
            "score": round(total, 4),
            "relevance": round(relevance, 3),
            "freshness": round(freshness, 3),
            "authority": round(authority, 3),
            "completeness": round(completeness, 3),
            "source": chunk.get("source", "document"),
        })
    scored.sort(key=lambda e: (-e["score"], str(e["chunk_id"])))
    return scored


def _dt(value) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return _utcnow()


def evidence_sufficiency(evidence: list[dict], query: str,
                         min_evidence: int = 2,
                         min_coverage: float = 0.5) -> dict:
    """Determine whether evidence is sufficient — never fabricate when not."""
    relevant = [e for e in evidence
                if e.get("relevance", 0.0) >= 0.3]
    coverage = min(1.0, len(relevant) / max(min_evidence, 1))
    sufficient = (len(relevant) >= min_evidence
                  and coverage >= min_coverage)
    return {
        "sufficient": sufficient,
        "evidence_count": len(evidence),
        "relevant_count": len(relevant),
        "coverage": round(coverage, 3),
        "suggestion": None if sufficient else (
            "More specific query or additional documents are required to "
            "answer with evidence."),
        "refusal": not sufficient,
    }


# ---------------------------------------------------------------------------
# Claim matrix + citation coverage
# ---------------------------------------------------------------------------

def claim_matrix(claims: list[str], evidence: list[dict]) -> dict:
    """Map every claim to its supporting evidence references."""
    evidence_texts = [{"id": e["chunk_id"], "text": e.get("text", "").lower()}
                      for e in evidence]
    rows = []
    for claim in claims:
        claim_lower = claim.lower()
        terms = [t for t in re.split(r"[^a-z0-9']+", claim_lower) if len(t) > 3]
        refs = []
        for ev in evidence_texts:
            hits = sum(1 for t in terms if t in ev["text"])
            if hits and len(terms):
                ratio = hits / len(terms)
                if ratio >= 0.25:
                    refs.append({"evidence_id": ev["id"],
                                 "support": "SUPPORTED" if ratio >= 0.5
                                 else "PARTIAL"})
        rows.append({
            "claim": claim,
            "evidence": refs,
            "support_status": ("SUPPORTED" if any(
                r["support"] == "SUPPORTED" for r in refs)
                else "PARTIAL" if refs else "UNSUPPORTED"),
        })
    supported = sum(1 for r in rows if r["support_status"] != "UNSUPPORTED")
    return {
        "claims": rows,
        "coverage": round(supported / len(rows), 3) if rows else 1.0,
        "unsupported": [r["claim"] for r in rows
                        if r["support_status"] == "UNSUPPORTED"],
    }


def citation_correctness(claims: list[str], evidence: list[dict]) -> dict:
    """Verify cited evidence actually supports each claim."""
    matrix = claim_matrix(claims, evidence)
    total = len(matrix["claims"])
    correct = sum(1 for r in matrix["claims"]
                  if r["support_status"] == "SUPPORTED")
    return {
        "total_claims": total,
        "correct_citations": correct,
        "coverage": round(correct / total, 3) if total else 1.0,
        "unsupported_claims": matrix["unsupported"],
    }


# ---------------------------------------------------------------------------
# Bounded answer repair
# ---------------------------------------------------------------------------

def repair_answer(answer: str, claims: list[str],
                  evidence: list[dict],
                  max_rounds: int = MAX_REPAIR_ROUNDS) -> dict:
    """Repair ONLY unsupported claims using retrieved evidence; never invent
    evidence. Bounded attempts; returns the repair log."""
    repaired = answer
    rounds = 0
    changes = []
    unsupported = claim_matrix(claims, evidence)["unsupported"]
    while unsupported and rounds < max_rounds:
        changed = False
        for claim in unsupported:
            # Replace an unsupported assertion with an explicit uncertainty
            # marker anchored to the evidence, not a guess.
            marker = f"[UNSUPPORTED: evidence insufficient for: {claim[:80]}]"
            if claim in repaired:
                repaired = repaired.replace(claim, marker, 1)
                changes.append({"claim": claim, "action": "flagged_unsupported"})
                changed = True
        unsupported = [c for c in unsupported
                       if c not in repaired]
        rounds += 1
        if not changed:
            break
    return {
        "repaired": repaired,
        "rounds": rounds,
        "max_rounds": max_rounds,
        "changes": changes,
        "fully_supported": not unsupported,
    }


# ---------------------------------------------------------------------------
# Refusal / conflict-aware answer assembly
# ---------------------------------------------------------------------------

def assemble_answer(query: str, evidence: list[dict],
                    draft: Optional[str] = None) -> dict:
    """Deterministic assembly with refusal and conflict surfacing."""
    suff = evidence_sufficiency(evidence, query)
    if not suff["sufficient"]:
        return {
            "answer": None,
            "refused": True,
            "reason": suff["suggestion"],
            "evidence_count": len(evidence),
        }
    conflicts = detect_conflicts(evidence)
    conflict_note = ""
    if conflicts:
        conflict_note = ("\n\nNote: sources conflict on "
                         + ", ".join(sorted({c["field"] for c in conflicts}))
                         + "; both versions are preserved below.")
    answer = draft or (
        "Answer synthesized from the evidence above" + conflict_note)
    return {
        "answer": answer,
        "refused": False,
        "conflicts": conflicts,
        "evidence_count": len(evidence),
        "citations": [e["chunk_id"] for e in evidence],
    }


def detect_conflicts(evidence: list[dict]) -> list[dict]:
    """Deterministic numeric/date conflict detection across evidence."""
    conflicts = []
    numbers: dict[str, list] = {}
    for e in evidence:
        for m in re.finditer(
                r"(\d+(?:\.\d+)?)\s*(%|percent|pct|USD|EUR|GBP|\$|€|£|days?)",
                e.get("text", ""), re.IGNORECASE):
            field = m.group(2)
            numbers.setdefault(field, []).append(
                {"value": float(m.group(1)), "evidence_id": e["chunk_id"]})
    for field, vals in numbers.items():
        if len(vals) < 2:
            continue
        values = {v["value"] for v in vals}
        if len(values) > 1:
            conflicts.append({
                "field": field,
                "values": sorted(values),
                "evidence_ids": [v["evidence_id"] for v in vals],
                "severity": "CONFLICT",
            })
    return conflicts