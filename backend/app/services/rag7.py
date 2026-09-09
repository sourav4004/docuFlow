"""Phase 19 — RAG 7.0.

Query intent planning 2.0 (8 classes), explicit retrieval plans over
multiple sources, evidence diversity (near-duplicate suppression), evidence
authority ranking (relevance/authority/freshness/completeness/confidence),
evidence sufficiency gating before generation, claim matrices 2.0 with
supporting/contradicting evidence, citation correctness + coverage,
conflict-aware answers, temporal reasoning (as-of/current/between),
insufficient-evidence refusal, bounded answer repair (loop-capped, evidence-
only), and a RAG quality score.

All functions are deterministic and side-effect free so they are directly
unit-testable.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

INTENTS = ("factual", "comparative", "temporal", "policy", "analytical",
           "entity", "procedural", "investigative", "multi_document")


def _now() -> datetime:
    return datetime.now(timezone.utc)


_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be",
               "been", "of", "to", "in", "for", "on", "and", "or",
               "at", "by", "with", "as", "it", "this", "that", "do",
               "does", "did", "can", "could", "will", "would", "should"}


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9']+", (text or "").lower())
            if t not in _STOPWORDS}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Intent planner 2.0
# ---------------------------------------------------------------------------

def intent_planner_v2(query: str) -> dict:
    """Deterministic query-intent classification (heuristic, single label
    with the strongest signal first)."""
    q = (query or "").lower()
    reasons = []
    if re.search(r"\b(as of|in \d{4}|before \d{4}|after \d{4}|current|"
                 r"historical|between \d{4}|until|when did|was this)\b", q):
        reasons.append("temporal markers present")
    if re.search(r"\b(vs\.?|versus|compare|comparison|difference|"
                 r"better|cheaper|faster than)\b", q):
        reasons.append("comparative markers present")
    if re.search(r"\b(how do i|steps|procedure|process|guide|workflow)\b",
                 q):
        reasons.append("procedural markers present")
    if re.search(r"\b(why did|root cause|investigate|what happened|"
                 r"escalat)\b", q):
        reasons.append("investigative markers present")
    if re.search(r"\b(analy|trend|impact|evaluate|forecast)\b", q):
        reasons.append("analytical markers present")
    if re.search(r"\b(across|all documents|multi-document|every "
                 r"workspace)\b", q):
        reasons.append("multi-document markers present")
    policy = re.search(r"\b(policy|approval|must|shall|require|compliance"
                       r"|rule)\b", q)
    if policy and "policy" in q:
        reasons.append("policy markers present")
    if not reasons:
        reasons.append("no strong markers; default factual")
    # multi-document overrides single-doc analytical choices conservatively
    order = ("temporal", "comparative", "procedural", "investigative",
             "policy", "analytical", "multi_document", "entity", "factual")
    chosen = "factual"
    for intent in order:
        marker = {
            "temporal": "temporal markers present",
            "comparative": "comparative markers present",
            "procedural": "procedural markers present",
            "investigative": "investigative markers present",
            "analytical": "analytical markers present",
            "policy": "policy markers present",
            "multi_document": "multi-document markers present",
        }.get(intent)
        if marker in reasons:
            chosen = intent
            break
    if re.search(r"\b(@|mr\.?|ms\.?|inc|llc|ltd|corp|company)\b",
                 q.lower()) and chosen == "factual":
        chosen = "entity"
    return {"intent": chosen, "reasons": reasons,
            "label": f"intent={chosen}"}


def retrieval_plan_v2(query: str) -> dict:
    """Explicit multi-source retrieval plan for the classified intent."""
    intent = intent_planner_v2(query)["intent"]
    source_map = {
        "factual": ["keyword", "vector", "metadata"],
        "comparative": ["keyword", "vector", "workspace"],
        "temporal": ["vector", "keyword", "versions"],
        "policy": ["keyword", "metadata", "workspace"],
        "analytical": ["vector", "keyword", "memory", "graph"],
        "entity": ["keyword", "graph", "metadata"],
        "procedural": ["keyword", "vector"],
        "investigative": ["vector", "graph", "memory", "connector"],
        "multi_document": ["workspace", "vector", "connector"],
    }
    return {"query": query, "intent": intent,
            "sources": source_map.get(intent, ["keyword", "vector"]),
            "strategy": "hybrid" if len(source_map.get(intent)) > 1
            else "keyword",
            "note": "retrieval plan is deterministic and explainable"}


# ---------------------------------------------------------------------------
# Evidence selection: diversity + authority
# ---------------------------------------------------------------------------

def evidence_diversity(evidence: list[dict],
                       overlap_threshold: float = 0.7) -> list[dict]:
    """Remove near-identical evidence chunks (bigram-ish token overlap)."""
    kept: list[dict] = []
    for item in evidence:
        dup = False
        for kept_item in kept:
            if _overlap(kept_item.get("content", ""),
                        item.get("content", "")) >= overlap_threshold:
                dup = True
                break
        if not dup:
            kept.append(item)
    return kept


def evidence_authority_rank(evidence: list[dict],
                            weights: Optional[dict] = None) -> list[dict]:
    """Rank by relevance, source authority, freshness, completeness,
    confidence. Missing numeric fields degrade to 0.5 so ranking stays
    deterministic."""
    weights = weights or {"relevance": 0.4, "authority": 0.25,
                          "freshness": 0.15, "completeness": 0.1,
                          "confidence": 0.1}
    ranked = []
    for item in evidence:
        item = dict(item)
        authority = float(item.get("authority") or _default_authority(
            item.get("source") or ""))
        freshness = float(item.get("freshness")
                          or (0.9 if item.get("date") is None else 0.5))
        relevance = float(item.get("relevance") or 0.5)
        completeness = float(item.get("completeness") or 0.5)
        confidence = float(item.get("confidence") or 0.5)
        score = (weights["relevance"] * relevance
                 + weights["authority"] * authority
                 + weights["freshness"] * freshness
                 + weights["completeness"] * completeness
                 + weights["confidence"] * confidence)
        item["_authority_score"] = round(score, 4)
        ranked.append(item)
    ranked.sort(key=lambda i: (-i["_authority_score"],
                               str(i.get("id") or "")))
    return ranked


def _default_authority(source: str) -> float:
    lowered = source.lower()
    if "policy" in lowered or "official" in lowered:
        return 0.9
    if "document" in lowered:
        return 0.8
    if "connector" in lowered:
        return 0.6
    if "memory" in lowered:
        return 0.5
    return 0.5


# ---------------------------------------------------------------------------
# Sufficiency + refusal
# ---------------------------------------------------------------------------

def evidence_sufficiency2(evidence: list[dict], query: str,
                          min_sources: int = 1,
                          min_support: float = 0.3) -> dict:
    """Gate generation on evidence sufficiency — never hallucinate."""
    deduped = evidence_diversity(evidence)
    # Conservative default: evidence without an explicit relevance score is
    # NOT counted as relevant — generation is gated, never optimistic.
    relevant = [e for e in deduped
                if _overlap(e.get("content", ""), query) >= min_support
                or float(e.get("relevance") or 0.0) >= min_support]
    sufficient = len(deduped) >= min_sources and bool(relevant)
    return {"sufficient": sufficient, "candidates": len(deduped),
            "relevant": len(relevant), "min_sources": min_sources,
            "reason": ("sufficient evidence found" if sufficient
                       else "insufficient evidence — refusing to generate "
                            "unsupported claims")}


def refuse_unsupported(query: str, evidence: list[dict]) -> dict:
    check = evidence_sufficiency2(evidence, query)
    if check["sufficient"]:
        return {"refuse": False, "reason": None}
    return {"refuse": True,
            "reason": "I do not have enough evidence in the workspace to "
                      "answer this safely. Please refine the question or "
                      "provide the relevant document."}


# ---------------------------------------------------------------------------
# Claims, citations, conflicts
# ---------------------------------------------------------------------------

def _support_for(claim: str, evidence: dict,
                 threshold: float = 0.3) -> bool:
    return _overlap(evidence.get("content", ""), claim) >= threshold


def claim_matrix2(claims: list[str], evidence: list[dict],
                  overlap_threshold: float = 0.3) -> dict:
    rows = []
    for claim in claims:
        supporting = []
        contradicting = []
        for idx, item in enumerate(evidence):
            if item.get("contradicts"):
                targets = item["contradicts"]
                if claim in targets or idx in targets:
                    contradicting.append(item.get("id"))
                    continue
            if _support_for(claim, item, overlap_threshold):
                supporting.append(item.get("id"))
        negated = bool(re.search(r"\b(not|never|no longer|cannot|doesn'?t"
                                 r"|must not)\b", claim.lower()))
        confidence = "HIGH" if supporting and not negated else (
            "LOW" if not supporting else "MEDIUM")
        rows.append({"claim": claim, "supporting_evidence": supporting,
                     "contradicting_evidence": contradicting,
                     "confidence": confidence})
    return {"claims": rows, "total": len(rows),
            "supported": sum(1 for r in rows
                             if r["supporting_evidence"])}


def citation_correctness2(claim: str, evidence: list[dict],
                          threshold: float = 0.3) -> dict:
    """Verify cited evidence actually supports the claim."""
    support_ids = []
    for item in evidence:
        if _support_for(claim, item, threshold):
            support_ids.append(item.get("id"))
    return {"claim": claim, "correct": bool(support_ids),
            "supporting_evidence_ids": support_ids,
            "citation_ok": bool(support_ids)}


def citation_coverage2(claims: list[str], evidence: list[dict],
                       overlap_threshold: float = 0.3) -> dict:
    matrix = claim_matrix2(claims, evidence, overlap_threshold)
    total = max(1, matrix["total"])
    supported = matrix["supported"]
    return {"supported_claims": supported, "total_claims": matrix["total"],
            "coverage": round(supported / total, 4),
            "note": "factual claims must have supporting citations where "
                    "required"}


def detect_conflicts2(evidence: list[dict]) -> list[dict]:
    """Surface meaningful evidence conflicts instead of silently choosing."""
    conflicts = []
    seen = set()
    for idx, item in enumerate(evidence):
        content = str(item.get("content") or "")
        negated = bool(re.search(r"\b(not|never|no longer|prohibit|cannot"
                                 r"|must not)\b", content.lower()))
        for other in evidence[idx + 1:]:
            other_content = str(other.get("content") or "")
            if _overlap(content, other_content) < 0.25:
                continue
            other_negated = bool(re.search(r"\b(not|never|no longer|prohibit"
                                           r"|cannot|must not)\b",
                                           other_content.lower()))
            if negated != other_negated:
                key = tuple(sorted([str(item.get("id")),
                                    str(other.get("id"))]))
                if key not in seen:
                    seen.add(key)
                    conflicts.append({"evidence_a": item.get("id"),
                                      "evidence_b": other.get("id"),
                                      "type": "CONTRADICTION",
                                      "detail": "conflicting statements "
                                                "detected — surfaced, not "
                                                "silently resolved"})
    return conflicts


# ---------------------------------------------------------------------------
# Temporal reasoning
# ---------------------------------------------------------------------------

def temporal_filter(evidence: list[dict], *, as_of: Optional[str] = None,
                    between_from: Optional[str] = None,
                    between_to: Optional[str] = None) -> list[dict]:
    """Honor effective/expiration/publication dates. No evidence date ⇒
    treated as currently valid (never excluded by accident)."""
    target = None
    if as_of is not None:
        try:
            target = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except ValueError:
            target = None
    out = []
    for item in evidence:
        effective = item.get("valid_from") or item.get("date")
        expires = item.get("valid_to") or item.get("expires_at")
        effective_dt = _parse_dt(effective)
        expires_dt = _parse_dt(expires)
        if target is not None:
            if effective_dt is not None and effective_dt > target:
                continue
            if expires_dt is not None and expires_dt <= target:
                continue
        if between_from or between_to:
            start = _parse_dt(between_from)
            end = _parse_dt(between_to)
            if effective_dt is not None:
                if start is not None and effective_dt < start:
                    continue
                if end is not None and effective_dt > end:
                    continue
        out.append(item)
    return out


def _parse_dt(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Bounded repair + quality score
# ---------------------------------------------------------------------------

def repair_answer2(answer: str, claims: list[str], evidence: list[dict],
                   max_rounds: int = 2) -> dict:
    """Repair unsupported claims using retrieved evidence only. Bounded:
    never more than ``max_rounds`` repair passes."""
    rounds = 0
    repaired = list(claims)
    for _ in range(min(max_rounds, 5)):
        rounds += 1
        unsupported = [c for c in repaired
                       if not any(_support_for(c, e)
                                  for e in evidence)]
        if not unsupported:
            break
        # drop unsupported claims from the repair candidate set (they are
        # not rephrased into existence — no evidence, no claim)
        repaired = [c for c in repaired if c not in unsupported]
    supported = [c for c in repaired
                 if any(_support_for(c, e) for e in evidence)]
    return {"rounds": rounds, "repaired_claims": supported,
            "removed_unsupported": len(claims) - len(repaired),
            "loop_terminated": True}


def rag_quality_score(*, query: str, claims: list[str],
                      evidence: list[dict]) -> dict:
    sufficiency = evidence_sufficiency2(evidence, query)
    coverage = citation_coverage2(claims, evidence)
    conflicts = detect_conflicts2(evidence)
    correctness = sum(
        1 for claim in claims if citation_correctness2(claim, evidence)
        ["citation_ok"])
    total_claims = max(1, len(claims))
    contradiction_level = round(min(1.0, len(conflicts) / max(1,
                                                              len(evidence))),
                                3)
    score = round((0.35 * (1.0 if sufficiency["sufficient"] else 0.0)
                   + 0.25 * coverage["coverage"]
                   + 0.25 * (correctness / total_claims)
                   + 0.15 * (1.0 - contradiction_level)), 4)
    return {"score": score, "evidence_sufficiency":
            sufficiency["sufficient"],
            "citation_coverage": coverage["coverage"],
            "citation_correctness": round(correctness / total_claims, 4),
            "contradiction_level": contradiction_level,
            "conflict_count": len(conflicts),
            "grade": "EXCELLENT" if score >= 0.9 else (
                "GOOD" if score >= 0.75 else ("FAIR" if score >= 0.6
                                              else "POOR"))}
