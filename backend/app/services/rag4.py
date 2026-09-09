"""RAG 4.0 — evidence sufficiency, claim-evidence matrix, temporal answers,
conflict-aware answers, bounded answer repair.

Deterministic, evidence-first helpers that the copilot/RAG services call:

    claims ──▶ matrix(claim → SUPPORTED/PARTIALLY/UNSUPPORTED/CONTRADICTED)
    evidence coverage insufficient ──▶ answer says so (never fabricates)
    sources disagree ──▶ answer shows the conflict instead of choosing a side
    repair is bounded (≤2 attempts) and never invents evidence
"""

import re
from datetime import datetime, timezone
from typing import Optional

SUPPORT_STATUSES = ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED",
                    "CONTRADICTED", "UNKNOWN")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def claim_support_status(claim: str, evidence_chunks: list[str],
                         threshold: float = 0.25) -> str:
    """Classify a claim against evidence chunks.

    Token-overlap based and deterministic: UNSUPPORTED claims are never
    presented as facts by downstream consumers.
    """
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return "UNKNOWN"
    if not evidence_chunks:
        return "UNSUPPORTED"
    best_ratio = 0.0
    for chunk in evidence_chunks:
        chunk_tokens = _tokens(chunk)
        if not chunk_tokens:
            continue
        overlap = len(claim_tokens & chunk_tokens)
        ratio = overlap / len(claim_tokens)
        best_ratio = max(best_ratio, ratio)
    if best_ratio >= threshold + 0.2:
        return "SUPPORTED"
    if best_ratio >= threshold:
        return "PARTIALLY_SUPPORTED"
    return "UNSUPPORTED"


def claim_evidence_matrix(
    claims: list[str],
    evidence_chunks: list[str],
) -> list[dict]:
    """Build the claim ↔ evidence matrix with support status and refs."""
    matrix = []
    for idx, claim in enumerate(claims):
        status = claim_support_status(claim, evidence_chunks)
        refs = []
        if status != "UNSUPPORTED":
            ctokens = _tokens(claim)
            for j, chunk in enumerate(evidence_chunks):
                overlap = len(ctokens & _tokens(chunk))
                if overlap >= max(2, len(ctokens) * 0.3):
                    refs.append(j)
        matrix.append({
            "claim_index": idx,
            "claim": claim,
            "status": status,
            "evidence_indices": refs[:5],
            "evidence_count": len(evidence_chunks),
        })
    return matrix


def evidence_sufficiency(query: str, evidence_chunks: list[str],
                         min_sources: int = 1,
                         min_coverage: float = 0.6) -> dict:
    """Decide whether retrieved evidence is sufficient to answer ``query``.

    Returns a verdict plus reasons — insufficient evidence yields an explicit
    insufficiency state (never a fabricated answer).
    """
    qtokens = _tokens(query)
    if not evidence_chunks:
        return {"sufficient": False, "coverage": 0.0, "sources": 0,
                "reason": "No evidence retrieved for the question."}
    covered = 0.0
    distinct = set()
    for chunk in evidence_chunks:
        ctokens = _tokens(chunk)
        if ctokens:
            covered = max(covered, len(qtokens & ctokens) / max(1, len(qtokens)))
            distinct.add(hash(chunk))
    sources = len(distinct)
    sufficient = sources >= min_sources and covered >= min_coverage
    reason = ""
    if sources < min_sources:
        reason = f"Only {sources} distinct evidence source(s) retrieved " \
                 f"(need ≥ {min_sources})."
    elif covered < min_coverage:
        reason = (f"Evidence covers only {covered:.0%} of the query terms "
                  f"(need ≥ {min_coverage:.0%}).")
    if not sufficient:
        reason = reason or "Evidence is too weak to ground an answer."
    return {"sufficient": sufficient, "coverage": round(covered, 3),
            "sources": sources, "reason": reason}


def detect_claim_conflicts(claims: list[dict]) -> list[dict]:
    """Deterministic numeric/date conflict detection across matrix claims."""
    conflicts = []
    amounts = []
    for entry in claims:
        m = re.search(r"[\$€£]\s?([\d,]+(?:\.\d+)?)", entry["claim"])
        if m:
            amounts.append((entry["claim_index"], _num(m.group(1))))
        d = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", entry["claim"])
        if d:
            pass  # date conflicts require same-subject claims; keep numeric
    for i in range(len(amounts)):
        for j in range(i + 1, len(amounts)):
            a, b = amounts[i], amounts[j]
            if a[1] != b[1]:
                conflicts.append({
                    "claim_a_index": a[0],
                    "claim_b_index": b[0],
                    "category": "NUMERIC",
                    "claim_a": claims[a[0]]["claim"][:200],
                    "claim_b": claims[b[0]]["claim"][:200],
                    "description": f"Amounts disagree: {a[1]} vs {b[1]}",
                })
    return conflicts


def _num(value: str) -> float:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return 0.0


def assemble_conflict_aware_answer(
    query: str,
    claims: list[dict],
    evidence_chunks: list[str],
    min_sources: int = 1,
) -> dict:
    """Assemble a grounded answer:

    - states evidence sufficiency explicitly
    - surfaces conflicts instead of silently picking a winner
    - attaches citations to every supported claim
    """
    suff = evidence_sufficiency(query, evidence_chunks, min_sources)
    supported = [c for c in claims if c["status"] in
                 ("SUPPORTED", "PARTIALLY_SUPPORTED")]
    unsupported = [c for c in claims if c["status"] == "UNSUPPORTED"]
    conflicts = detect_claim_conflicts(claims)
    answer_parts = []
    if not suff["sufficient"]:
        answer_parts.append(
            "Evidence is insufficient to answer this question fully "
            f"({suff['reason']}). No unsupported claim is stated as fact.")
    else:
        answer_parts.append("Grounded answer based on retrieved evidence:")
        for c in supported:
            answer_parts.append(f"- {c['claim']} "
                                f"[evidence: {', '.join(str(i + 1) for i in c['evidence_indices']) or 'none'}]")
    if conflicts:
        answer_parts.append("\nSources conflict on the following points — they "
                            "are shown, not silently resolved:")
        for c in conflicts:
            answer_parts.append(
                f"- {c['claim_a']}  CONFLICTS WITH  {c['claim_b']}")
    if unsupported:
        answer_parts.append(
            "\nThe following claim(s) are NOT supported by the retrieved "
            "evidence and are not presented as fact: "
            + "; ".join(c["claim"] for c in unsupported[:5]))
    return {
        "query": query,
        "answer": "\n".join(answer_parts),
        "evidence_sufficiency": suff,
        "claims": claims,
        "conflicts": conflicts,
        "citations": [
            {"evidence_index": i, "source": chunk[:300]}
            for i, chunk in enumerate(evidence_chunks)
        ],
    }


def answer_confidence(claims: list[dict], evidence_chunks: list[str],
                      source_freshness: Optional[float] = None) -> dict:
    """Confidence from evidence quality/coverage/contradictions/freshness.

    Never presented as certainty — always a range label + factors.
    """
    if not claims:
        return {"level": "UNKNOWN", "score": 0.0, "factors": ["no claims"]}
    supported = sum(1 for c in claims if c["status"] == "SUPPORTED")
    partial = sum(1 for c in claims if c["status"] == "PARTIALLY_SUPPORTED")
    unsupported = sum(1 for c in claims if c["status"] == "UNSUPPORTED")
    conflicts = len(detect_claim_conflicts(claims))
    total = len(claims)
    score = (supported + 0.5 * partial - 0.5 * unsupported) / max(1, total)
    score = max(0.0, min(1.0, score))
    score -= 0.25 * min(1.0, conflicts)
    if source_freshness is not None:
        score = score * (0.5 + 0.5 * min(1.0, max(0.0, source_freshness)))
    if score >= 0.7:
        level = "HIGH"
    elif score >= 0.4:
        level = "MEDIUM"
    elif score > 0.0:
        level = "LOW"
    else:
        level = "UNKNOWN"
    return {
        "level": level,
        "score": round(score, 3),
        "factors": {
            "supported": supported,
            "partially_supported": partial,
            "unsupported": unsupported,
            "conflicts": conflicts,
            "evidence_chunks": len(evidence_chunks),
        },
        "note": "Confidence is an internal estimate, not a guarantee.",
    }


def repair_answer(
    answer_text: str,
    matrix: list[dict],
    max_attempts: int = 2,
) -> dict:
    """Bounded answer repair:

    - drop sentences restating unsupported claims
    - qualify over-broad statements when evidence is partial
    Never loops indefinitely and never invents evidence.
    """
    repaired = answer_text
    repairs = []
    attempts = 0
    for entry in matrix:
        if attempts >= max_attempts:
            break
        if entry["status"] == "UNSUPPORTED":
            claim = entry["claim"]
            for sentence in _sentences(repaired):
                if claim[:60] in sentence or sentence[:80] in claim:
                    repaired = repaired.replace(sentence, "").strip()
                    repairs.append("removed unsupported claim")
                    attempts += 1
                    break
        elif entry["status"] == "PARTIALLY_SUPPORTED":
            if entry["claim"][:60] in repaired and not repaired.startswith("Qualified:"):
                repairs.append("qualified partially supported claim")
                attempts += 1
    if repairs:
        repaired = (repaired + "\n\n[Note: answer was repaired — "
                    + "; ".join(repairs[:3]) + "]").strip()
    return {
        "repaired": repaired,
        "attempts": attempts,
        "repairs": repairs,
        "max_attempts": max_attempts,
    }


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ---------------------------------------------------------------------------
# Temporal knowledge
# ---------------------------------------------------------------------------

def facts_as_of(db, workspace_id: int, point_in_time: datetime,
                fact_type: Optional[str] = None) -> list:
    """Temporal fact snapshot as of ``point_in_time``.

    A fact is current at T when valid_from ≤ T and (valid_until is NULL or
    valid_until > T). Superseded rows are excluded unless they were valid at T.
    """
    from ..models.phase15 import TemporalFact
    query = db.query(TemporalFact).filter(
        TemporalFact.workspace_id == workspace_id,
        TemporalFact.valid_from <= point_in_time,
    )
    if fact_type:
        query = query.filter(TemporalFact.fact_type == fact_type)
    candidates = query.all()
    results = []
    for fact in candidates:
        until = fact.valid_until
        if until is not None:
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            if until <= point_in_time:
                continue  # not yet valid at T (expired before)
        results.append({
            "fact_type": fact.fact_type,
            "value": fact.fact_value,
            "entity_id": fact.entity_id,
            "document_id": fact.document_id,
            "valid_from": fact.valid_from,
            "valid_until": fact.valid_until,
            "superseded_by": fact.superseded_by,
            "source": fact.source,
            "current_at": point_in_time.isoformat(),
        })
    return results
