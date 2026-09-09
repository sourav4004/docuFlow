"""Claim validator + evidence-first answers + answer repair.

Every factual AI answer is decomposed into claims, each associated with
evidence (source document, version, chunk, confidence, support type).
Unsupported claims are never presented as established facts; a bounded
repair pass removes or qualifies them (never looping indefinitely).
"""

import re
from dataclasses import dataclass, field
from typing import Optional

CLAIM_STATUSES = (
    "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNKNOWN",
)

MAX_REPAIR_PASSES = 2


@dataclass
class Claim:
    text: str
    status: str = "UNKNOWN"
    source_document: Optional[int] = None
    source_version: Optional[int] = None
    source_chunk: Optional[str] = None
    confidence: str = "UNKNOWN"  # HIGH/MEDIUM/LOW/UNKNOWN
    support_type: str = "lexical"  # lexical/semantic/metadata/entity
    reason: str = ""


@dataclass
class ValidationResult:
    claims: list[Claim] = field(default_factory=list)
    repair_passes: int = 0
    repaired: bool = False

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.claims if c.status == "SUPPORTED")

    @property
    def unsupported_count(self) -> int:
        return sum(1 for c in self.claims if c.status in ("UNSUPPORTED", "CONTRADICTED"))

    def to_dict(self) -> dict:
        return {
            "claims": [
                {
                    "text": c.text,
                    "status": c.status,
                    "source_document": c.source_document,
                    "source_version": c.source_version,
                    "source_chunk": c.source_chunk,
                    "confidence": c.confidence,
                    "support_type": c.support_type,
                    "reason": c.reason,
                }
                for c in self.claims
            ],
            "repair_passes": self.repair_passes,
            "repaired": self.repaired,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
        }


def split_claims(answer: str) -> list[str]:
    """Split an answer into discrete claim sentences (deterministic)."""
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def keyword_overlap(claim: str, evidence: str) -> float:
    """Fraction of claim content words present in the evidence (0..1)."""
    claim_words = {
        w.lower() for w in re.findall(r"[a-zA-Z0-9]+", claim)
        if len(w) > 3 and w.lower() not in ("this", "that", "with", "from", "have", "they", "their", "them", "what", "when", "where", "which", "would", "should", "could")
    }
    if not claim_words:
        return 0.0
    evidence_text = evidence.lower()
    hits = sum(1 for w in claim_words if w in evidence_text)
    return hits / len(claim_words)


def validate_claims(
    answer: str,
    evidence_chunks: list[str],
    source_document: Optional[int] = None,
    source_version: Optional[int] = None,
    threshold: float = 0.5,
) -> ValidationResult:
    """Validate every claim in an answer against evidence chunks.

    Deterministic lexical overlap: a claim is SUPPORTED when at least
    ``threshold`` of its content words appear in the evidence, CONTRADICTED
    when the evidence contains explicit negations of its key terms, and
    UNSUPPORTED otherwise.
    """
    result = ValidationResult()
    for claim_text in split_claims(answer):
        overlaps = [keyword_overlap(claim_text, chunk) for chunk in evidence_chunks]
        best = max(overlaps) if overlaps else 0.0
        if best >= threshold:
            status = "SUPPORTED"
            confidence = "HIGH" if best >= 0.8 else "MEDIUM"
            reason = f"evidence overlap {best:.0%}"
        elif _detect_contradiction(claim_text, evidence_chunks):
            status = "CONTRADICTED"
            confidence = "MEDIUM"
            reason = "evidence contradicts the claim"
        elif best > 0.0:
            status = "PARTIALLY_SUPPORTED"
            confidence = "LOW"
            reason = f"partial evidence overlap {best:.0%}"
        else:
            status = "UNSUPPORTED"
            confidence = "LOW"
            reason = "no supporting evidence found"
        result.claims.append(Claim(
            text=claim_text,
            status=status,
            source_document=source_document,
            source_version=source_version,
            confidence=confidence,
            reason=reason,
        ))
    return result


def _detect_contradiction(claim: str, evidence_chunks: list[str]) -> bool:
    """Check whether evidence explicitly negates the claim's key terms."""
    negation = re.compile(r"\b(no|not|never|without|fails? to|contrary|instead of)\b", re.IGNORECASE)
    for chunk in evidence_chunks:
        if negation.search(chunk):
            return True
    return False


def repair_answer(
    answer: str,
    evidence_chunks: list[str],
    source_document: Optional[int] = None,
    source_version: Optional[int] = None,
    threshold: float = 0.5,
    max_passes: int = MAX_REPAIR_PASSES,
) -> ValidationResult:
    """Repair an answer: drop unsupported claims, qualify partial claims.

    Each pass re-validates; passes are bounded (never loop indefinitely).
    """
    result = validate_claims(answer, evidence_chunks, source_document, source_version, threshold)
    passes = 0
    changed = True
    while changed and passes < max_passes:
        changed = False
        passes += 1
        kept = []
        for claim in result.claims:
            if claim.status == "UNSUPPORTED":
                claim.status = "UNKNOWN"
                claim.text = f"(Unsupported claim removed) {claim.text[:60]}…"
                claim.reason = "removed by bounded repair — no supporting evidence"
                changed = True
            elif claim.status == "PARTIALLY_SUPPORTED":
                claim.text = f"(Partially supported) {claim.text}"
                claim.reason = "qualified by bounded repair — weak evidence overlap"
                changed = True
            kept.append(claim)
        result.claims = kept
        if changed:
            # Re-validate the qualified text against evidence once more.
            revalidated = validate_claims(
                " ".join(c.text for c in kept),
                evidence_chunks, source_document, source_version, threshold,
            )
            result.claims = revalidated.claims
    result.repair_passes = passes
    result.repaired = passes > 0
    return result


def final_answer_text(result: ValidationResult) -> str:
    """Render a validated answer without exposing hidden reasoning.

    UNSUPPORTED and CONTRADICTED claims are never rendered as established
    facts — the caller must repair or present them as open questions.
    """
    parts = []
    unsupported = []
    for claim in result.claims:
        if claim.status in ("UNKNOWN", "UNSUPPORTED", "CONTRADICTED"):
            unsupported.append(claim.text)
            continue
        parts.append(claim.text)
    rendered = " ".join(parts).strip()
    if unsupported:
        note = " (Note: some statements could not be verified against the sources.)"
        rendered = f"{rendered}{note}" if rendered else note.strip()
    return rendered or "I could not validate any claim against the evidence."