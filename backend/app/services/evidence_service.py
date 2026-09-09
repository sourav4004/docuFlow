"""Evidence Service - Evidence pack creation and grounding validation."""

from typing import Dict, List, Any, Optional
from .ai_orchestration import (
    EvidencePack, GroundingResult, GroundingStatus, ConflictResult
)


class EvidenceService:
    """Service for creating evidence packs and validating grounding."""
    
    @classmethod
    def create_evidence_pack(
        cls,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        source_documents: List[Dict[str, Any]],
        retrieval_method: str = "hybrid"
    ) -> EvidencePack:
        """Create an evidence pack from retrieved chunks."""
        evidence = EvidencePack(
            query=query,
            retrieved_chunks=retrieved_chunks,
            source_documents=source_documents,
            retrieval_method=retrieval_method,
            evidence_count=len(retrieved_chunks)
        )
        
        # Extract relevance scores
        evidence.relevance_scores = [
            chunk.get("similarity", 0.0) for chunk in retrieved_chunks
        ]
        
        # Extract page information
        evidence.page_information = [
            {
                "document_id": chunk.get("document_id"),
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "chunk_index": chunk.get("chunk_index"),
            }
            for chunk in retrieved_chunks
            if chunk.get("page") or chunk.get("section")
        ]
        
        return evidence
    
    @classmethod
    def validate_grounding(
        cls,
        answer: str,
        evidence: EvidencePack,
        claims: Optional[List[str]] = None
    ) -> GroundingResult:
        """Validate that claims in the answer are grounded in evidence."""
        if not claims:
            claims = cls._extract_claims(answer)
        
        if not claims:
            return GroundingResult(
                status=GroundingStatus.GROUNDED,
                grounding_score=1.0,
                evidence_coverage=1.0
            )
        
        # Combine all evidence text
        evidence_text = " ".join([
            chunk.get("content", "") for chunk in evidence.retrieved_chunks
        ]).lower()
        
        grounded = []
        partially_grounded = []
        unsupported = []
        
        for claim in claims:
            claim_lower = claim.lower()
            
            # Check for direct evidence
            if cls._has_direct_evidence(claim_lower, evidence_text):
                grounded.append(claim)
            # Check for partial evidence
            elif cls._has_partial_evidence(claim_lower, evidence_text):
                partially_grounded.append(claim)
            else:
                unsupported.append(claim)
        
        total_claims = len(claims)
        grounding_score = (len(grounded) + 0.5 * len(partially_grounded)) / max(total_claims, 1)
        evidence_coverage = (len(grounded) + len(partially_grounded)) / max(total_claims, 1)
        
        # Determine overall status
        if unsupported:
            status = GroundingStatus.PARTIALLY_GROUNDED if partially_grounded else GroundingStatus.UNSUPPORTED
        elif partially_grounded:
            status = GroundingStatus.PARTIALLY_GROUNDED
        else:
            status = GroundingStatus.GROUNDED
        
        return GroundingResult(
            status=status,
            grounded_claims=grounded,
            partially_grounded_claims=partially_grounded,
            unsupported_claims=unsupported,
            grounding_score=grounding_score,
            evidence_coverage=evidence_coverage
        )
    
    @classmethod
    def detect_conflicts(
        cls,
        documents: List[Dict[str, Any]]
    ) -> List[ConflictResult]:
        """Detect potential conflicts between documents."""
        conflicts = []
        
        # Compare each pair of documents
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                doc_a = documents[i]
                doc_b = documents[j]
                
                # Check for factual conflicts
                doc_conflicts = cls._check_factual_conflicts(doc_a, doc_b)
                conflicts.extend(doc_conflicts)
                
                # Check for temporal conflicts
                temporal_conflicts = cls._check_temporal_conflicts(doc_a, doc_b)
                conflicts.extend(temporal_conflicts)
        
        return conflicts
    
    @classmethod
    def _extract_claims(cls, text: str) -> List[str]:
        """Extract factual claims from text (simplified)."""
        sentences = text.split(".")
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                # Simple heuristic: sentences with numbers or dates are claims
                if any(c.isdigit() for c in sentence):
                    claims.append(sentence)
                # Sentences with absolute statements
                elif any(word in sentence.lower() for word in ["is", "are", "was", "were", "must", "shall"]):
                    claims.append(sentence)
        
        return claims[:10]  # Limit to 10 claims
    
    @classmethod
    def _has_direct_evidence(cls, claim: str, evidence_text: str) -> bool:
        """Check if claim has direct evidence."""
        # Simple word overlap check
        claim_words = set(claim.split())
        evidence_words = set(evidence_text.split())
        
        overlap = claim_words & evidence_words
        return len(overlap) / max(len(claim_words), 1) > 0.6
    
    @classmethod
    def _has_partial_evidence(cls, claim: str, evidence_text: str) -> bool:
        """Check if claim has partial evidence."""
        claim_words = set(claim.split())
        evidence_words = set(evidence_text.split())
        
        overlap = claim_words & evidence_words
        return len(overlap) / max(len(claim_words), 1) > 0.3
    
    @classmethod
    def _check_factual_conflicts(
        cls,
        doc_a: Dict[str, Any],
        doc_b: Dict[str, Any]
    ) -> List[ConflictResult]:
        """Check for factual conflicts between two documents."""
        conflicts = []
        
        # Simple keyword-based conflict detection
        content_a = doc_a.get("content", "").lower()
        content_b = doc_b.get("content", "").lower()
        
        # Check for contradictory statements (simplified)
        contradiction_patterns = [
            (r"must not", r"must"),
            (r"shall not", r"shall"),
            (r"is required", r"is optional"),
            (r"is prohibited", r"is allowed"),
        ]
        
        for negative, positive in contradiction_patterns:
            if re.search(negative, content_a) and re.search(positive, content_b):
                conflicts.append(ConflictResult(
                    document_a_id=doc_a.get("id", 0),
                    document_b_id=doc_b.get("id", 0),
                    claim_a=f"Document contains '{negative}'",
                    claim_b=f"Document contains '{positive}'",
                    conflict_type="policy",
                    severity="medium",
                    confidence=0.7
                ))
        
        return conflicts
    
    @classmethod
    def _check_temporal_conflicts(
        cls,
        doc_a: Dict[str, Any],
        doc_b: Dict[str, Any]
    ) -> List[ConflictResult]:
        """Check for temporal conflicts between documents."""
        conflicts = []
        
        date_a = doc_a.get("effective_date")
        date_b = doc_b.get("effective_date")
        
        if date_a and date_b and date_a != date_b:
            # Different effective dates might indicate version conflicts
            conflicts.append(ConflictResult(
                document_a_id=doc_a.get("id", 0),
                document_b_id=doc_b.get("id", 0),
                claim_a=f"Effective date: {date_a}",
                claim_b=f"Effective date: {date_b}",
                conflict_type="temporal",
                severity="low",
                confidence=0.5
            ))
        
        return conflicts


import re  # Import at module level for use in methods
