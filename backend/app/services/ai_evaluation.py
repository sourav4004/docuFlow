"""AI Evaluation - Quality metrics and evaluation framework."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate
        }


@dataclass
class GroundingMetrics:
    """Metrics for answer grounding."""
    grounded_claims: int = 0
    partially_grounded_claims: int = 0
    unsupported_claims: int = 0
    grounding_score: float = 0.0
    citation_coverage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounded_claims": self.grounded_claims,
            "partially_grounded_claims": self.partially_grounded_claims,
            "unsupported_claims": self.unsupported_claims,
            "grounding_score": self.grounding_score,
            "citation_coverage": self.citation_coverage
        }


@dataclass
class ExtractionMetrics:
    """Metrics for structured extraction."""
    total_fields: int = 0
    extracted_fields: int = 0
    valid_fields: int = 0
    extraction_rate: float = 0.0
    validation_rate: float = 0.0
    schema_compliance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_fields": self.total_fields,
            "extracted_fields": self.extracted_fields,
            "valid_fields": self.valid_fields,
            "extraction_rate": self.extraction_rate,
            "validation_rate": self.validation_rate,
            "schema_compliance": self.schema_compliance
        }


@dataclass
class AIEvaluationResult:
    """Complete AI evaluation result."""
    id: str = ""
    task_type: str = ""
    query: str = ""
    retrieval_metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    grounding_metrics: GroundingMetrics = field(default_factory=GroundingMetrics)
    extraction_metrics: ExtractionMetrics = field(default_factory=ExtractionMetrics)
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    estimated_cost: float = 0.0
    model: str = ""
    provider: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "query": self.query,
            "retrieval_metrics": self.retrieval_metrics.to_dict(),
            "grounding_metrics": self.grounding_metrics.to_dict(),
            "extraction_metrics": self.extraction_metrics.to_dict(),
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "estimated_cost": self.estimated_cost,
            "model": self.model,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat()
        }


class RetrievalEvaluator:
    """Evaluates retrieval quality."""
    
    @classmethod
    def evaluate(
        cls,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 10
    ) -> RetrievalMetrics:
        """Evaluate retrieval metrics."""
        if not retrieved_ids or not relevant_ids:
            return RetrievalMetrics()
        
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        # Precision@k
        relevant_retrieved = retrieved_set & relevant_set
        precision = len(relevant_retrieved) / k if k > 0 else 0.0
        
        # Recall@k
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in relevant_set:
                mrr = 1.0 / i
                break
        
        # Hit Rate
        hit_rate = 1.0 if relevant_retrieved else 0.0
        
        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            hit_rate=hit_rate
        )


class GroundingEvaluator:
    """Evaluates answer grounding."""
    
    @classmethod
    def evaluate(
        cls,
        claims: List[str],
        evidence_texts: List[str],
        citations: List[Dict[str, Any]]
    ) -> GroundingMetrics:
        """Evaluate grounding metrics."""
        if not claims:
            return GroundingMetrics(
                grounded_claims=0,
                partially_grounded_claims=0,
                unsupported_claims=0,
                grounding_score=1.0,
                citation_coverage=1.0
            )
        
        evidence_text = " ".join(evidence_texts).lower()
        
        grounded = 0
        partially_grounded = 0
        unsupported = 0
        
        for claim in claims:
            claim_lower = claim.lower()
            claim_words = set(claim_lower.split())
            evidence_words = set(evidence_text.split())
            
            overlap = claim_words & evidence_words
            overlap_ratio = len(overlap) / max(len(claim_words), 1)
            
            if overlap_ratio > 0.6:
                grounded += 1
            elif overlap_ratio > 0.3:
                partially_grounded += 1
            else:
                unsupported += 1
        
        total = len(claims)
        grounding_score = (grounded + 0.5 * partially_grounded) / total if total > 0 else 0.0
        
        # Citation coverage
        cited_claims = len([c for c in citations if c.get("claim_id")])
        citation_coverage = cited_claims / total if total > 0 else 0.0
        
        return GroundingMetrics(
            grounded_claims=grounded,
            partially_grounded_claims=partially_grounded,
            unsupported_claims=unsupported,
            grounding_score=grounding_score,
            citation_coverage=citation_coverage
        )


class ExtractionEvaluator:
    """Evaluates structured extraction quality."""
    
    @classmethod
    def evaluate(
        cls,
        schema: Dict[str, Any],
        extracted: Dict[str, Any],
        validation_results: Optional[Dict[str, bool]] = None
    ) -> ExtractionMetrics:
        """Evaluate extraction metrics."""
        total_fields = len(schema)
        extracted_fields = len([k for k, v in extracted.items() if v is not None])
        
        valid_fields = extracted_fields
        if validation_results:
            valid_fields = len([k for k, v in validation_results.items() if v])
        
        extraction_rate = extracted_fields / total_fields if total_fields > 0 else 0.0
        validation_rate = valid_fields / extracted_fields if extracted_fields > 0 else 0.0
        schema_compliance = valid_fields / total_fields if total_fields > 0 else 0.0
        
        return ExtractionMetrics(
            total_fields=total_fields,
            extracted_fields=extracted_fields,
            valid_fields=valid_fields,
            extraction_rate=extraction_rate,
            validation_rate=validation_rate,
            schema_compliance=schema_compliance
        )


class AIEvaluationService:
    """Comprehensive AI evaluation service."""
    
    def __init__(self):
        self._evaluations: List[AIEvaluationResult] = []
    
    def record_evaluation(self, result: AIEvaluationResult):
        """Record an evaluation result."""
        self._evaluations.append(result)
    
    def get_evaluations(
        self,
        task_type: Optional[str] = None,
        workspace_id: Optional[int] = None,
        limit: int = 100
    ) -> List[AIEvaluationResult]:
        """Get evaluation results with filters."""
        results = self._evaluations
        
        if task_type:
            results = [r for r in results if r.task_type == task_type]
        
        return results[-limit:]
    
    def get_summary_stats(
        self,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary statistics."""
        results = self._evaluations
        
        if task_type:
            results = [r for r in results if r.task_type == task_type]
        
        if not results:
            return {}
        
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        avg_cost = sum(r.estimated_cost for r in results) / len(results)
        
        avg_grounding = sum(
            r.grounding_metrics.grounding_score for r in results
        ) / len(results)
        
        return {
            "total_evaluations": len(results),
            "avg_latency_ms": avg_latency,
            "avg_cost": avg_cost,
            "avg_grounding_score": avg_grounding
        }
