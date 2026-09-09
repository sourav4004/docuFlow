"""Retrieval intelligence service for query expansion, reranking, and quality metrics."""

import re
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class QueryNormalizer:
    """Normalize queries for better retrieval."""
    
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "s", "t", "just", "don", "now",
    }
    
    @classmethod
    def normalize(cls, query: str) -> str:
        """Normalize query text."""
        # Lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters but keep meaningful ones
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)
        
        return normalized.strip()
    
    @classmethod
    def expand_synonyms(cls, query: str) -> List[str]:
        """Expand query with simple synonyms."""
        # Simple synonym mapping
        synonyms = {
            "buy": ["purchase", "acquire"],
            "sell": ["sell", "dispose"],
            "big": ["large", "major"],
            "small": ["little", "minor"],
            "fast": ["quick", "rapid"],
            "slow": ["gradual", "steady"],
        }
        
        words = query.lower().split()
        expanded = [query]
        
        for word in words:
            if word in synonyms:
                for synonym in synonyms[word]:
                    expanded.append(query.replace(word, synonym))
        
        return list(set(expanded))
    
    @classmethod
    def remove_stop_words(cls, query: str) -> str:
        """Remove stop words from query."""
        words = query.lower().split()
        filtered = [w for w in words if w not in cls.STOP_WORDS]
        return " ".join(filtered) if filtered else query


class QueryClassifier:
    """Classify query type for retrieval optimization."""
    
    QUERY_TYPES = {
        "factual": ["what", "who", "when", "where", "which", "how many", "how much"],
        "procedural": ["how to", "how do", "steps", "process", "procedure"],
        "comparative": ["compare", "difference", "versus", "vs", "better"],
        "exploratory": ["tell me about", "explain", "describe", "overview", "summary"],
        "specific": ["exact", "specific", "particular", "name", "identify"],
    }
    
    @classmethod
    def classify(cls, query: str) -> str:
        """Classify query type."""
        query_lower = query.lower()
        
        for qtype, keywords in cls.QUERY_TYPES.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return qtype
        
        return "general"


class ResultReranker:
    """Rerank retrieval results for better relevance."""
    
    @classmethod
    def rerank(
        cls,
        results: List[Dict[str, Any]],
        query: str,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank results based on multiple signals."""
        if not results:
            return []
        
        scored_results = []
        for result in results:
            score = cls._calculate_score(result, query)
            scored_results.append((score, result))
        
        # Sort by score descending
        scored_results.sort(reverse=True, key=lambda x: x[0])
        
        # Apply max_results limit
        if max_results:
            scored_results = scored_results[:max_results]
        
        return [r[1] for r in scored_results]
    
    @classmethod
    def _calculate_score(cls, result: Dict[str, Any], query: str) -> float:
        """Calculate composite score for a result."""
        score = 0.0
        
        # Base similarity score
        similarity = result.get("similarity", 0.0)
        score += similarity * 0.6
        
        # Keyword match bonus
        content = result.get("content", "").lower()
        query_words = query.lower().split()
        keyword_matches = sum(1 for w in query_words if w in content)
        keyword_bonus = (keyword_matches / max(len(query_words), 1)) * 0.2
        score += keyword_bonus
        
        # Recency bonus (if available)
        created_at = result.get("created_at")
        if created_at:
            score += 0.1  # Small bonus for having timestamp
        
        # Source quality bonus
        if result.get("source_type") == "verified":
            score += 0.1
        
        return score


class ResultDeduplicator:
    """Deduplicate retrieval results."""
    
    @classmethod
    def deduplicate(
        cls,
        results: List[Dict[str, Any]],
        content_threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """Remove duplicate or near-duplicate results."""
        if not results:
            return []
        
        unique_results = []
        seen_contents = []
        
        for result in results:
            content = result.get("content", "")
            is_duplicate = False
            
            for seen in seen_contents:
                if cls._similarity(content, seen) > content_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_contents.append(content)
        
        return unique_results
    
    @classmethod
    def _similarity(cls, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0


class RetrievalEvaluator:
    """Evaluate retrieval quality."""
    
    @classmethod
    def calculate_metrics(
        cls,
        retrieved: List[Dict[str, Any]],
        relevant: List[str],  # Document IDs that are relevant
        k: int = 10
    ) -> Dict[str, float]:
        """Calculate retrieval metrics."""
        if not retrieved:
            return {"precision": 0.0, "recall": 0.0, "mrr": 0.0}
        
        retrieved_ids = [r.get("document_id", r.get("id")) for r in retrieved[:k]]
        
        # Precision@k
        relevant_retrieved = sum(1 for rid in retrieved_ids if rid in relevant)
        precision = relevant_retrieved / k if k > 0 else 0.0
        
        # Recall@k
        recall = relevant_retrieved / len(relevant) if relevant else 0.0
        
        # MRR (Mean Reciprocal Rank)
        rr = 0.0
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in relevant:
                rr = 1.0 / i
                break
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mrr": round(rr, 4),
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant),
        }


class RetrievalDiagnostics:
    """Track and report retrieval diagnostics."""
    
    @classmethod
    def create_diagnostics(
        cls,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        final_results: List[Dict],
        query: str,
        duration_ms: float,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create retrieval diagnostics report."""
        return {
            "query": query[:100],  # Truncate for logging
            "vector_result_count": len(vector_results),
            "keyword_result_count": len(keyword_results),
            "final_result_count": len(final_results),
            "duration_ms": round(duration_ms, 2),
            "has_vector_results": len(vector_results) > 0,
            "has_keyword_results": len(keyword_results) > 0,
            "metadata": metadata or {},
        }
