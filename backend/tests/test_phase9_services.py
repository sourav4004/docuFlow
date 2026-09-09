"""Tests for Phase 9 services: document intelligence and retrieval intelligence."""

import pytest
from app.services.document_intelligence import (
    DocumentClassifier,
    MetadataExtractor,
    DocumentSummarizer,
    EntityExtractor,
)
from app.services.retrieval_intelligence import (
    QueryNormalizer,
    QueryClassifier,
    ResultReranker,
    ResultDeduplicator,
    RetrievalEvaluator,
    RetrievalDiagnostics,
)


# ============================================================
# Document Classification Tests
# ============================================================

class TestDocumentClassifier:
    def test_classify_invoice(self):
        content = "Invoice #12345. Amount due: $500. Payment terms: Net 30."
        result = DocumentClassifier.classify(content, "invoice.pdf")
        assert result["document_type"] == "invoice"
        assert result["confidence"] > 0.0

    def test_classify_contract(self):
        content = "This agreement is between Party A and Party B. Terms and conditions apply."
        result = DocumentClassifier.classify(content, "contract.docx")
        assert result["document_type"] == "contract"
        assert result["confidence"] > 0.0

    def test_classify_report(self):
        content = "Report: Analysis of quarterly performance. Findings show growth."
        result = DocumentClassifier.classify(content, "quarterly_report.pdf")
        assert result["document_type"] == "report"

    def test_classify_unknown(self):
        content = "Short content"
        result = DocumentClassifier.classify(content, "file.txt")
        assert result["document_type"] == "unknown"
        assert result["confidence"] == 0.0

    def test_filename_bonus(self):
        content = "Invoice details and payment information"
        result_with = DocumentClassifier.classify(content, "invoice.pdf")
        result_without = DocumentClassifier.classify(content, "document.pdf")
        assert result_with["confidence"] >= result_without["confidence"]


# ============================================================
# Metadata Extraction Tests
# ============================================================

class TestMetadataExtractor:
    def test_extract_basic_metadata(self):
        content = "This is a test document with some content."
        metadata = MetadataExtractor.extract_metadata(content, "test.txt")
        assert "word_count" in metadata
        assert "character_count" in metadata
        assert "filename" in metadata
        assert metadata["word_count"] > 0

    def test_extract_title(self):
        content = "Document Title\nThis is the content of the document."
        metadata = MetadataExtractor.extract_metadata(content)
        assert "title" in metadata
        assert metadata["title"] == "Document Title"

    def test_language_detection(self):
        content = "The quick brown fox jumps over the lazy dog. This is English text with many common words. " * 10
        metadata = MetadataExtractor.extract_metadata(content)
        assert metadata["language"] == "en"

    def test_date_extraction(self):
        content = "Created on 2024-01-15. Updated on 01/20/2024."
        metadata = MetadataExtractor.extract_metadata(content)
        assert "dates_found" in metadata
        assert len(metadata["dates_found"]) >= 1


# ============================================================
# Document Summarization Tests
# ============================================================

class TestDocumentSummarizer:
    def test_extract_key_points(self):
        content = "First sentence. Second sentence is longer. Third sentence with more detail. Fourth sentence."
        points = DocumentSummarizer.extract_key_points(content, max_points=3)
        assert len(points) <= 3
        assert len(points) > 0

    def test_short_summary(self):
        content = "This is sentence one. This is sentence two. This is sentence three with more words."
        summary = DocumentSummarizer.generate_short_summary(content)
        assert len(summary) > 0
        assert len(summary) < len(content)

    def test_empty_content(self):
        assert DocumentSummarizer.extract_key_points("") == []
        assert DocumentSummarizer.generate_short_summary("") == ""


# ============================================================
# Entity Extraction Tests
# ============================================================

class TestEntityExtractor:
    def test_extract_emails(self):
        content = "Contact us at support@example.com or sales@company.org"
        entities = EntityExtractor.extract_entities(content)
        assert "email" in entities
        assert len(entities["email"]) == 2

    def test_extract_phones(self):
        content = "Call us at 555-123-4567 or 555.987.6543"
        entities = EntityExtractor.extract_entities(content)
        assert "phone" in entities

    def test_empty_content(self):
        entities = EntityExtractor.extract_entities("")
        assert entities == {}


# ============================================================
# Query Normalizer Tests
# ============================================================

class TestQueryNormalizer:
    def test_normalize_basic(self):
        normalized = QueryNormalizer.normalize("  What is   the  API?  ")
        assert normalized == "what is the api"

    def test_remove_stop_words(self):
        result = QueryNormalizer.remove_stop_words("What is the API documentation")
        assert "the" not in result
        assert "is" not in result

    def test_expand_synonyms(self):
        expanded = QueryNormalizer.expand_synonyms("buy this product")
        assert len(expanded) > 1


# ============================================================
# Query Classifier Tests
# ============================================================

class TestQueryClassifier:
    def test_classify_factual(self):
        qtype = QueryClassifier.classify("What is the API key?")
        assert qtype == "factual"

    def test_classify_procedural(self):
        qtype = QueryClassifier.classify("How to create a document?")
        assert qtype == "procedural"

    def test_classify_general(self):
        qtype = QueryClassifier.classify("random query")
        assert qtype == "general"


# ============================================================
# Result Reranker Tests
# ============================================================

class TestResultReranker:
    def test_rerank_by_similarity(self):
        results = [
            {"content": "Low similarity content", "similarity": 0.3},
            {"content": "High similarity content", "similarity": 0.9},
            {"content": "Medium similarity content", "similarity": 0.6},
        ]
        reranked = ResultReranker.rerank(results, "similarity query")
        assert reranked[0]["similarity"] == 0.9

    def test_rerank_empty(self):
        assert ResultReranker.rerank([], "query") == []

    def test_rerank_with_max_results(self):
        results = [{"content": f"Result {i}", "similarity": 0.5 + i * 0.1} for i in range(5)]
        reranked = ResultReranker.rerank(results, "query", max_results=3)
        assert len(reranked) == 3


# ============================================================
# Result Deduplicator Tests
# ============================================================

class TestResultDeduplicator:
    def test_deduplicate_exact(self):
        results = [
            {"content": "This is the same content"},
            {"content": "This is the same content"},
            {"content": "Different content"},
        ]
        deduplicated = ResultDeduplicator.deduplicate(results)
        assert len(deduplicated) == 2

    def test_deduplicate_empty(self):
        assert ResultDeduplicator.deduplicate([]) == []


# ============================================================
# Retrieval Evaluator Tests
# ============================================================

class TestRetrievalEvaluator:
    def test_calculate_perfect_precision(self):
        retrieved = [{"document_id": "1"}, {"document_id": "2"}]
        relevant = ["1", "2"]
        metrics = RetrievalEvaluator.calculate_metrics(retrieved, relevant, k=2)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_calculate_mrr(self):
        retrieved = [{"document_id": "3"}, {"document_id": "1"}, {"document_id": "2"}]
        relevant = ["1"]
        metrics = RetrievalEvaluator.calculate_metrics(retrieved, relevant, k=3)
        assert metrics["mrr"] == 0.5  # 1/2 (second position)

    def test_calculate_empty(self):
        metrics = RetrievalEvaluator.calculate_metrics([], ["1"])
        assert metrics["precision"] == 0.0


# ============================================================
# Retrieval Diagnostics Tests
# ============================================================

class TestRetrievalDiagnostics:
    def test_create_diagnostics(self):
        vector_results = [{"id": 1}]
        keyword_results = [{"id": 2}, {"id": 3}]
        final_results = [{"id": 1}, {"id": 2}]
        
        diagnostics = RetrievalDiagnostics.create_diagnostics(
            vector_results, keyword_results, final_results,
            "test query", 45.5
        )
        
        assert diagnostics["vector_result_count"] == 1
        assert diagnostics["keyword_result_count"] == 2
        assert diagnostics["final_result_count"] == 2
        assert diagnostics["duration_ms"] == 45.5
        assert diagnostics["has_vector_results"] is True
