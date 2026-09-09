"""Document intelligence service for classification, summarization, and metadata extraction."""

import re
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Simple rule-based document classifier."""
    
    DOCUMENT_TYPES = {
        "invoice": ["invoice", "bill", "payment", "amount due", "total"],
        "contract": ["agreement", "party", "terms", "conditions", "signature"],
        "report": ["report", "analysis", "findings", "conclusion", "summary"],
        "resume": ["resume", "experience", "skills", "education", "objective"],
        "policy": ["policy", "procedure", "guidelines", "compliance", "regulation"],
        "meeting_notes": ["meeting", "attendees", "agenda", "action items", "minutes"],
        "technical": ["specification", "api", "technical", "architecture", "implementation"],
        "academic": ["abstract", "introduction", "methodology", "references", "citation"],
    }
    
    @classmethod
    def classify(cls, content: str, filename: str = "") -> Dict[str, Any]:
        """Classify document based on content and filename."""
        content_lower = content.lower() if content else ""
        filename_lower = filename.lower()
        
        scores = {}
        for doc_type, keywords in cls.DOCUMENT_TYPES.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            # Bonus for filename match
            if doc_type in filename_lower:
                score += 2
            if score > 0:
                scores[doc_type] = score
        
        if not scores:
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "classifier_version": "1.0"
            }
        
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        total_keywords = len(cls.DOCUMENT_TYPES[best_type])
        confidence = min(max_score / total_keywords, 1.0)
        
        return {
            "document_type": best_type,
            "confidence": round(confidence, 2),
            "classifier_version": "1.0"
        }


class MetadataExtractor:
    """Extract useful metadata from documents."""
    
    @classmethod
    def extract_metadata(cls, content: str, filename: str = "") -> Dict[str, Any]:
        """Extract metadata from document content."""
        if not content:
            return {}
        
        metadata = {
            "word_count": len(content.split()),
            "character_count": len(content),
            "filename": filename,
            "language": cls._detect_language(content),
        }
        
        # Extract potential title (first line or filename)
        lines = content.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 200:  # Reasonable title length
                metadata["title"] = first_line
        
        # Extract dates
        dates = cls._extract_dates(content)
        if dates:
            metadata["dates_found"] = dates
        
        return metadata
    
    @classmethod
    def _detect_language(cls, content: str) -> str:
        """Simple language detection based on common words."""
        common_english = ["the", "and", "is", "in", "to", "of", "for", "that", "with", "this"]
        words = content.lower().split()[:100]  # Sample first 100 words
        
        english_count = sum(1 for w in words if w in common_english)
        if english_count > 10:
            return "en"
        return "unknown"
    
    @classmethod
    def _extract_dates(cls, content: str) -> List[str]:
        """Extract date patterns from content."""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',      # YYYY-MM-DD
            r'\d{1,2}-\d{1,2}-\d{4}',      # DD-MM-YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            dates.extend(matches[:5])  # Limit to 5 dates
        
        return dates


class DocumentSummarizer:
    """Generate document summaries."""
    
    @classmethod
    def extract_key_points(cls, content: str, max_points: int = 5) -> List[str]:
        """Extract key points from document content."""
        if not content:
            return []
        
        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Score sentences by length and position
        scored = []
        for i, sentence in enumerate(sentences[:50]):  # Limit to first 50 sentences
            score = len(sentence) / 100  # Longer sentences score higher
            if i < 5:  # Boost early sentences
                score *= 1.5
            scored.append((score, sentence))
        
        # Return top sentences
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s[1] for s in scored[:max_points]]
    
    @classmethod
    def generate_short_summary(cls, content: str) -> str:
        """Generate a short summary (first 2-3 sentences)."""
        if not content:
            return ""
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return content[:200] + "..." if len(content) > 200 else content
        
        # Return first 2-3 meaningful sentences
        summary_sentences = []
        for s in sentences[:5]:
            if len(s) > 20:
                summary_sentences.append(s)
                if len(summary_sentences) >= 3:
                    break
        
        return ". ".join(summary_sentences) + "." if summary_sentences else content[:200]


class EntityExtractor:
    """Extract entities from document content."""
    
    ENTITY_PATTERNS = {
        "person": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        "organization": r'\b[A-Z][a-z]+ (?:Corp|Inc|LLC|Ltd|Company|Organization)\b',
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    }
    
    @classmethod
    def extract_entities(cls, content: str, max_per_type: int = 5) -> Dict[str, List[str]]:
        """Extract entities from content."""
        if not content:
            return {}
        
        entities = {}
        for entity_type, pattern in cls.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, content)
            # Deduplicate and limit
            unique_matches = list(set(matches))[:max_per_type]
            if unique_matches:
                entities[entity_type] = unique_matches
        
        return entities
