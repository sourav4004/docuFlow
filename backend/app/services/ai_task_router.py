"""AI Task Router - Intelligent task classification and routing."""

import re
from typing import Dict, List, Tuple
from .ai_orchestration import AITask, AITaskType, AITaskPriority


class AITaskRouter:
    """Routes AI tasks based on query analysis and context."""
    
    # Pattern mappings for task classification
    TASK_PATTERNS = {
        AITaskType.DOCUMENT_QA: [
            r"what (is|are|does|do|was|were)",
            r"how (does|do|is|are|was|were)",
            r"why (is|are|does|do|was|were)",
            r"when (is|are|does|do|was|were)",
            r"where (is|are|does|do|was|were)",
            r"explain",
            r"describe",
            r"tell me about",
        ],
        AITaskType.SUMMARIZATION: [
            r"summar(?:ize|ise|y|ies)",
            r"brief (overview|summary)",
            r"key points",
            r"main (ideas|points|concepts)",
            r"condensed",
            r"tl;?dr",
        ],
        AITaskType.EXTRACTION: [
            r"extract",
            r"pull out",
            r"identify",
            r"find (all|every)",
            r"list (all|every)",
            r"get (all|every)",
        ],
        AITaskType.COMPARISON: [
            r"compar?e",
            r"difference(?:s)?",
            r"versus|vs\.?",
            r"contrast",
            r"how (are|is) .+ (different|similar|alike)",
        ],
        AITaskType.CLASSIFICATION: [
            r"classif(?:y|ication)",
            r"categoriz?e",
            r"what (type|kind|category)",
            r"what (document|file) (type|kind)",
        ],
        AITaskType.CONFLICT_DETECTION: [
            r"conflict(?:s|ing)?",
            r"contradict(?:s|ion|ory)?",
            r"inconsisten(?:t|cy|cies)",
            r"disagree(?:s|ment)?",
            r"differ(?:s|ence)? .+ (policy|requirement|date)",
        ],
        AITaskType.MULTI_DOCUMENT_ANALYSIS: [
            r"across (these|all|the) documents",
            r"between (these|all|the) documents",
            r"compare (these|all|the) documents",
            r"in .+ (collection|set|group)",
        ],
        AITaskType.ENTITY_EXTRACTION: [
            r"who (is|are|was|were)",
            r"(people|persons|companies|organizations)",
            r"entity|entities",
            r"named (entities|entities)",
        ],
    }
    
    # Priority patterns
    PRIORITY_PATTERNS = {
        AITaskPriority.FAST: [
            r"^is\b",
            r"^does\b",
            r"^can\b",
            r"^yes or no",
        ],
        AITaskPriority.POWERFUL: [
            r"analyz?e",
            r"reason",
            r"explain why",
            r"what if",
            r"hypothetical",
        ],
    }
    
    @classmethod
    def classify_task(cls, task: AITask) -> AITask:
        """Classify and enrich an AI task with type and priority."""
        query_lower = task.query.lower()
        
        # Classify task type
        task_type = cls._classify_type(query_lower)
        task.task_type = task_type
        
        # Classify priority
        priority = cls._classify_priority(query_lower, task)
        task.priority = priority
        
        # Enrich context based on task type
        task = cls._enrich_context(task)
        
        return task
    
    @classmethod
    def _classify_type(cls, query: str) -> AITaskType:
        """Classify the task type based on query patterns."""
        scores: Dict[AITaskType, int] = {}
        
        for task_type, patterns in cls.TASK_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[task_type] = score
        
        if not scores:
            return AITaskType.DOCUMENT_QA  # Default
        
        return max(scores, key=scores.get)
    
    @classmethod
    def _classify_priority(cls, query: str, task: AITask) -> AITaskPriority:
        """Classify task priority."""
        for priority, patterns in cls.PRIORITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return priority
        
        # Default priority based on task type
        type_priority_map = {
            AITaskType.DOCUMENT_QA: AITaskPriority.BALANCED,
            AITaskType.SUMMARIZATION: AITaskPriority.BALANCED,
            AITaskType.EXTRACTION: AITaskPriority.BALANCED,
            AITaskType.COMPARISON: AITaskPriority.POWERFUL,
            AITaskType.MULTI_DOCUMENT_ANALYSIS: AITaskPriority.POWERFUL,
            AITaskType.CONFLICT_DETECTION: AITaskPriority.POWERFUL,
        }
        
        return type_priority_map.get(task.task_type, AITaskPriority.BALANCED)
    
    @classmethod
    def _enrich_context(cls, task: AITask) -> AITask:
        """Enrich task context based on classification."""
        if task.task_type == AITaskType.MULTI_DOCUMENT_ANALYSIS:
            # Ensure multiple documents are referenced
            if len(task.document_ids) < 2:
                task.context["needs_document_selection"] = True
        
        elif task.task_type == AITaskType.COMPARISON:
            # Comparison requires at least 2 documents
            if len(task.document_ids) < 2:
                task.context["needs_document_selection"] = True
        
        elif task.task_type == AITaskType.CONFLICT_DETECTION:
            # Conflict detection requires multiple documents
            if len(task.document_ids) < 2:
                task.context["needs_document_selection"] = True
        
        return task
    
    @classmethod
    def get_supported_tasks(cls) -> List[Dict[str, str]]:
        """Get list of supported task types with descriptions."""
        return [
            {
                "type": AITaskType.DOCUMENT_QA.value,
                "description": "Answer questions about a specific document"
            },
            {
                "type": AITaskType.MULTI_DOCUMENT_ANALYSIS.value,
                "description": "Analyze and compare multiple documents"
            },
            {
                "type": AITaskType.SUMMARIZATION.value,
                "description": "Generate summaries of documents"
            },
            {
                "type": AITaskType.EXTRACTION.value,
                "description": "Extract structured information from documents"
            },
            {
                "type": AITaskType.COMPARISON.value,
                "description": "Compare documents and identify differences"
            },
            {
                "type": AITaskType.CLASSIFICATION.value,
                "description": "Classify documents by type or category"
            },
            {
                "type": AITaskType.CONFLICT_DETECTION.value,
                "description": "Detect conflicts between documents"
            },
            {
                "type": AITaskType.ENTITY_EXTRACTION.value,
                "description": "Extract named entities from documents"
            },
        ]
