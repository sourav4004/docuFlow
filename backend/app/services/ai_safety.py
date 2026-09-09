"""AI Safety - Prompt injection defense and output sanitization."""

import re
from typing import Any, Dict, List, Optional, Tuple


class PromptInjectionDefender:
    """Defends against prompt injection attacks."""
    
    # Known injection patterns
    INJECTION_PATTERNS = [
        r"ignore (previous|all|above) (instructions?|prompts?|rules?)",
        r"disregard (previous|all|above) (instructions?|prompts?|rules?)",
        r"forget (previous|all|above) (instructions?|prompts?|rules?)",
        r"you are now",
        r"act as if",
        r"pretend (you|to be)",
        r"new (instructions?|role|persona)",
        r"system (prompt|message|instruction)",
        r"ADMIN:|USER:|SYSTEM:",
        r"\[INST\]|\[/INST\]",
        r"<\|im_start\|>|<\|im_end\|>",
        r"```(system|admin)",
    ]
    
    # Suspicious instruction patterns
    SUSPICIOUS_PATTERNS = [
        r"reveal (the|your) (secret|key|password|token|api)",
        r"show (me )?(the|your) (secret|key|password|token|api)",
        r"what (is|are) (the|your) (secret|key|password|token|api)",
        r"execute (this|the) (code|command|script)",
        r"run (this|the) (code|command|script)",
        r"delete (all|every|the) (files?|data|records?)",
        r"drop (table|database)",
    ]
    
    @classmethod
    def detect_injection(cls, text: str) -> Tuple[bool, List[str]]:
        """Detect potential prompt injection attempts."""
        findings = []
        text_lower = text.lower()
        
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                findings.append(f"Potential injection: {pattern}")
        
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                findings.append(f"Suspicious instruction: {pattern}")
        
        return len(findings) > 0, findings
    
    @classmethod
    def sanitize_for_context(cls, text: str) -> str:
        """Sanitize text before including in AI context."""
        # Remove potential injection markers
        sanitized = text
        
        # Remove HTML/script tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
        
        # Remove potential instruction markers
        sanitized = re.sub(r'\[INST\]|\[/INST\]', '', sanitized)
        sanitized = re.sub(r'<\|im_start\|>|<\|im_end\|>', '', sanitized)
        
        # Remove system/user/admin markers
        sanitized = re.sub(r'^(SYSTEM|ADMIN|USER):', '', sanitized, flags=re.MULTILINE)
        
        # Escape special characters that might be interpreted
        sanitized = sanitized.replace('\\', '\\\\')
        
        return sanitized
    
    @classmethod
    def create_safe_context(
        cls,
        system_instructions: str,
        user_query: str,
        retrieved_data: List[str]
    ) -> Dict[str, str]:
        """Create a safe context with clear separation."""
        # Sanitize retrieved data
        safe_retrieved = []
        for data in retrieved_data:
            safe_data = cls.sanitize_for_context(data)
            safe_retrieved.append(safe_data)
        
        return {
            "system": system_instructions,
            "user": user_query,
            "data": "\n\n".join(safe_retrieved),
            "context_separator": "---END_OF_DATA---"
        }


class OutputSanitizer:
    """Sanitizes AI-generated output."""
    
    @classmethod
    def sanitize_output(cls, text: str) -> str:
        """Sanitize AI output for safe rendering."""
        if not text:
            return ""
        
        # Remove potential XSS
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        # Remove potential executable content
        sanitized = re.sub(r'data:text/html', '', sanitized, flags=re.IGNORECASE)
        
        # Remove potential malicious URLs
        sanitized = re.sub(r'href\s*=\s*["\']?javascript:', 'href="#"', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @classmethod
    def sanitize_markdown(cls, text: str) -> str:
        """Sanitize markdown content."""
        if not text:
            return ""
        
        # Remove potentially dangerous HTML in markdown
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<iframe[^>]*>.*?</iframe>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<object[^>]*>.*?</object>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<embed[^>]*>.*?</embed>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    @classmethod
    def extract_safe_citations(cls, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract safe citation information."""
        safe_citations = []
        
        for citation in citations:
            safe_citation = {
                "document_id": citation.get("document_id"),
                "document_name": citation.get("document_name", "Unknown"),
                "page": citation.get("page"),
                "section": citation.get("section"),
                "chunk_id": citation.get("chunk_id"),
                "relevance_score": citation.get("relevance_score", 0.0),
                "snippet": cls.sanitize_output(citation.get("snippet", "")[:200])
            }
            safe_citations.append(safe_citation)
        
        return safe_citations


class DataExfiltrationDefender:
    """Prevents AI from leaking unauthorized data."""
    
    @classmethod
    def validate_workspace_access(
        cls,
        requested_workspace_id: int,
        user_workspace_ids: List[int]
    ) -> bool:
        """Validate that the requested workspace is accessible."""
        return requested_workspace_id in user_workspace_ids
    
    @classmethod
    def validate_document_access(
        cls,
        document_workspace_id: int,
        user_workspace_ids: List[int]
    ) -> bool:
        """Validate that a document's workspace is accessible."""
        return document_workspace_id in user_workspace_ids
    
    @classmethod
    def filter_unauthorized_results(
        cls,
        results: List[Dict[str, Any]],
        user_workspace_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Filter out results from unauthorized workspaces."""
        authorized = []
        
        for result in results:
            result_workspace_id = result.get("workspace_id")
            if result_workspace_id in user_workspace_ids:
                authorized.append(result)
        
        return authorized
    
    @classmethod
    def validate_tool_parameters(
        cls,
        tool_name: str,
        parameters: Dict[str, Any],
        user_workspace_id: int
    ) -> Tuple[bool, str]:
        """Validate tool parameters for security."""
        # Check for workspace ID in parameters
        if "workspace_id" in parameters:
            if parameters["workspace_id"] != user_workspace_id:
                return False, "Cannot access resources from another workspace"
        
        # Check for document ID with workspace validation
        if "document_id" in parameters:
            # Would need to validate document belongs to workspace
            pass
        
        # Check for potentially dangerous parameters
        dangerous_params = ["sql", "query", "command", "script", "exec"]
        for param in dangerous_params:
            if param in parameters:
                return False, f"Potentially dangerous parameter: {param}"
        
        return True, ""


class AISafetyService:
    """Comprehensive AI safety service."""
    
    def __init__(self):
        self.injection_defender = PromptInjectionDefender()
        self.output_sanitizer = OutputSanitizer()
        self.exfiltration_defender = DataExfiltrationDefender()
    
    def validate_input(self, text: str) -> Tuple[bool, List[str]]:
        """Validate user input for safety."""
        return self.injection_defender.detect_injection(text)
    
    def sanitize_for_context(self, text: str) -> str:
        """Sanitize text for AI context."""
        return self.injection_defender.sanitize_for_context(text)
    
    def sanitize_output(self, text: str) -> str:
        """Sanitize AI output."""
        return self.output_sanitizer.sanitize_output(text)
    
    def validate_workspace_access(
        self,
        requested_workspace_id: int,
        user_workspace_ids: List[int]
    ) -> bool:
        """Validate workspace access."""
        return self.exfiltration_defender.validate_workspace_access(
            requested_workspace_id, user_workspace_ids
        )
    
    def validate_tool_parameters(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_workspace_id: int
    ) -> Tuple[bool, str]:
        """Validate tool parameters."""
        return self.exfiltration_defender.validate_tool_parameters(
            tool_name, parameters, user_workspace_id
        )
