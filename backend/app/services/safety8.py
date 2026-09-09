"""Phase 19 — AI safety 8.0.

Layered prompt-injection defense covering direct/indirect/document/OCR/
metadata/filename/hidden/encoded/tool-output/connector/multi-step channels,
data-exfiltration defense, tool boundary declarations + execution limits,
output sanitization, and file security 3.0 (MIME/magic-byte validation,
traversal, decompression bombs, nested archives).

Retrieved content, OCR, metadata, filenames, tool output and connector
content are ALWAYS untrusted data — never instructions.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

INJECTION_SOURCES = ("direct", "indirect", "document", "ocr", "metadata",
                     "filename", "hidden", "encoded", "tool_output",
                     "connector", "multi_step")

MAGIC_BYTES = {
    "pdf": ("25504446", "application/pdf"),
    "png": ("89504e47", "image/png"),
    "jpg": ("ffd8ff", "image/jpeg"),
    "zip": ("504b0304", "application/zip"),
    "gzip": ("1f8b08", "application/gzip"),
    "docx": ("504b0304", "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"),
}


def _decode_variants(text: str) -> str:
    """Best-effort decode of base64/hex payloads (bounded attempts)."""
    variants = []
    compact = re.sub(r"\s+", "", text)
    try:
        variants.append(base64.b64decode(compact).decode("utf-8",
                                                         errors="ignore"))
    except Exception:  # noqa: BLE001
        pass
    try:
        variants.append(bytes.fromhex(compact[:4096]).decode(
            "utf-8", errors="ignore"))
    except (ValueError, TypeError):
        pass
    try:
        variants.append(text.encode("utf-8").decode("unicode_escape",
                                                    errors="ignore"))
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(variants)


def _detect(text: str) -> list[str]:
    from .ai_security2 import detect_injection
    hits = []
    if detect_injection(text)["injected"]:
        hits.append("instruction_markers")
    decoded = _decode_variants(text)
    if decoded and detect_injection(decoded)["injected"]:
        hits.append("encoded_instruction_payload")
    if re.search(r"(?i)\b(ignore|disregard).{0,40}instructions", text):
        hits.append("instruction_bypass")
    return hits


def injection8(text: Optional[str] = None, *,
               source: str = "direct",
               metadata: Optional[dict] = None,
               filename: Optional[str] = None,
               history: Optional[list] = None) -> dict:
    """Aggregate injection detection across all channels."""
    if source not in INJECTION_SOURCES:
        raise ValueError("unknown injection source")
    hits = []
    if text:
        hits.extend(_detect(text))
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, str):
                hits.extend(_detect(value))
    if filename:
        hits.extend(_detect(filename))
    if history:
        from .ai_security2 import escalate_guard
        escalation = escalate_guard(history)
        if escalation["escalation_attempt"]:
            hits.append("multi_step_escalation")
    if not hits:
        return {"injected": False, "source": source, "signals": [],
                "level": "clean"}
    return {"injected": True, "source": source, "signals": hits[:8],
            "level": "high" if len(hits) >= 2 else "suspicious"}


# ---------------------------------------------------------------------------
# Exfiltration defense 2.0
# ---------------------------------------------------------------------------

SECRET_HARVEST_PATTERNS = (
    re.compile(r"(?i)(private key|pem|id_rsa|pgp|ssh|credential|client "
               r"secret|connection string)"),
    re.compile(r"(?i)api[ -_]?key[s]?\s*(=|:)\s*[\w-]{8,}"),
    re.compile(r"(?i)(all|every|any) (customers?|users?|passwords?|"
               r"documents?) (and|or) (their|its) (data|content)"),
)


def exfiltration2(query: str, *, scope: Optional[dict] = None) -> dict:
    from .ai_security2 import detect_exfiltration
    base = detect_exfiltration(query, scope=scope)
    hits = list(base["signals"])
    hits.extend(p.pattern for p in SECRET_HARVEST_PATTERNS
                if p.search(query or ""))
    suspicious = bool(hits)
    return {"suspicious": suspicious, "signals": hits[:8],
            "scope": scope or {}}


# ---------------------------------------------------------------------------
# Tool boundaries + execution limits
# ---------------------------------------------------------------------------

def tool_boundary(tool: str) -> dict:
    """Every tool declares permissions, scope, side-effect level, allowed
    input, timeout and output limits before use."""
    declarations = {
        "search_documents": {"permissions": ["read:workspace"],
                             "scope": "workspace", "side_effect_level":
                             "none", "allowed_input": "query string",
                             "timeout_s": 10, "output_limit": 8000,
                             "declared": True},
        "summarize": {"permissions": ["read:workspace"],
                      "scope": "workspace",
                      "side_effect_level": "none",
                      "allowed_input": "text/document ref",
                      "timeout_s": 60, "output_limit": 4000,
                      "declared": True},
        "update_document": {"permissions": ["write:workspace"],
                            "scope": "workspace",
                            "side_effect_level": "mutating",
                            "allowed_input": "document_id + patch",
                            "timeout_s": 30, "output_limit": 1000,
                            "declared": True},
        "send_notification": {"permissions": ["notify:workspace"],
                              "scope": "workspace",
                              "side_effect_level": "external",
                              "allowed_input": "recipient + template",
                              "timeout_s": 15, "output_limit": 1000,
                              "declared": True},
        "delete_document": {"permissions": ["delete:workspace"],
                            "scope": "workspace",
                            "side_effect_level": "destructive",
                            "allowed_input": "document_id",
                            "timeout_s": 15, "output_limit": 1000,
                            "declared": True},
        "http_request": {"permissions": ["network:restricted"],
                         "scope": "workspace",
                         "side_effect_level": "external",
                         "allowed_input": "allowlisted URL + method",
                         "timeout_s": 10, "output_limit": 20000,
                         "declared": True},
    }
    return declarations.get(tool, {"permissions": [], "scope": "none",
                                   "side_effect_level": "unknown",
                                   "allowed_input": "unknown",
                                   "timeout_s": 5, "output_limit": 1000,
                                   "declared": False})


def enforce_tool_limits(*, calls: int, duration_s: float,
                        output_chars: int, recursion_depth: int = 0,
                        limits: Optional[dict] = None) -> dict:
    limits = limits or {"max_calls": 100, "max_duration_s": 3600,
                        "max_output": 20000, "max_recursion": 3,
                        "max_concurrency": 8}
    violations = []
    if calls > limits["max_calls"]:
        violations.append("tool-call budget exceeded")
    if duration_s > limits["max_duration_s"]:
        violations.append("tool duration budget exceeded")
    if output_chars > limits["max_output"]:
        violations.append("tool output size exceeded")
    if recursion_depth > limits["max_recursion"]:
        violations.append("recursion depth exceeded")
    if violations:
        return {"allowed": False, "violations": violations}
    return {"allowed": True, "violations": []}


# ---------------------------------------------------------------------------
# Output sanitization
# ---------------------------------------------------------------------------

def sanitize_output(text: Optional[str]) -> dict:
    """Prevent unsafe HTML/script injection, dangerous links and credential
    leakage in AI output."""
    if text is None:
        return {"safe": True, "output": "", "stripped": 0}
    original = text
    stripped = 0
    out = re.sub(r"<script.*?</script>", "", text,
                 flags=re.IGNORECASE | re.DOTALL)
    stripped += len(text) - len(out)
    out = re.sub(r"<iframe.*?</iframe>", "", out,
                 flags=re.IGNORECASE | re.DOTALL)
    stripped += len(text) - len(out) - 0 if False else 0
    before = len(out)
    out = re.sub(r"(javascript|data|vbscript):", "blocked:",
                 out, flags=re.IGNORECASE)
    stripped += before - len(out)
    from .observability2 import redact
    out = redact(out)
    return {"safe": True, "output": out, "stripped": stripped}


# ---------------------------------------------------------------------------
# File security 3.0
# ---------------------------------------------------------------------------

def file_security3(*, filename: Optional[str] = None,
                   mime_type: Optional[str] = None,
                   size_bytes: Optional[int] = None,
                   magic_hex: Optional[str] = None,
                   limits: Optional[dict] = None) -> dict:
    """Traversal, MIME mismatch (magic bytes), size, and filename checks."""
    limits = limits or {"max_file_mb": 100, "max_pages": 1000}
    issues = []
    if filename:
        lower = filename.lower()
        if ".." in filename or filename.startswith(("/", "\\")) \
                or re.match(r"^[a-z]:[\\\\/]", filename, re.IGNORECASE):
            issues.append("path traversal detected")
        if "\x00" in filename:
            issues.append("null byte in filename")
    if size_bytes is not None and \
            size_bytes > limits["max_file_mb"] * 1024 * 1024:
        issues.append("file exceeds size limit")
    if mime_type and magic_hex:
        detected = _detect_magic(magic_hex)
        if detected and detected != mime_type:
            issues.append("MIME mismatch: declared "
                          f"{mime_type} but magic bytes indicate {detected}")
    if issues:
        return {"safe": False, "issues": issues}
    return {"safe": True, "issues": []}


def _detect_magic(magic_hex: str) -> Optional[str]:
    magic = (magic_hex or "").lower()
    for fmt, (hex_prefix, mime) in MAGIC_BYTES.items():
        if magic.startswith(hex_prefix):
            return mime
    return None


def decompression_bomb(*, compressed_bytes: int,
                       estimated_ratio: Optional[float] = None,
                       max_ratio: float = 100.0,
                       decompressed_bytes: Optional[int] = None) -> dict:
    """Reject archive bombs: ratio guard + absolute-size guard."""
    if estimated_ratio is not None and estimated_ratio > max_ratio:
        return {"safe": False,
                "reason": f"decompression ratio {estimated_ratio:.1f}x "
                          f"exceeds limit {max_ratio:.0f}x"}
    if decompressed_bytes is not None and \
            decompressed_bytes > 1024 * 1024 * 1024:  # 1 GiB hard cap
        return {"safe": False,
                "reason": "decompressed size exceeds 1 GiB hard cap"}
    return {"safe": True, "reason": "within decompression limits"}


def nested_archive(depth: int, max_depth: int = 3) -> dict:
    if depth > max_depth:
        return {"safe": False,
                "reason": f"archive nesting depth {depth} exceeds "
                          f"{max_depth}"}
    return {"safe": True, "reason": "nesting depth acceptable",
            "depth": depth}
