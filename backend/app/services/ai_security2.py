"""Phase 18 AI security — prompt injection 7.0, exfiltration defense,
tool isolation, SSRF defense.

Retrieved content, OCR text, metadata, and tool output are ALWAYS treated as
untrusted data. Defense is layered: separation delimiters + instruction
detection + output sanitization, with explicit per-layer results so failures
are observable, never silent.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

INSTRUCTION_PATTERNS = (
    re.compile(r"ignore (all )?(previous|prior|above|earlier) instructions",
               re.IGNORECASE),
    re.compile(r"(disregard|forget|override) (your|the) (system|developer)"
               r" (instructions|prompt|policy)", re.IGNORECASE),
    re.compile(r"you are now (the |an? )?(new |different )?(system|assistant)",
               re.IGNORECASE),
    re.compile(r"repeat (your|the) (system|developer|instructions) (prompt|"
               r"instructions)", re.IGNORECASE),
    re.compile(r"reveal (your|the) (system )?(prompt|instructions|secrets)",
               re.IGNORECASE),
    re.compile(r"act as (an? )?unfiltered|no restrictions|no (safety )?rules",
               re.IGNORECASE),
    re.compile(r"<system|system_prompt|developer_message", re.IGNORECASE),
    re.compile(r"## (system|instructions|developer)", re.IGNORECASE),
    re.compile(r"you are now (an? )?(unfiltered|unrestricted|dan)\b",
               re.IGNORECASE),
    re.compile(r"jailbreak|unrestricted mode|without (any )?restrictions|"
               r"no rules apply", re.IGNORECASE),
    # exfil-style instructions (list/print/show credentials, env vars)
    re.compile(r"(print|show|display|reveal|list|dump|export) (the |all |any )?"
               r"(content of )?(environment variables|system (prompt|config|"
               r"settings)|credentials|api[ _-]?keys?|passwords|secrets|"
               r"private keys)", re.IGNORECASE),
    # connector/email exfil + workflow instructions + hidden content
    re.compile(r"(workflow instruction|when approved, call|reply to this email "
               r"with|respond to this email with)", re.IGNORECASE),
    re.compile(r"send (all |the |any |every )?(prompts|data|documents|files|"
               r"emails|records|conversations).{0,20}(to|via) ",
               re.IGNORECASE),
    re.compile(r"white[ -]?on[ -]?white|invisible text|hidden (content|text)",
               re.IGNORECASE),
)

ENCODED_VARIANTS = (
    re.compile(r"i\s*g\s*n\s*o\s*r\s*e", re.IGNORECASE),
    re.compile(r"d\s*i\s*s\s*r\s*e\s*g\s*a\s*r\s*d", re.IGNORECASE),
    re.compile(r"s\s*y\s*s\s*t\s*e\s*m", re.IGNORECASE),
    # base64("ignore...") and hex("ignore") prefixes
    re.compile(r"aWdub3Jl", re.IGNORECASE),
    re.compile(r"69676e6f7265", re.IGNORECASE),
)


def detect_injection(text: str) -> dict:
    """Layered instruction detection over untrusted content."""
    if not text:
        return {"injected": False, "patterns": [], "level": "clean"}
    patterns_hit = []
    for pattern in INSTRUCTION_PATTERNS:
        if pattern.search(text):
            patterns_hit.append(pattern.pattern[:60])
    for pattern in ENCODED_VARIANTS:
        if pattern.search(text):
            patterns_hit.append("encoded:" + pattern.pattern[:40])
    if not patterns_hit:
        return {"injected": False, "patterns": [], "level": "clean"}
    level = "high" if len(patterns_hit) >= 2 else "suspicious"
    return {"injected": True, "patterns": patterns_hit[:5], "level": level}


def sanitize_tool_output(output: str, max_length: int = 8000) -> dict:
    """Treat tool output as untrusted data — strip instruction carriers and
    cap size. Never lets tool output escalate instructions."""
    if output is None:
        return {"safe": True, "output": "", "truncated": False,
                "stripped": 0}
    text = str(output)
    stripped = 0
    original_len = len(text)
    # Remove common markup that carries instructions.
    text = re.sub(r"<system[^>]*>.*?</system>", "", text,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"```(system|developer|instructions).*?```", "", text,
                  flags=re.IGNORECASE | re.DOTALL)
    stripped += original_len - len(text)
    truncated = False
    if len(text) > max_length:
        text = text[:max_length]
        truncated = True
    return {"safe": True, "output": text, "truncated": truncated,
            "stripped": stripped}


# ---------------------------------------------------------------------------
# Exfiltration defense
# ---------------------------------------------------------------------------

EXFILTRATION_SIGNALS = (
    re.compile(r"select .* from .*(users|api_keys|sessions|passwords)",
               re.IGNORECASE),
    re.compile(r"(reveal|show|dump|give) (all |any |every )?(users|emails|"
               r"passwords|api[ _-]?keys|tokens|secrets|credentials|private "
               r"keys)", re.IGNORECASE),
    re.compile(r"(export|download|copy) (all|every|another) (documents|files|"
               r"records|data) (in|from) (the |this )?(organization|tenant|"
               r"company|workspace)", re.IGNORECASE),
    re.compile(r"(other|another) (tenant|organization|workspace)'?s? (data|"
               r"records|documents|customers)", re.IGNORECASE),
    re.compile(r"(system )?(prompt|instructions|developer) (text|content|"
               r"string|config)", re.IGNORECASE),
    re.compile(r"(view|read) another (tenant|organization|workspace)",
               re.IGNORECASE),
    re.compile(r"(another|other|some (other )?|different) (tenant|organization|"
               r"workspace|company)'?s? .{0,30}(records|documents|customers|"
               r"data|files)", re.IGNORECASE),
    # list/send/forward-style credential and PII exfiltration
    re.compile(r"(list|print|read|dump|send|forward|copy|email|exfiltrate) "
               r"(all |any |the |every )?(api[ _-]?keys?|credentials|"
               r"passwords|secrets|tokens|private keys|pii|customer "
               r"(data|records|pii))", re.IGNORECASE),
)


def detect_exfiltration(query: str, *, scope: Optional[dict] = None) -> dict:
    """Detect suspicious requests attempting unauthorized data extraction."""
    if not query:
        return {"suspicious": False, "signals": []}
    hits = [p.pattern[:60] for p in EXFILTRATION_SIGNALS if p.search(query)]
    suspicious = bool(hits)
    if scope is not None:
        # Cross-tenant scope requests are always suspicious.
        if scope.get("cross_tenant"):
            hits.append("cross-tenant scope requested")
            suspicious = True
    return {"suspicious": suspicious, "signals": hits[:5]}


def tool_isolation_allowed(tool: str, risk: str) -> tuple[bool, str]:
    """High-risk tools require isolation + approval — never bare execution."""
    high_risk_tools = {"shell", "subprocess", "http_request", "file_write",
                       "db_write", "delete", "email_send"}
    if tool in high_risk_tools:
        if risk in ("HIGH", "CRITICAL"):
            return False, ("high-risk tool requires isolated sandbox + "
                           "explicit approval")
        return False, f"tool {tool!r} requires sandbox isolation"
    return True, "safe"


# ---------------------------------------------------------------------------
# SSRF defense helpers
# ---------------------------------------------------------------------------

def _is_private(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast
                or addr.is_unspecified)
    except ValueError:
        return False


def validate_redirect_target(url: str) -> dict:
    """Validate a URL BEFORE and AFTER redirects (block private targets)."""
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        return {"allowed": False, "reason": f"invalid URL: {exc}"}
    if parsed.scheme not in ("http", "https"):
        return {"allowed": False,
                "reason": f"unsupported scheme {parsed.scheme!r}"}
    host = parsed.hostname or ""
    if not host:
        return {"allowed": False, "reason": "no host"}
    if host in ("localhost", "metadata.google.internal",
                "169.254.169.254"):
        return {"allowed": False, "reason": f"blocked host {host!r}"}
    if _is_private(host):
        return {"allowed": False, "reason": f"blocked private host {host!r}"}
    try:
        import socket
        for info in socket.getaddrinfo(host, None):
            if _is_private(info[4][0]):
                return {"allowed": False,
                        "reason": f"DNS resolves to private IP {info[4][0]}"}
    except OSError:
        return {"allowed": False, "reason": "DNS resolution failed"}
    return {"allowed": True, "host": host, "scheme": parsed.scheme}


def ssrf_check(url: str) -> dict:
    """One-call SSRF defense: scheme, host, DNS, and private-IP block."""
    return validate_redirect_target(url)


# ---------------------------------------------------------------------------
# Multi-step escalation defense
# ---------------------------------------------------------------------------

def escalate_guard(history: list[dict]) -> dict:
    """Detect multi-step instruction-escalation chains in tool history.

    Each entry: {"tool": ..., "output_preview": ...}. If an earlier tool
    output contained instruction markers and a later step requested system
    data, block the escalation.
    """
    poisoned = False
    for entry in history or []:
        preview = str(entry.get("output_preview", ""))
        if detect_injection(preview)["injected"]:
            poisoned = True
            break
    later = [e for e in (history or [])[1:]
             if detect_exfiltration(str(e.get("query", "")))["suspicious"]]
    return {
        "escalation_attempt": bool(poisoned and later),
        "poisoned_context": poisoned,
        "suspicious_steps": len(later),
    }