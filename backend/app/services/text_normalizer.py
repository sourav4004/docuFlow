"""Text normalization service for document processing.

Converts raw PDF extraction output into clean, deterministic text
suitable for chunking and embedding. Preserves meaningful content,
paragraphs, punctuation, headings, and non-English characters.

No summarization, translation, rewriting, or LLM involvement.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Compiled regex patterns for performance
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r"[^\S\n]+")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_TABS = re.compile(r"\t")


def normalize_text(text: str) -> str:
    """Normalize raw extracted text into clean, deterministic output.

    Handles:
        - CRLF / CR → LF normalization
        - Tab → spaces
        - Trailing whitespace on each line
        - Multiple horizontal spaces → single space
        - Three or more blank lines → two blank lines
        - Leading / trailing whitespace on the full document
        - Unicode NFC normalization for consistent representation

    Preserves:
        - Paragraphs (double newlines)
        - Meaningful single newlines
        - Punctuation and document wording
        - Non-English characters (Unicode)

    Args:
        text: Raw text extracted from a PDF.

    Returns:
        Normalized text string. Empty string if input is empty or whitespace-only.
    """
    if not text:
        return ""

    # 1. Unicode NFC normalization — consistent representation
    text = unicodedata.normalize("NFC", text)

    # 2. Normalize line endings: CRLF / CR → LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Convert tabs to spaces (standard 4-space tab)
    text = _TABS.sub("    ", text)

    # 4. Strip trailing whitespace from each line
    text = _TRAILING_WHITESPACE.sub("", text)

    # 5. Collapse multiple horizontal spaces (but not newlines) into one
    text = _MULTI_SPACES.sub(" ", text)

    # 6. Collapse three or more blank lines into exactly two newlines (one blank line)
    text = _MULTI_NEWLINES.sub("\n\n", text)

    # 7. Strip leading and trailing whitespace from the entire document
    text = text.strip()

    return text
