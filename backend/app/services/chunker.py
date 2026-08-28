"""Text chunking service for document processing.

Splits normalized document text into retrieval-friendly chunks while
preserving document structure (paragraphs → sentences → words → characters).

Pure function, deterministic, no external dependencies.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CHUNK_SIZE = 4000      # ~1000 tokens at ~4 chars/token
DEFAULT_CHUNK_OVERLAP = 400    # 10% of chunk_size


class ChunkingError(Exception):
    """Raised when chunking configuration is invalid."""


@dataclass(frozen=True)
class TextChunk:
    """A single chunk of text with positional metadata."""
    chunk_index: int
    text: str
    char_start: int
    char_end: int

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", "\\n")
        if len(self.text) > 60:
            preview += "..."
        return (
            f"TextChunk(index={self.chunk_index}, "
            f"chars=[{self.char_start}:{self.char_end}], "
            f"len={len(self.text)}, text={preview!r})"
        )


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[TextChunk]:
    """Split text into ordered, overlapping chunks.

    Respects document structure with boundary preference:
        1. Paragraph boundaries (double newline)
        2. Sentence boundaries (. ! ?)
        3. Word boundaries (space)
        4. Character boundaries (fallback)

    Args:
        text: Normalized text to chunk.
        chunk_size: Maximum characters per chunk. Must be > 0.
        chunk_overlap: Overlap in characters between consecutive chunks.
            Must be >= 0 and < chunk_size.

    Returns:
        Ordered list of TextChunk objects.
        Empty list if input is empty or whitespace-only.

    Raises:
        ChunkingError: If configuration is invalid.
    """
    _validate_config(chunk_size, chunk_overlap)

    if not text or not text.strip():
        return []

    text_len = len(text)

    # Text fits in a single chunk
    if text_len <= chunk_size:
        return [TextChunk(chunk_index=0, text=text, char_start=0, char_end=text_len)]

    chunks: List[TextChunk] = []
    position = 0
    chunk_index = 0

    while position < text_len:
        # Determine the end of this chunk
        remaining = text_len - position

        if remaining <= chunk_size:
            # Last chunk — take everything remaining
            chunk_text_str = text[position:]
            chunks.append(TextChunk(
                chunk_index=chunk_index,
                text=chunk_text_str,
                char_start=position,
                char_end=text_len,
            ))
            break

        # Find the best split point within the window [position, position + chunk_size)
        window_end = position + chunk_size
        split_pos = _find_split_position(text, position, window_end)

        chunk_text_str = text[position:split_pos]
        chunks.append(TextChunk(
            chunk_index=chunk_index,
            text=chunk_text_str,
            char_start=position,
            char_end=split_pos,
        ))

        # Advance position with overlap
        if chunk_overlap > 0 and split_pos < text_len:
            position = split_pos - chunk_overlap
            # Ensure we make forward progress
            if position <= chunks[-1].char_start:
                position = split_pos
        else:
            position = split_pos

        chunk_index += 1

    logger.debug(
        "chunk_text: split %d chars into %d chunks (size=%d, overlap=%d)",
        text_len, len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def _validate_config(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunking configuration parameters."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ChunkingError(
            f"chunk_size must be a positive integer, got {chunk_size!r}"
        )
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ChunkingError(
            f"chunk_overlap must be a non-negative integer, got {chunk_overlap!r}"
        )
    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )


def _find_split_position(text: str, start: int, window_end: int) -> int:
    """Find the best split position within the window, preferring natural boundaries.

    Boundary preference (highest to lowest):
        1. Paragraph boundary (\n\n)
        2. Sentence boundary (.!?\n followed by space/newline)
        3. Word boundary (space)
        4. Character boundary (window_end — fallback)
    """
    # 1. Try paragraph boundaries — scan from the end of the window backward
    best_para = -1
    for i in range(window_end - 1, start, -1):
        if text[i] == "\n":
            # Check for double newline (paragraph boundary)
            if i > start and text[i - 1] == "\n":
                best_para = i + 1  # Split after the \n\n
                break
            # Single newline at end of line — also good
            if best_para == -1:
                best_para = i + 1

    if best_para > start:
        return best_para

    # 2. Try sentence boundaries — scan backward for .!? followed by space/newline
    for i in range(window_end - 1, start, -1):
        if text[i] in ".!?" and i + 1 < window_end and text[i + 1] in " \n":
            return i + 2  # Include the punctuation and the space/newline

    # 3. Try word boundaries — scan backward for spaces
    for i in range(window_end - 1, start, -1):
        if text[i] == " ":
            return i + 1  # Split after the space

    # 4. Fallback: hard split at window end
    return window_end
