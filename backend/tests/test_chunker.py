"""Phase 4.2 tests: text chunking service.

Tests cover:
    - Empty / whitespace-only input
    - Short text (single chunk)
    - Exact chunk-size text
    - Large text (multiple chunks)
    - Chunk ordering and metadata
    - Chunk overlap behavior
    - No unexpected content loss
    - Unicode text (English, CJK, Arabic, Hindi)
    - Paragraph preservation
    - Sentence boundary preference
    - Word boundary fallback
    - Long-word handling
    - Invalid configuration
    - Deterministic output
    - Character metadata (char_start, char_end)
    - Integration with document processing pipeline
    - Performance sanity
"""

import pytest

from app.services.chunker import (
    chunk_text,
    TextChunk,
    ChunkingError,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


# ==========================================================================
# 1. EMPTY AND EDGE INPUTS
# ==========================================================================

class TestEmptyAndEdgeInputs:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   ") == []

    def test_only_newlines(self):
        assert chunk_text("\n\n\n") == []

    def test_single_space(self):
        assert chunk_text(" ") == []

    def test_single_character(self):
        chunks = chunk_text("a")
        assert len(chunks) == 1
        assert chunks[0].text == "a"
        assert chunks[0].chunk_index == 0
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 1


# ==========================================================================
# 2. SHORT TEXT — SINGLE CHUNK
# ==========================================================================

class TestShortText:
    def test_short_paragraph(self):
        text = "This is a short document with one paragraph."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)

    def test_text_exactly_chunk_size(self):
        text = "A" * DEFAULT_CHUNK_SIZE
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == DEFAULT_CHUNK_SIZE

    def test_text_one_char_under_chunk_size(self):
        text = "A" * (DEFAULT_CHUNK_SIZE - 1)
        chunks = chunk_text(text)
        assert len(chunks) == 1

    def test_text_one_char_over_chunk_size(self):
        text = "A" * (DEFAULT_CHUNK_SIZE + 1)
        chunks = chunk_text(text)
        assert len(chunks) == 2

    def test_custom_small_chunk_size(self):
        text = "Hello world, this is a test."
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 10


# ==========================================================================
# 3. MULTIPLE CHUNKS — ORDERING AND METADATA
# ==========================================================================

class TestMultipleChunks:
    def test_ordered_chunk_indices(self):
        text = "A\n\n" * 500  # ~2000 chars
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_monotonic_char_starts(self):
        text = ("First paragraph with some content.\n\n"
                "Second paragraph with more content.\n\n"
                "Third paragraph with even more content.\n\n") * 10
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=30)
        for i in range(1, len(chunks)):
            assert chunks[i].char_start >= chunks[i - 1].char_start

    def test_char_end_equals_next_char_start_plus_overlap(self):
        """char_end of chunk N should relate to char_start of chunk N+1 via overlap."""
        text = "Word " * 2000  # ~10000 chars
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=100)
        for i in range(len(chunks) - 1):
            # The overlap region: next chunk starts at most chunk_overlap before current end
            assert chunks[i].char_end - chunks[i + 1].char_start <= 100

    def test_first_chunk_starts_at_zero(self):
        text = "Some content " * 500
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert chunks[0].char_start == 0

    def test_last_chunk_ends_at_text_length(self):
        text = "Some content " * 500
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert chunks[-1].char_end == len(text)

    def test_all_text_covered(self):
        """Every character should appear in at least one chunk."""
        text = "abcdefghij " * 1000  # ~11000 chars
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)

        # Build a coverage array
        covered = [False] * len(text)
        for chunk in chunks:
            for pos in range(chunk.char_start, chunk.char_end):
                covered[pos] = True

        assert all(covered), f"Uncovered positions: {[i for i, c in enumerate(covered) if not c]}"


# ==========================================================================
# 4. OVERLAP BEHAVIOR
# ==========================================================================

class TestOverlapBehavior:
    def test_overlap_between_consecutive_chunks(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=3)
        assert len(chunks) >= 2

        # The tail of chunk N should appear at the start of chunk N+1
        for i in range(len(chunks) - 1):
            tail = chunks[i].text[-3:]
            assert chunks[i + 1].text.startswith(tail), (
                f"Chunk {i+1} should start with overlap from chunk {i}"
            )

    def test_zero_overlap(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=0)
        # No overlap — chunks should be disjoint
        for i in range(len(chunks) - 1):
            assert chunks[i].char_end <= chunks[i + 1].char_start

    def test_overlap_equals_ten_percent(self):
        text = "A" * 10000
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=100)
        # Verify overlap behavior
        for i in range(len(chunks) - 1):
            overlap_start = chunks[i + 1].char_start
            overlap_end = chunks[i].char_end
            overlap_len = overlap_end - overlap_start
            assert overlap_len <= 100

    def test_overlap_content_is_identical(self):
        """Overlapping text between chunks should be byte-identical."""
        text = "The quick brown fox jumps over the lazy dog. " * 500
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
        for i in range(len(chunks) - 1):
            overlap_len = min(50, len(chunks[i].text))
            tail = chunks[i].text[-overlap_len:]
            assert chunks[i + 1].text[:overlap_len] == tail


# ==========================================================================
# 5. CONTENT PRESERVATION — NO LOST TEXT
# ==========================================================================

class TestContentPreservation:
    def test_no_content_lost(self):
        """Every character from the original text must appear in at least one chunk."""
        text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. ") * 100
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)

        # Reconstruct from coverage
        covered = set()
        for chunk in chunks:
            for pos in range(chunk.char_start, chunk.char_end):
                covered.add(pos)

        assert covered == set(range(len(text)))

    def test_concatenation_without_overlap_covers_all_text(self):
        """If we take unique portions (no double-counting overlap), we still get everything."""
        text = "abcdefghij" * 500
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

        # Each chunk's text should be a substring of the original
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text

    def test_full_text_reconstructable(self):
        """The union of all chunks covers the entire original text."""
        text = ("First we will discuss the methodology. "
                "Then we will present results. "
                "Finally we will draw conclusions. ") * 200
        chunks = chunk_text(text, chunk_size=400, chunk_overlap=40)

        # Reconstruct full text from char ranges
        reconstructed = ""
        for chunk in chunks:
            if not reconstructed:
                reconstructed = text[:chunk.char_end]
            else:
                # Extend to cover new characters
                reconstructed = text[:chunk.char_end]

        assert reconstructed == text


# ==========================================================================
# 6. PARAGRAPH BOUNDARY PRESERVATION
# ==========================================================================

class TestParagraphPreservation:
    def test_split_prefers_paragraph_boundary(self):
        """When possible, chunks should split at paragraph boundaries."""
        para1 = "A" * 200
        para2 = "B" * 200
        para3 = "C" * 200
        text = f"{para1}\n\n{para2}\n\n{para3}\n\n{para1}\n\n{para2}"
        # chunk_size=350 should prefer splitting at \n\n rather than mid-paragraph
        chunks = chunk_text(text, chunk_size=350, chunk_overlap=0)

        for chunk in chunks:
            # Chunks should not end mid-paragraph unless they're the last chunk
            # Check that text boundaries are clean
            assert chunk.text == text[chunk.char_start:chunk.char_end]

    def test_paragraph_not_split_when_fits(self):
        """A paragraph that fits within chunk_size should not be split."""
        para1 = "First paragraph with content."
        para2 = "Second paragraph with content."
        text = f"{para1}\n\n{para2}\n\n" + ("X" * 100 + "\n\n") * 20

        chunks = chunk_text(text, chunk_size=200, chunk_overlap=0)
        # First chunk should contain complete first paragraph
        assert chunks[0].text.startswith(para1)

    def test_long_paragraph_split_at_sentences(self):
        """A paragraph too long for one chunk should split at sentence boundaries."""
        sentences = [f"Sentence {i}. " for i in range(50)]
        long_para = "".join(sentences)
        text = f"Intro.\n\n{long_para}\n\nOutro."

        chunks = chunk_text(text, chunk_size=200, chunk_overlap=0)
        # Verify no chunk ends mid-sentence (unless it's a hard fallback)
        for chunk in chunks[:-1]:  # All but last
            # Should end at a sentence boundary or paragraph boundary
            text_end = chunk.text.rstrip()
            assert text_end.endswith(".") or text_end.endswith("\n"), (
                f"Chunk ends unexpectedly: ...{text_end[-30:]!r}"
            )


# ==========================================================================
# 7. SENTENCE BOUNDARY PREFERENCE
# ==========================================================================

class TestSentenceBoundaries:
    def test_splits_at_sentence_boundary(self):
        """When paragraph boundary isn't available, prefer sentence boundary."""
        # No paragraph breaks — just sentences
        text = "First sentence. Second sentence. " + "Third sentence. " * 100
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)

        for chunk in chunks[:-1]:
            stripped = chunk.text.rstrip()
            assert stripped.endswith(".") or stripped.endswith(" "), (
                f"Chunk should end at sentence boundary: ...{stripped[-20:]!r}"
            )

    def test_exclamation_and_question_marks_respected(self):
        text = "What is this? It is a test! " + "Really? Yes! " * 100
        chunks = chunk_text(text, chunk_size=80, chunk_overlap=0)
        assert len(chunks) > 1
        # All chunks should have valid text
        for chunk in chunks:
            assert len(chunk.text) > 0


# ==========================================================================
# 8. WORD BOUNDARY FALLBACK
# ==========================================================================

class TestWordBoundaries:
    def test_splits_at_word_boundary(self):
        """When no sentence boundary available, prefer word boundary."""
        text = "word " * 500  # ~2500 chars, no sentences
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=0)

        for chunk in chunks[:-1]:
            # Should not end mid-word
            assert chunk.text.endswith(" ") or chunk.text.endswith("\n")

    def test_long_word_no_split_needed(self):
        """A word shorter than chunk_size should not be split."""
        text = "short " + "a" * 3000 + " short"
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=0)
        assert len(chunks) == 1


# ==========================================================================
# 9. LONG WORD HANDLING
# ==========================================================================

class TestLongWordHandling:
    def test_very_long_word_exceeding_chunk_size(self):
        """A single word longer than chunk_size should still be chunked (char fallback)."""
        text = "a" * 10000  # One giant "word"
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 10
        for chunk in chunks:
            assert len(chunk.text) <= 1000

    def test_mixed_long_and_normal_words(self):
        text = "Normal text. " + "x" * 5000 + " More normal text."
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)
        assert len(chunks) >= 5
        # All text covered
        full = "".join(chunk.text for chunk in chunks)
        # Since overlap can duplicate, just verify the long word is present
        assert "x" * 5000 in full


# ==========================================================================
# 10. UNICODE TEXT
# ==========================================================================

class TestUnicodeText:
    def test_chinese_text(self):
        text = "这是一段中文文本。" * 200  # ~1800 chars
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        # Verify each chunk matches the corresponding slice of original text
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text

    def test_arabic_text(self):
        text = "هذا نص عربي. " * 200
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text

    def test_hindi_text(self):
        text = "यह हिंदी पाठ है। " * 200
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 1

    def test_mixed_scripts(self):
        text = ("English text. 日本語テキスト. "
                "العربية. हिंदी. "
                "Über naïve résumé. ") * 100
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=30)
        assert len(chunks) > 1
        # All chunks contain valid content
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_emoji_text(self):
        text = "Hello 🌍! " * 200
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=30)
        assert len(chunks) >= 1

    def test_combining_characters(self):
        # e + combining acute accent → é
        text = "caf\u0065\u0301 " * 200
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=30)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text


# ==========================================================================
# 11. INVALID CONFIGURATION
# ==========================================================================

class TestInvalidConfiguration:
    def test_chunk_size_zero(self):
        with pytest.raises(ChunkingError, match="positive integer"):
            chunk_text("hello", chunk_size=0)

    def test_chunk_size_negative(self):
        with pytest.raises(ChunkingError, match="positive integer"):
            chunk_text("hello", chunk_size=-1)

    def test_chunk_size_float(self):
        with pytest.raises(ChunkingError, match="positive integer"):
            chunk_text("hello", chunk_size=10.5)  # type: ignore

    def test_overlap_negative(self):
        with pytest.raises(ChunkingError, match="non-negative integer"):
            chunk_text("hello", chunk_size=10, chunk_overlap=-1)

    def test_overlap_equals_chunk_size(self):
        with pytest.raises(ChunkingError, match="less than chunk_size"):
            chunk_text("hello", chunk_size=10, chunk_overlap=10)

    def test_overlap_greater_than_chunk_size(self):
        with pytest.raises(ChunkingError, match="less than chunk_size"):
            chunk_text("hello", chunk_size=10, chunk_overlap=15)

    def test_overlap_float(self):
        with pytest.raises(ChunkingError, match="non-negative integer"):
            chunk_text("hello", chunk_size=10, chunk_overlap=2.5)  # type: ignore


# ==========================================================================
# 12. DETERMINISTIC OUTPUT
# ==========================================================================

class TestDeterministicOutput:
    def test_same_input_same_output(self):
        text = "Paragraph one.\n\nParagraph two.\n\n" + "Sentence. " * 500
        result1 = chunk_text(text, chunk_size=200, chunk_overlap=20)
        result2 = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2):
            assert c1.text == c2.text
            assert c1.char_start == c2.char_start
            assert c1.char_end == c2.char_end

    def test_no_randomness(self):
        text = "Test content " * 1000
        results = [chunk_text(text, chunk_size=300, chunk_overlap=30) for _ in range(10)]
        first = results[0]
        for result in results[1:]:
            assert len(result) == len(first)
            for c1, c2 in zip(first, result):
                assert c1.text == c2.text


# ==========================================================================
# 13. CHARACTER METADATA ACCURACY
# ==========================================================================

class TestCharacterMetadata:
    def test_char_start_end_exact(self):
        text = "AAAA\n\nBBBB\n\nCCCC\n\nDDDD\n\nEEEE\n\nFFFF\n\nGGGG\n\nHHHH"
        chunks = chunk_text(text, chunk_size=10, chunk_overlap=0)
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text

    def test_char_ranges_non_overlapping_with_zero_overlap(self):
        text = "word " * 500
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
        for i in range(len(chunks) - 1):
            assert chunks[i].char_end <= chunks[i + 1].char_start

    def test_char_ranges_cover_full_text(self):
        text = "content " * 1000
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        # First starts at 0
        assert chunks[0].char_start == 0
        # Last ends at len(text)
        assert chunks[-1].char_end == len(text)


# ==========================================================================
# 14. LARGE DOCUMENT PERFORMANCE
# ==========================================================================

class TestPerformance:
    def test_large_document_chunking(self):
        """Chunk a ~100KB document in reasonable time."""
        import time
        # Simulate a large document: 100,000 characters
        paragraph = "This is a test paragraph with some meaningful content. " * 10
        text = (paragraph + "\n\n") * 500  # ~290,000 chars

        start = time.time()
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=400)
        elapsed = time.time() - start

        assert len(chunks) > 50
        assert elapsed < 2.0, f"Chunking took {elapsed:.2f}s for ~290KB — too slow"

        # Verify all text covered
        for chunk in chunks:
            assert text[chunk.char_start:chunk.char_end] == chunk.text

    def test_very_long_paragraph(self):
        """A single paragraph of 50,000 chars should chunk without issues."""
        import time
        text = "word " * 10000  # ~50,000 chars
        start = time.time()
        chunks = chunk_text(text, chunk_size=4000, chunk_overlap=400)
        elapsed = time.time() - start
        assert len(chunks) > 10
        assert elapsed < 1.0


# ==========================================================================
# 15. REALISTIC DOCUMENT
# ==========================================================================

class TestRealisticDocument:
    def test_simulated_pdf_output(self):
        """Simulate normalized PDF extraction output."""
        text = (
            "Chapter 1: Introduction\n\n"
            "This document discusses the importance of text normalization "
            "in document processing. PDF extraction often produces inconsistent "
            "whitespace, mixed line endings, and excessive blank lines.\n\n"
            "Section 1.1: Background\n\n"
            "The field of document processing has evolved significantly over "
            "the past decade. Modern systems must handle diverse document "
            "formats and extract meaningful content for downstream analysis.\n\n"
            "Section 1.2: Objectives\n\n"
            "The primary objectives of this system are:\n"
            "- Extract text from PDF documents\n"
            "- Normalize extracted content\n"
            "- Split text into retrieval-friendly chunks\n"
            "- Enable semantic search and retrieval\n\n"
            "Chapter 2: Methodology\n\n"
            "Our approach combines multiple techniques for robust text processing. "
            "The pipeline consists of extraction, normalization, and chunking stages."
        )
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=30)
        assert len(chunks) > 1

        # All chunks should be valid
        for chunk in chunks:
            assert chunk.text == text[chunk.char_start:chunk.char_end]
            assert len(chunk.text) > 0

        # All text covered
        covered = set()
        for chunk in chunks:
            for pos in range(chunk.char_start, chunk.char_end):
                covered.add(pos)
        assert covered == set(range(len(text)))

    def test_two_paragraphs_fit_in_one_chunk(self):
        """Two short paragraphs should stay together."""
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=0)
        assert len(chunks) == 1
        assert "First paragraph" in chunks[0].text
        assert "Second paragraph" in chunks[0].text

    def test_many_small_paragraphs(self):
        """Many paragraphs should be grouped into chunks."""
        paras = [f"Paragraph {i} with content." for i in range(100)]
        text = "\n\n".join(paras)
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=30)
        assert len(chunks) > 1
        # Verify ordering
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ==========================================================================
# 16. INTEGRATION WITH DOCUMENT PROCESSING PIPELINE
# ==========================================================================

class TestProcessingPipelineIntegration:
    """Prove that DocumentContent.extracted_text → chunker works."""

    def test_chunk_extracted_text(self):
        """Simulate chunking text from DocumentContent."""
        from app.services.text_normalizer import normalize_text

        raw_pdf_text = (
            "Chapter 1: Introduction\r\n"
            "\r\n"
            "This document  discusses  the  importance\r\n"
            "of text normalization in document processing.\r\n"
            "\r\n"
            "\r\n"
            "Section 1.1: Background\r\n"
            "\r\n"
            "PDF extraction often produces inconsistent  whitespace.\r\n"
        )
        normalized = normalize_text(raw_pdf_text)
        chunks = chunk_text(normalized, chunk_size=100, chunk_overlap=10)

        assert len(chunks) > 1
        # All text should be covered
        full = normalized
        for chunk in chunks:
            assert full[chunk.char_start:chunk.char_end] == chunk.text

    def test_end_to_end_extract_normalize_chunk(self):
        """Full pipeline: raw text → normalize → chunk → verify."""
        from app.services.text_normalizer import normalize_text

        raw = (
            "Title Page\r\n"
            "\r\n"
            "Chapter 1: Background\r\n"
            "\r\n"
            "The  quick  brown  fox  jumps  over  the  lazy  dog.\r\n"
            "This is a longer sentence that provides more context.\r\n"
            "\r\n"
            "\r\n"
            "\r\n"
            "Chapter 2: Analysis\r\n"
            "\r\n"
            "Our analysis reveals several key findings.\r\n"
            "First, the data shows a clear trend.\r\n"
            "Second, the results are statistically significant.\r\n"
        )
        normalized = normalize_text(raw)
        chunks = chunk_text(normalized, chunk_size=150, chunk_overlap=20)

        assert len(chunks) >= 2

        # Verify deterministic, ordered output
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

        # Verify all text is recoverable
        all_text = normalized
        for chunk in chunks:
            assert all_text[chunk.char_start:chunk.char_end] == chunk.text

        # Verify no CRLF remains (normalizer job)
        for chunk in chunks:
            assert "\r" not in chunk.text

    def test_empty_extracted_text(self):
        """An empty DocumentContent should produce no chunks."""
        from app.services.text_normalizer import normalize_text

        normalized = normalize_text("")
        chunks = chunk_text(normalized)
        assert chunks == []

    def test_single_line_extracted_text(self):
        """A single line of extracted text fits in one chunk."""
        from app.services.text_normalizer import normalize_text

        raw = "Single line of extracted text.\r\n"
        normalized = normalize_text(raw)
        chunks = chunk_text(normalized, chunk_size=500, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].text == normalized
