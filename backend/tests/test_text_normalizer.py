"""Phase 4.1 tests: text normalization service.

Tests cover:
    - Empty / whitespace-only input
    - CRLF / CR / mixed line endings
    - Tab conversion
    - Trailing whitespace removal
    - Horizontal space collapsing
    - Blank line collapsing
    - Leading / trailing document whitespace
    - Unicode NFC normalization
    - Non-English characters preserved
    - Paragraphs preserved
    - Headings / punctuation preserved
    - Deterministic output
    - Integration with document processing pipeline
"""

import pytest

from app.services.text_normalizer import normalize_text


# ==========================================================================
# 1. EMPTY AND EDGE INPUTS
# ==========================================================================

class TestEmptyAndEdgeInputs:
    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_like_whitespace(self):
        assert normalize_text("   ") == ""

    def test_only_newlines(self):
        assert normalize_text("\n\n\n") == ""

    def test_only_tabs(self):
        assert normalize_text("\t\t") == ""

    def test_single_space(self):
        assert normalize_text(" ") == ""

    def test_single_character(self):
        assert normalize_text("a") == "a"

    def test_no_whitespace_at_all(self):
        assert normalize_text("HelloWorld") == "HelloWorld"


# ==========================================================================
# 2. LINE ENDING NORMALIZATION
# ==========================================================================

class TestLineEndingNormalization:
    def test_crlf_to_lf(self):
        assert normalize_text("line1\r\nline2") == "line1\nline2"

    def test_cr_to_lf(self):
        assert normalize_text("line1\rline2") == "line1\nline2"

    def test_mixed_endings(self):
        input_text = "line1\r\nline2\rline3\nline4"
        expected = "line1\nline2\nline3\nline4"
        assert normalize_text(input_text) == expected

    def test_lf_unchanged(self):
        assert normalize_text("line1\nline2") == "line1\nline2"

    def test_multiple_crlf_sequences(self):
        assert normalize_text("a\r\n\r\nb") == "a\n\nb"


# ==========================================================================
# 3. TAB CONVERSION
# ==========================================================================

class TestTabConversion:
    def test_single_tab_becomes_space(self):
        """Tabs are converted to spaces then collapsed to a single space."""
        assert normalize_text("col1\tcol2") == "col1 col2"

    def test_multiple_tabs_become_single_space(self):
        assert normalize_text("a\t\tb") == "a b"

    def test_tab_at_start_collapsed(self):
        """Tab at start is converted to spaces, then leading spaces are stripped."""
        assert normalize_text("\tHello") == "Hello"

    def test_tab_at_end_stripped(self):
        assert normalize_text("Hello\t") == "Hello"

    def test_tab_between_words_single_space(self):
        assert normalize_text("word1\tword2\tword3") == "word1 word2 word3"


# ==========================================================================
# 4. TRAILING WHITESPACE
# ==========================================================================

class TestTrailingWhitespace:
    def test_trailing_spaces_removed(self):
        assert normalize_text("Hello   ") == "Hello"

    def test_trailing_tabs_removed(self):
        assert normalize_text("Hello\t\t") == "Hello"

    def test_trailing_mixed_whitespace_removed(self):
        assert normalize_text("Hello \t ") == "Hello"

    def test_trailing_whitespace_preserved_on_each_line(self):
        input_text = "line1   \nline2  \nline3"
        expected = "line1\nline2\nline3"
        assert normalize_text(input_text) == expected


# ==========================================================================
# 5. HORIZONTAL SPACE COLLAPSING
# ==========================================================================

class TestHorizontalSpaceCollapsing:
    def test_multiple_spaces_collapsed(self):
        assert normalize_text("hello    world") == "hello world"

    def test_leading_spaces_collapsed(self):
        assert normalize_text("  hello") == "hello"

    def test_mixed_space_types_collapsed(self):
        assert normalize_text("hello   \t  world") == "hello world"

    def test_single_space_preserved(self):
        assert normalize_text("hello world") == "hello world"

    def test_newlines_not_collapsed(self):
        """Newlines are NOT treated as horizontal spaces."""
        assert normalize_text("a\n\nb") == "a\n\nb"


# ==========================================================================
# 6. BLANK LINE COLLAPSING
# ==========================================================================

class TestBlankLineCollapsing:
    def test_three_blank_lines_to_two(self):
        input_text = "para1\n\n\n\npara2"
        expected = "para1\n\npara2"
        assert normalize_text(input_text) == expected

    def test_four_blank_lines_to_two(self):
        input_text = "para1\n\n\n\n\npara2"
        expected = "para1\n\npara2"
        assert normalize_text(input_text) == expected

    def test_two_blank_lines_preserved(self):
        input_text = "para1\n\npara2"
        assert normalize_text(input_text) == "para1\n\npara2"

    def test_one_blank_line_preserved(self):
        input_text = "para1\npara2"
        assert normalize_text(input_text) == "para1\npara2"

    def test_many_blank_lines_collapsed(self):
        input_text = "a\n\n\n\n\n\n\n\n\nb"
        expected = "a\n\nb"
        assert normalize_text(input_text) == expected


# ==========================================================================
# 7. LEADING / TRAILING DOCUMENT WHITESPACE
# ==========================================================================

class TestDocumentWhitespaceStripping:
    def test_leading_newlines_stripped(self):
        assert normalize_text("\n\nHello") == "Hello"

    def test_trailing_newlines_stripped(self):
        assert normalize_text("Hello\n\n") == "Hello"

    def test_leading_spaces_stripped(self):
        assert normalize_text("   Hello") == "Hello"

    def test_trailing_spaces_stripped(self):
        assert normalize_text("Hello   ") == "Hello"

    def test_surrounding_whitespace_stripped(self):
        assert normalize_text("  \n  Hello  \n  ") == "Hello"

    def test_leading_blank_lines_with_content(self):
        input_text = "\n\n\nHello world\n\n"
        assert normalize_text(input_text) == "Hello world"


# ==========================================================================
# 8. UNICODE NORMALIZATION
# ==========================================================================

class TestUnicodeNormalization:
    def test_nfc_normalization_combining_chars(self):
        """é as combining sequence should be NFC-normalized."""
        # e + combining acute accent (U+0301) → é (U+00E9)
        input_text = "caf\u0065\u0301"
        expected = "caf\u00e9"
        assert normalize_text(input_text) == expected

    def test_nfc_does_not_change_precomposed(self):
        """Already precomposed é stays as é."""
        input_text = "caf\u00e9"
        assert normalize_text(input_text) == "caf\u00e9"

    def test_cjk_characters_preserved(self):
        input_text = "日本語テスト"
        assert normalize_text(input_text) == "日本語テスト"

    def test_arabic_characters_preserved(self):
        input_text = "مرحبا بالعالم"
        assert normalize_text(input_text) == "مرحبا بالعالم"

    def test_emoji_preserved(self):
        input_text = "Hello 🌍!"
        assert normalize_text(input_text) == "Hello 🌍!"

    def test_latin_extended_preserved(self):
        input_text = "Über naïve résumé"
        assert normalize_text(input_text) == "Über naïve résumé"

    def test_mixed_scripts(self):
        input_text = "English 日本語 عربي 123"
        assert normalize_text(input_text) == "English 日本語 عربي 123"


# ==========================================================================
# 9. MEANINGFUL CONTENT PRESERVATION
# ==========================================================================

class TestMeaningfulContentPreservation:
    def test_paragraphs_preserved(self):
        input_text = "First paragraph.\n\nSecond paragraph."
        assert normalize_text(input_text) == "First paragraph.\n\nSecond paragraph."

    def test_headings_preserved(self):
        input_text = "Chapter 1: Introduction\n\nSome text here."
        assert normalize_text(input_text) == "Chapter 1: Introduction\n\nSome text here."

    def test_punctuation_preserved(self):
        input_text = "Hello, world! How are you? I'm fine."
        assert normalize_text(input_text) == "Hello, world! How are you? I'm fine."

    def test_numbers_preserved(self):
        input_text = "Price: $19.99 (was $29.99)"
        assert normalize_text(input_text) == "Price: $19.99 (was $29.99)"

    def test_bullets_preserved(self):
        input_text = "• Item 1\n• Item 2\n• Item 3"
        assert normalize_text(input_text) == "• Item 1\n• Item 2\n• Item 3"

    def test_hyphenated_words_preserved(self):
        input_text = "well-known self-documenting"
        assert normalize_text(input_text) == "well-known self-documenting"

    def test_code_like_content_preserved(self):
        """Code content is preserved but indentation (horizontal whitespace) is collapsed."""
        input_text = "def foo():\n    return 42"
        assert normalize_text(input_text) == "def foo():\n return 42"

    def test_urls_preserved(self):
        input_text = "Visit https://example.com/path?q=1 for info."
        assert normalize_text(input_text) == "Visit https://example.com/path?q=1 for info."


# ==========================================================================
# 10. DETERMINISTIC OUTPUT
# ==========================================================================

class TestDeterministicOutput:
    def test_same_input_same_output(self):
        text = "Hello  world\n\n\n   Test   "
        assert normalize_text(text) == normalize_text(text)

    def test_repeated_normalization_idempotent(self):
        """Normalizing the output of normalization should produce the same result."""
        text = "Hello  world\n\n\n   Test   \r\nOther\tlines"
        once = normalize_text(text)
        twice = normalize_text(once)
        assert once == twice

    def test_various_inputs_all_deterministic(self):
        inputs = [
            "",
            "Hello",
            "Hello  world",
            "\n\nHello\n\n",
            "line1\r\nline2\rline3\nline4",
            "café",
            "日本語",
        ]
        for text in inputs:
            assert normalize_text(text) == normalize_text(text)


# ==========================================================================
# 11. REALISTIC PDF EXTRACTION INPUT
# ==========================================================================

class TestRealisticPDFInput:
    def test_typical_pdf_output(self):
        """Simulate raw output from PyMuPDF extraction with typical artifacts."""
        raw = (
            "Chapter 1: Introduction\r\n"
            "\r\n"
            "This document  discusses  the  importance\r\n"
            "of text normalization in document processing.\r\n"
            "\r\n"
            "\r\n"
            "\r\n"
            "Section 1.1: Background\r\n"
            "\r\n"
            "PDF extraction often produces inconsistent  whitespace,\r\n"
            "mixed line endings, and excessive blank lines.\r\n"
            "\r\n"
            "   - Item one  \r\n"
            "   - Item two  \r\n"
        )
        result = normalize_text(raw)

        # No CRLF
        assert "\r" not in result
        # No trailing spaces on lines
        for line in result.split("\n"):
            assert not line.endswith(" ")
        # No excessive blank lines
        assert "\n\n\n" not in result
        # Content preserved
        assert "Chapter 1: Introduction" in result
        assert "text normalization" in result
        assert "Section 1.1: Background" in result
        # Whitespace collapsed
        assert "  " not in result.replace("\n\n", "|||")

    def test_multi_page_pdf_concatenation(self):
        """Simulate text from multiple pages joined by newlines."""
        page1 = "Page 1 content\r\nWith two lines.\r\n"
        page2 = "Page 2 content\r\nMore text here.\r\n"
        raw = page1 + "\r\n" + page2
        result = normalize_text(raw)

        assert "Page 1 content" in result
        assert "Page 2 content" in result
        assert "More text here" in result
        assert "\r" not in result

    def test_unicode_heavy_document(self):
        """Simulate a document with mixed scripts."""
        raw = (
            "Title: Überblick über Dokumente\n"
            "\n"
            "概要：この文書は...  \n"
            "\n"
            "Résumé: هذا ملخص\n"
        )
        result = normalize_text(raw)
        assert "Überblick" in result
        assert "概要" in result
        assert "Résumé" in result
        assert "هذا ملخص" in result


# ==========================================================================
# 12. INTEGRATION WITH PROCESSING PIPELINE
# ==========================================================================

class TestProcessingPipelineIntegration:
    """Verify normalization is called during document processing."""

    def test_process_document_normalizes_text(self):
        """Test that process_document stores normalized text in DocumentContent."""
        from unittest.mock import patch, MagicMock
        from app.services.document_processor import process_document

        mock_result = MagicMock()
        mock_result.text = "Raw  text  with   excessive   spaces\r\nand\r\n"
        mock_result.page_count = 1
        mock_result.char_count = 50

        mock_db = MagicMock()
        mock_doc = MagicMock()
        mock_doc.id = 1
        mock_doc.status = "UPLOADED"
        mock_doc.storage_key = "test_key"
        mock_doc.content = None

        mock_query = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_doc
        # For DocumentContent query
        mock_db.query.return_value.filter.return_value.first.side_effect = [mock_doc, None]

        with patch("app.services.document_processor.extract_text_from_pdf", return_value=mock_result):
            with patch("app.services.document_processor.normalize_text", wraps=__import__("app.services.text_normalizer", fromlist=["normalize_text"]).normalize_text) as mock_normalize:
                process_document(mock_db, 1)

                # normalize_text should have been called with the raw text
                mock_normalize.assert_called_once_with("Raw  text  with   excessive   spaces\r\nand\r\n")

    def test_normalization_called_before_content_save(self):
        """Ensure normalized text (not raw) is passed to DocumentContent."""
        from unittest.mock import patch, MagicMock
        from app.services.document_processor import process_document
        from app.models.job import ProcessingJob

        raw_text = "Hello    world\r\n\n\n\nMore  text"
        expected_normalized = "Hello world\n\nMore text"

        mock_result = MagicMock()
        mock_result.text = raw_text
        mock_result.page_count = 1
        mock_result.char_count = len(raw_text)

        mock_db = MagicMock()
        mock_doc = MagicMock()
        mock_doc.id = 1
        mock_doc.user_id = 1
        mock_doc.status = "UPLOADED"
        mock_doc.storage_key = "test_key"
        mock_doc.content = None  # No existing content → upsert creates new

        # First query: Document lookup → mock_doc
        # Second query: ProcessingJob lookup → None (no existing job)
        # Third query: DocumentContent lookup → None (no existing content)
        query_mock = MagicMock()
        filter_mock = MagicMock()
        query_mock.filter.return_value = filter_mock
        filter_mock.first.side_effect = [mock_doc, None, None]
        mock_db.query.return_value = query_mock

        with patch("app.services.document_processor.extract_text_from_pdf", return_value=mock_result):
            process_document(mock_db, 1)

            # Check that DocumentContent was created with normalized text
            # First add is ProcessingJob, second add is DocumentContent
            db_add_calls = mock_db.add.call_args_list
            assert len(db_add_calls) == 2
            
            # First add should be ProcessingJob
            job_arg = db_add_calls[0][0][0]
            assert isinstance(job_arg, ProcessingJob)
            
            # Second add should be DocumentContent
            content_arg = db_add_calls[1][0][0]
            assert content_arg.extracted_text == expected_normalized
            assert content_arg.char_count == len(expected_normalized)
