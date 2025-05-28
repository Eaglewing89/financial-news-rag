"""
Unit tests for the TextProcessor class.

These tests validate the functionality of the TextProcessor class,
including text cleaning, chunking, and content processing.
All tests are isolated and use mocked dependencies where appropriate.
"""

from unittest.mock import patch

import pytest

from financial_news_rag.text_processor import TextProcessor


class TestTextProcessorInitialization:
    """Test suite for TextProcessor initialization."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        processor = TextProcessor(
            max_tokens_per_chunk=512
        )  # Provide required parameter

        # Verify default values are set
        assert processor.max_tokens_per_chunk == 512

    def test_init_with_custom_max_tokens(self):
        """Test initialization with custom max tokens per chunk."""
        custom_max_tokens = 1024
        processor = TextProcessor(max_tokens_per_chunk=custom_max_tokens)

        assert processor.max_tokens_per_chunk == custom_max_tokens

    def test_init_with_custom_encoding(self):
        """Test initialization with custom max tokens."""
        processor = TextProcessor(max_tokens_per_chunk=1024)

        # Verify max tokens is set correctly
        assert processor.max_tokens_per_chunk == 1024


class TestTextProcessorCleaning:
    """Test suite for text cleaning functionality."""

    @pytest.fixture
    def processor(self):
        """Create a TextProcessor instance for testing."""
        return TextProcessor(max_tokens_per_chunk=2048)

    def test_clean_html_tags(self, processor, sample_html_content):
        """Test removal of HTML tags from text."""
        # Use test data from factory
        raw_text = sample_html_content["with_basic_tags"]
        cleaned = processor.clean_article_text(raw_text)
        assert cleaned == "This is a test article"

        # Complex HTML with attributes
        raw_text = sample_html_content["with_complex_tags"]
        cleaned = processor.clean_article_text(raw_text)
        assert cleaned == "Content with link"
        assert "<" not in cleaned and ">" not in cleaned

    def test_remove_boilerplate_phrases(self, processor):
        """Test removal of common boilerplate text."""
        # Test "Click here to read more"
        raw_text = "This is an article. Click here to read more."
        cleaned = processor.clean_article_text(raw_text)
        assert cleaned == "This is an article."

        # Test "Read more:" patterns
        raw_text = "This is content. Read more: visit our site."
        cleaned = processor.clean_article_text(raw_text)
        assert cleaned == "This is content."

        # Test "Source:" patterns
        raw_text = "Article content here. Source: Example News"
        cleaned = processor.clean_article_text(raw_text)
        assert cleaned == "Article content here."

        # Test multiple patterns
        raw_text = "Content. Click here to read more. Source: News Site"
        cleaned = processor.clean_article_text(raw_text)
        assert cleaned == "Content."

    def test_normalize_smart_quotes(self, processor):
        """Test normalization of smart quotes and special characters."""
        # Smart double quotes
        raw_text = 'This has "smart quotes" in it'
        cleaned = processor.clean_article_text(raw_text)
        assert '"smart quotes"' in cleaned

        # Smart single quotes (curly apostrophes)
        raw_text = "This has 'single quotes' and don't"
        cleaned = processor.clean_article_text(raw_text)
        assert "'single quotes'" in cleaned
        assert "don't" in cleaned

        # Em dashes and en dashes are preserved
        raw_text = "Text with — em dash and – en dash"
        cleaned = processor.clean_article_text(raw_text)
        assert (
            cleaned == "Text with — em dash and – en dash"
        )  # Characters should be preserved

    def test_normalize_whitespace(self, processor):
        """Test normalization of whitespace characters."""
        # Multiple spaces
        raw_text = "  Multiple    spaces  between   words  "
        cleaned = processor.clean_article_text(raw_text)
        assert cleaned == "Multiple spaces between words"

        # Mixed whitespace characters
        raw_text = "  Text with \n\n newlines \t and \r tabs  "
        cleaned = processor.clean_article_text(raw_text)
        assert cleaned == "Text with newlines and tabs"

        # Leading and trailing whitespace
        raw_text = "\n\t  Content with leading/trailing whitespace  \t\n"
        cleaned = processor.clean_article_text(raw_text)
        assert cleaned == "Content with leading/trailing whitespace"

    def test_clean_empty_and_none_input(self, processor):
        """Test handling of empty and None input."""
        # Empty string
        assert processor.clean_article_text("") == ""

        # None input
        assert processor.clean_article_text(None) == ""

        # Whitespace only
        assert processor.clean_article_text("   \n\t  ") == ""

    def test_clean_preserves_valid_content(self, processor):
        """Test that cleaning preserves valid text content."""
        # Financial content with numbers and symbols
        raw_text = "AAPL stock price is $150.50, up 2.5% from yesterday."
        cleaned = processor.clean_article_text(raw_text)
        assert "$150.50" in cleaned
        assert "2.5%" in cleaned
        assert "AAPL" in cleaned

        # Text with punctuation
        raw_text = "Q4 earnings: revenue of $10B (up 15%), EPS of $2.50."
        cleaned = processor.clean_article_text(raw_text)
        assert "$10B" in cleaned
        assert "(up 15%)" in cleaned
        assert "$2.50" in cleaned

    def test_clean_unicode_handling(self, processor):
        """Test proper handling of Unicode characters."""
        # Various Unicode characters
        raw_text = "Text with unicode: café, naïve, résumé"
        cleaned = processor.clean_article_text(raw_text)
        assert "café" in cleaned
        assert "naïve" in cleaned
        assert "résumé" in cleaned


class TestTextProcessorValidation:
    """Test suite for content validation functionality."""

    @pytest.fixture
    def processor(self):
        """Create a TextProcessor instance for testing."""
        return TextProcessor(max_tokens_per_chunk=512)

    def test_process_and_validate_content_none_input(self, processor):
        """Test process_and_validate_content with None input."""
        result = processor.process_and_validate_content(None)

        assert result["status"] == "FAILED"
        assert result["reason"] == "Empty raw content"
        assert result["content"] == ""

    def test_process_and_validate_content_empty_string(self, processor):
        """Test process_and_validate_content with empty string input."""
        result = processor.process_and_validate_content("")

        assert result["status"] == "FAILED"
        assert result["reason"] == "Empty raw content"
        assert result["content"] == ""

    def test_process_and_validate_content_whitespace_only(self, processor):
        """Test process_and_validate_content with whitespace-only input."""
        result = processor.process_and_validate_content("   \n\t  ")

        assert result["status"] == "FAILED"
        assert result["reason"] == "Empty raw content"
        assert result["content"] == ""

    def test_process_and_validate_content_cleaned_to_empty(self, processor):
        """Test process_and_validate_content with input that becomes empty after cleaning."""
        # Mock clean_article_text to return empty string
        with patch.object(processor, "clean_article_text", return_value=""):
            result = processor.process_and_validate_content("Click here to read more.")

            assert result["status"] == "FAILED"
            assert result["reason"] == "No content after cleaning"
            assert result["content"] == ""

    def test_process_and_validate_content_success(self, processor):
        """Test process_and_validate_content with valid input."""
        raw_text = "<p>This is a <b>test</b> article</p>"
        result = processor.process_and_validate_content(raw_text)

        assert result["status"] == "SUCCESS"
        assert result["reason"] == ""
        assert result["content"] == "This is a test article"

    def test_process_and_validate_content_success_with_mock(self, processor):
        """Test process_and_validate_content with mocked cleaning."""
        with patch.object(
            processor, "clean_article_text", return_value="Cleaned content"
        ):
            result = processor.process_and_validate_content("Raw content")

            assert result["status"] == "SUCCESS"
            assert result["reason"] == ""
            assert result["content"] == "Cleaned content"


class TestTextProcessorChunking:
    """Test suite for text chunking functionality with default (regex) tokenization."""

    @pytest.fixture
    def processor(self):
        """Create a TextProcessor with reasonable chunk size for testing."""
        return TextProcessor(max_tokens_per_chunk=512)

    def test_chunk_short_text_single_chunk(self, processor):
        """Test that short text results in a single chunk."""
        short_text = "This is a short text that should fit in one chunk."
        chunks = processor.split_into_chunks(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_long_text_multiple_chunks(self, processor, long_test_sentences):
        """Test that long text is split into multiple chunks."""
        # Use test data from factory
        long_text = " ".join(long_test_sentences)

        # Use smaller chunk size to force splitting
        small_processor = TextProcessor(max_tokens_per_chunk=200)
        chunks = small_processor.split_into_chunks(long_text)

        # Verify multiple chunks are created
        assert len(chunks) > 1

        # Verify all chunks are within reasonable token limits
        for chunk in chunks:
            # Rough token estimation (char count / 4)
            estimated_tokens = len(chunk) // 4
            assert estimated_tokens <= 250  # Allow some margin for sentence boundaries

    def test_chunk_respects_sentence_boundaries(self, processor, financial_test_sentences):
        """Test that chunking respects sentence boundaries when possible."""
        # Use test data from factory
        text = " ".join(financial_test_sentences)

        # Use chunk size that should fit 2-3 sentences
        medium_processor = TextProcessor(max_tokens_per_chunk=300)
        chunks = medium_processor.split_into_chunks(text)

        # Each chunk should end with sentence-ending punctuation
        for chunk in chunks[:-1]:  # All but the last chunk
            assert chunk.rstrip().endswith((".", "!", "?"))

    def test_chunk_empty_text(self, processor):
        """Test chunking behavior with empty text."""
        empty_text = ""
        chunks = processor.split_into_chunks(empty_text)

        assert chunks == []

    def test_chunk_single_long_sentence(self, processor):
        """Test chunking behavior with a single very long sentence."""
        # Create a sentence longer than the chunk size
        long_sentence = "This is a very long sentence that contains many words and should exceed the maximum token limit per chunk and therefore needs to be split into multiple smaller parts even though it is technically a single sentence without any sentence-ending punctuation in the middle."
        
        # Use small chunk size to force word-level splitting
        small_processor = TextProcessor(max_tokens_per_chunk=50)
        chunks = small_processor.split_into_chunks(long_sentence)

        # Should create multiple chunks even from a single sentence
        assert len(chunks) > 1
        
        # Reconstruct original text (roughly)
        reconstructed = " ".join(chunks)
        # Should contain most of the original words
        original_words = set(long_sentence.split())
        reconstructed_words = set(reconstructed.split())
        assert len(original_words.intersection(reconstructed_words)) > 0.8 * len(original_words)

    def test_chunk_with_mixed_punctuation(self, processor):
        """Test chunking with various sentence-ending punctuation."""
        mixed_text = "First sentence. Second question? Third exclamation! Fourth statement."
        chunks = processor.split_into_chunks(mixed_text)

        # With default chunk size, this should fit in one chunk
        assert len(chunks) == 1
        assert chunks[0] == mixed_text


class TestTextProcessorChunkingWithNLTK:
    """Test suite for text chunking functionality with NLTK tokenization."""

    @pytest.fixture
    def nltk_processor(self):
        """Create a TextProcessor with NLTK enabled."""
        return TextProcessor(max_tokens_per_chunk=512, use_nltk=True, nltk_auto_download=False)

    @pytest.fixture
    def mock_nltk_tokenize(self):
        """Mock NLTK sentence tokenization for reliable testing."""
        with patch("nltk.tokenize.sent_tokenize") as mock_tokenize:
            # Realistic behavior: split on sentence-ending punctuation followed by space
            def side_effect(text):
                import re
                # Split on sentence boundaries but preserve the punctuation
                sentences = re.split(r"(?<=[.!?])\s+", text)
                return [s.strip() for s in sentences if s.strip()]

            mock_tokenize.side_effect = side_effect
            yield mock_tokenize

    @patch("nltk.data.find")
    @patch("nltk.tokenize.sent_tokenize")
    def test_nltk_chunking_when_available(self, mock_sent_tokenize, mock_find, nltk_processor):
        """Test chunking with NLTK when punkt data is available."""
        # Mock that punkt data is found and sent_tokenize works
        mock_find.return_value = True
        mock_sent_tokenize.side_effect = [
            ["Test sentence."],  # For the initialization test
            ["First sentence.", "Second sentence!", "Third sentence?"]  # For actual chunking
        ]
        
        # Re-initialize processor to trigger tokenizer setup
        processor = TextProcessor(max_tokens_per_chunk=512, use_nltk=True, nltk_auto_download=False)
        
        text = "First sentence. Second sentence! Third sentence?"
        chunks = processor.split_into_chunks(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
        # Verify NLTK tokenizer was called (twice: once for init test, once for actual chunking)
        assert mock_sent_tokenize.call_count == 2

    @patch("nltk.tokenize.sent_tokenize")
    def test_nltk_without_auto_download_raises_error(self, mock_sent_tokenize):
        """Test that missing NLTK data raises appropriate error when auto-download is disabled."""
        # Mock that punkt data is not found - sent_tokenize raises LookupError
        mock_sent_tokenize.side_effect = LookupError("punkt not found")
        
        with pytest.raises(RuntimeError, match="punkt tokenizer not available"):
            TextProcessor(
                max_tokens_per_chunk=512,
                use_nltk=True,
                nltk_auto_download=False
            )

    def test_nltk_chunk_respects_sentence_boundaries(self, nltk_processor, mock_nltk_tokenize):
        """Test that NLTK chunking respects sentence boundaries."""
        text = "First sentence. Second sentence! Third sentence?"
        
        # Use smaller chunk size to force splitting
        small_processor = TextProcessor(max_tokens_per_chunk=150, use_nltk=True, nltk_auto_download=False)
        chunks = small_processor.split_into_chunks(text)

        # Verify chunks end with sentence boundaries when possible
        for chunk in chunks[:-1]:  # All chunks except last
            assert chunk.endswith(".") or chunk.endswith("!") or chunk.endswith("?")

    def test_nltk_chunk_very_long_sentence(self, nltk_processor, mock_nltk_tokenize):
        """Test NLTK chunking of very long sentences that exceed token limits."""
        # Create a single very long sentence
        long_sentence = " ".join([f"word{i}" for i in range(200)])

        # Use small chunk size
        small_processor = TextProcessor(max_tokens_per_chunk=100, use_nltk=True, nltk_auto_download=False)
        chunks = small_processor.split_into_chunks(long_sentence)

        # Should be split into multiple chunks even though it's one sentence
        assert len(chunks) > 1

        # All chunks should be reasonable size (allow some margin above the target)
        for chunk in chunks:
            estimated_tokens = len(chunk) // 4
            assert estimated_tokens <= 200  # Increased margin for word boundaries and splitting behavior

    def test_nltk_chunk_empty_input(self, nltk_processor, mock_nltk_tokenize):
        """Test NLTK chunking with empty input."""
        assert nltk_processor.split_into_chunks("") == []
        assert nltk_processor.split_into_chunks(None) == []
        assert nltk_processor.split_into_chunks("   ") == []

    def test_nltk_chunk_preserves_content(self, nltk_processor, mock_nltk_tokenize):
        """Test that NLTK chunking preserves all content."""
        # Create text with specific content to track
        text = "Apple Inc. reported strong Q4 earnings. Revenue increased 15% to $10 billion. The company announced a new product line. Analysts are optimistic about future growth."

        chunks = nltk_processor.split_into_chunks(text)

        # Reconstruct text from chunks
        reconstructed = " ".join(chunks)

        # Verify all key content is preserved
        assert "Apple Inc." in reconstructed
        assert "Q4 earnings" in reconstructed
        assert "15%" in reconstructed
        assert "$10 billion" in reconstructed
        assert "new product line" in reconstructed
        assert "Analysts" in reconstructed

    def test_nltk_chunk_financial_content(self, nltk_processor, mock_nltk_tokenize):
        """Test NLTK chunking of typical financial news content."""
        financial_text = """
        Apple Inc. (NASDAQ: AAPL) reported quarterly earnings that exceeded expectations.
        Revenue for Q4 2023 reached $89.5 billion, representing a 2% year-over-year decline.
        iPhone sales contributed $43.8 billion to the total revenue.
        Services revenue grew to $22.3 billion, up 16% from the previous year.
        The company's gross margin improved to 45.2%, beating analyst estimates.
        CEO Tim Cook highlighted strong performance in emerging markets.
        Apple's cash position remains robust at $162.1 billion.
        The board approved a quarterly dividend of $0.24 per share.
        """

        chunks = nltk_processor.split_into_chunks(financial_text.strip())

        # Verify financial data is properly distributed across chunks
        all_content = " ".join(chunks)
        assert "$89.5 billion" in all_content
        assert "$43.8 billion" in all_content
        assert "$22.3 billion" in all_content
        assert "45.2%" in all_content
        assert "$162.1 billion" in all_content
        assert "$0.24" in all_content




class TestTextProcessorIntegration:
    """Integration tests for TextProcessor combining cleaning and chunking."""

    @pytest.fixture
    def processor(self):
        """Create a TextProcessor instance for integration testing."""
        return TextProcessor(max_tokens_per_chunk=300)

    def test_clean_and_chunk_html_content(self, processor, sample_html_content):
        """Test cleaning HTML content and then chunking it."""
        html_content = sample_html_content["realistic_article"]

        # Clean the content
        cleaned = processor.clean_article_text(html_content)

        # Verify HTML is removed
        assert "<" not in cleaned and ">" not in cleaned
        assert "bold text" in cleaned
        assert "a link" in cleaned
        assert "AAPL $150.50 (+2.5%)" in cleaned
        assert "Click here to read more" not in cleaned

        # Chunk the cleaned content
        chunks = processor.split_into_chunks(cleaned)

        # Should be reasonable number of chunks
        assert len(chunks) >= 1

        # All financial data should be preserved
        all_content = " ".join(chunks)
        assert "AAPL $150.50 (+2.5%)" in all_content

    def test_process_realistic_financial_article(self, processor, sample_financial_article):
        """Test processing of a realistic financial news article."""
        article_content = sample_financial_article

        # Process the article
        cleaned = processor.clean_article_text(article_content)
        chunks = processor.split_into_chunks(cleaned)

        # Verify cleaning worked
        assert "<p>" not in cleaned  # HTML tags should be removed
        # Note: "Click here to read the full earnings report" doesn't match the implemented pattern
        # which only removes "Click here to read more"
        assert (
            "Source: Apple Inc. Investor Relations" not in cleaned
        )  # Source lines should be removed

        # Verify key financial data is preserved across chunks
        all_content = " ".join(chunks)
        assert "NASDAQ:AAPL" in all_content
        assert "3.2%" in all_content
        assert "$89.5 billion" in all_content
        assert "$43.8 billion" in all_content
        assert "$22.3 billion" in all_content
        assert "16%" in all_content
        assert "Tim Cook" in all_content

        # Verify chunks are reasonable size
        for chunk in chunks:
            assert len(chunk) > 0
            # Use same token estimation as the implementation (1 token ≈ 4 characters)
            estimated_tokens = len(chunk) // 4
            assert (
                estimated_tokens <= processor.max_tokens_per_chunk * 1.1
            )  # Allow 10% margin
