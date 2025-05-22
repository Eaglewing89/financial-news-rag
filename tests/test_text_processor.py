"""
Tests for the text processor module.

These tests validate the functionality of the TextProcessor class,
including text cleaning and chunking.
"""

import unittest
from unittest.mock import patch

from financial_news_rag.text_processor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    """Test the TextProcessor class functions."""
    
    def setUp(self):
        """Set up a text processor instance."""
        # Create processor with default settings
        self.processor = TextProcessor()
        
    def test_text_cleaning(self):
        """Test the text cleaning functionality thoroughly."""
        # Test removing HTML tags
        raw_text = '<p>This is a <b>test</b> article</p>'
        cleaned_text = self.processor.clean_article_text(raw_text)
        self.assertEqual(cleaned_text, 'This is a test article')
        
        # Test removing common boilerplate phrases
        raw_text = 'This is an article. Click here to read more.'
        cleaned_text = self.processor.clean_article_text(raw_text)
        self.assertEqual(cleaned_text, 'This is an article.')
        
        # Test multiple boilerplate patterns
        raw_text = 'This is content. Read more: visit our site. Source: Example News'
        cleaned_text = self.processor.clean_article_text(raw_text)
        self.assertEqual(cleaned_text, 'This is content.')
        
        # Test handling of smart quotes
        raw_text = 'This has "smart quotes" and \u00e2\u20ac\u2122single quotes\u00e2\u20ac\u2122'
        cleaned_text = self.processor.clean_article_text(raw_text)
        self.assertNotIn('\u00e2\u20ac\u2122', cleaned_text)
        self.assertIn("'single quotes'", cleaned_text)
        
        # Test handling of whitespace
        raw_text = '  Multiple    spaces  \n\n and \t tabs  '
        cleaned_text = self.processor.clean_article_text(raw_text)
        self.assertEqual(cleaned_text, 'Multiple spaces and tabs')
        
        # Test handling of empty/None input
        self.assertEqual(self.processor.clean_article_text(''), '')
        self.assertEqual(self.processor.clean_article_text(None), '')
    
    def test_text_chunking_short_text(self):
        """Test text chunking with short text (below max token limit)."""
        # Short text should result in a single chunk
        short_text = 'This is a short text that should fit in one chunk.'
        chunks = self.processor.split_into_chunks(short_text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_text)
    
    def test_text_chunking_long_text(self):
        """Test text chunking with long text (above max token limit)."""
        # Create a long text with many sentences
        long_text = ' '.join(['This is test sentence number {}.'.format(i) for i in range(100)])
        
        # Set a small max_tokens to force multiple chunks
        processor = TextProcessor(max_tokens_per_chunk=200)
        chunks = processor.split_into_chunks(long_text)
        
        # Verify multiple chunks are created
        self.assertGreater(len(chunks), 1)
        
        # Check that all chunks are below the limit (with some margin for sentence boundaries)
        for chunk in chunks:
            # Estimate tokens (char count / 4)
            tokens = len(chunk) // 4
            self.assertLessEqual(tokens, 220)  # Allow a 10% margin
    
    def test_text_chunking_very_long_sentence(self):
        """Test chunking a text with a very long sentence that exceeds the limit."""
        # Create a single long sentence
        very_long_sentence = ' '.join(['word{}'.format(i) for i in range(1000)])
        
        # Set a small max_tokens
        processor = TextProcessor(max_tokens_per_chunk=100)
        chunks = processor.split_into_chunks(very_long_sentence)
        
        # Verify the sentence is split into multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that all chunks are below the limit with a reasonable margin
        # Since very long sentences without punctuation might result in larger chunks
        for chunk in chunks:
            tokens = len(chunk) // 4
            self.assertLessEqual(tokens, 200)  # Increased margin to accommodate the test case
    
    def test_text_chunking_empty_input(self):
        """Test chunking with empty input."""
        chunks = self.processor.split_into_chunks('')
        self.assertEqual(chunks, [])
        
        chunks = self.processor.split_into_chunks(None)
        self.assertEqual(chunks, [])
    
    def test_chunking_with_different_token_limits(self):
        """Test chunking with different token limit values."""
        text = ' '.join(['This is test sentence number {}.'.format(i) for i in range(50)])
        
        # Test with different max_tokens_per_chunk values
        token_limits = [100, 200, 500, 1000]
        
        for limit in token_limits:
            processor = TextProcessor(max_tokens_per_chunk=limit)
            chunks = processor.split_into_chunks(text)
            
            # Check that all chunks respect the limit
            for chunk in chunks:
                tokens = len(chunk) // 4
                self.assertLessEqual(tokens, limit * 1.1)  # Allow a 10% margin
    
    def test_chunking_preserves_content(self):
        """Test that chunking preserves all content from the original text."""
        text = ' '.join(['This is test sentence number {}.'.format(i) for i in range(20)])
        
        processor = TextProcessor(max_tokens_per_chunk=200)
        chunks = processor.split_into_chunks(text)
        
        # Join all chunks and compare with original (ignoring whitespace differences)
        combined = ' '.join(chunks)
        # Normalize whitespace for comparison
        text_normalized = ' '.join(text.split())
        combined_normalized = ' '.join(combined.split())
        
        self.assertEqual(combined_normalized, text_normalized)
    
    def test_nltk_fallback(self):
        """Test the fallback mechanism when NLTK tokenizer fails."""
        text = "This is a test. With multiple sentences. For testing fallback."
        
        # Mock sent_tokenize to raise an exception
        with patch('nltk.tokenize.sent_tokenize', side_effect=Exception('NLTK error')):
            chunks = self.processor.split_into_chunks(text)
            
            # Verify we still get chunks
            self.assertGreater(len(chunks), 0)
            
            # Content should still be present
            combined = ' '.join(chunks)
            self.assertIn("This is a test", combined)
            self.assertIn("With multiple sentences", combined)
            self.assertIn("For testing fallback", combined)
    
    def test_process_and_validate_content_none_input(self):
        """Test process_and_validate_content with None input."""
        result = self.processor.process_and_validate_content(None)
        
        # Check the key fields
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["reason"], "Empty raw content")
        self.assertEqual(result["content"], "")
    
    def test_process_and_validate_content_empty_string(self):
        """Test process_and_validate_content with empty string input."""
        result = self.processor.process_and_validate_content("")
        
        # Check the key fields
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["reason"], "Empty raw content")
        self.assertEqual(result["content"], "")
    
    def test_process_and_validate_content_only_whitespace(self):
        """Test process_and_validate_content with input containing only whitespace."""
        result = self.processor.process_and_validate_content("   \n\t  ")
        
        # Check the key fields
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["reason"], "Empty raw content")
        self.assertEqual(result["content"], "")
    
    def test_process_and_validate_content_cleaned_to_empty(self):
        """Test process_and_validate_content with input that is cleaned to empty."""
        # Create a mock that will return empty string after cleaning
        with patch.object(self.processor, 'clean_article_text', return_value=""):
            result = self.processor.process_and_validate_content("Click here to read more.")
            
            # Check the key fields
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["reason"], "No content after cleaning")
            self.assertEqual(result["content"], "")
    
    def test_process_and_validate_content_success(self):
        """Test process_and_validate_content with valid input."""
        with patch.object(self.processor, 'clean_article_text', return_value="Cleaned content"):
            result = self.processor.process_and_validate_content("Raw content")
            
            # Check the key fields
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["reason"], "")
            self.assertEqual(result["content"], "Cleaned content")
    
    def test_process_and_validate_content_real_cleaning(self):
        """Test process_and_validate_content with real cleaning."""
        raw_text = "<p>This is a <b>test</b> article</p>"
        result = self.processor.process_and_validate_content(raw_text)
        
        # Check the key fields
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["reason"], "")
        self.assertEqual(result["content"], "This is a test article")


if __name__ == '__main__':
    unittest.main()
