"""
Tests for the text processing pipeline module.

These tests validate the functionality of the TextProcessingPipeline class,
including text cleaning, chunking, and database interactions.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from financial_news_rag.text_processor import TextProcessingPipeline


class TestTextProcessingPipeline(unittest.TestCase):
    """Test the TextProcessingPipeline class functions."""
    
    def setUp(self):
        """Set up a test database and pipeline instance."""
        # Create a temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test_financial_news.db')
        
        # Create pipeline with test database
        self.pipeline = TextProcessingPipeline(db_path=self.db_path)
        
        # Sample test data
        self.test_article = {
            'url_hash': '1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
            'title': 'Test Article',
            'raw_content': '<p>This is a test article with <b>HTML</b> tags.</p> Click here to read more.',
            'url': 'https://example.com/test-article',
            'published_at': '2025-05-18T12:00:00+00:00',
            'fetched_at': '2025-05-19T09:00:00+00:00',
            'source_api': 'EODHD',
            'symbols': ['AAPL.US', 'MSFT.US'],
            'tags': ['TECHNOLOGY', 'EARNINGS'],
            'sentiment': {'polarity': 0.5, 'neg': 0.1, 'neu': 0.5, 'pos': 0.4}
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Close database connection
        self.pipeline.close_connection()
        
        # Remove temporary directory and database
        shutil.rmtree(self.test_dir)
    
    def test_database_initialization(self):
        """Test that the database tables are properly created."""
        # Check if tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query for tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Check expected tables
        self.assertIn('articles', tables)
        self.assertIn('api_call_log', tables)
        self.assertIn('api_errors_log', tables)
        
        conn.close()
    
    def test_store_and_retrieve_article(self):
        """Test storing an article and retrieving it."""
        # Store the test article
        result = self.pipeline.store_articles([self.test_article])
        self.assertEqual(result, 1)
        
        # Check if article exists
        exists = self.pipeline.article_exists(self.test_article['url_hash'])
        self.assertTrue(exists)
        
        # Get pending articles
        pending_articles = self.pipeline.get_pending_articles()
        self.assertEqual(len(pending_articles), 1)
        self.assertEqual(pending_articles[0]['url_hash'], self.test_article['url_hash'])
    
    def test_text_cleaning(self):
        """Test the text cleaning functionality."""
        raw_text = '<p>This is a test article with <b>HTML</b> tags.</p> Click here to read more.'
        cleaned_text = self.pipeline.clean_article_text(raw_text)
        
        # Check cleaning results
        self.assertNotIn('<p>', cleaned_text)
        self.assertNotIn('<b>', cleaned_text)
        self.assertNotIn('</p>', cleaned_text)
        self.assertNotIn('Click here to read more', cleaned_text)
        self.assertIn('This is a test article with HTML tags', cleaned_text)
    
    def test_text_chunking(self):
        """Test the text chunking functionality."""
        # Create a long text with sentences
        long_text = ' '.join(['This is test sentence number {}.'.format(i) for i in range(100)])
        
        # Set max_tokens to force multiple chunks
        pipeline = TextProcessingPipeline(db_path=self.db_path, max_tokens_per_chunk=50)
        chunks = pipeline.split_into_chunks(long_text)
        
        # Verify multiple chunks are created
        self.assertGreater(len(chunks), 1)
        
        # Verify each chunk is within token limit
        for chunk in chunks:
            # Estimate tokens (char count / 4)
            tokens = len(chunk) // 4
            self.assertLessEqual(tokens, 50)
    
    def test_process_article(self):
        """Test processing an article in the database."""
        # Store the article
        self.pipeline.store_articles([self.test_article])
        
        # Process pending articles
        processed, failed = self.pipeline.process_articles()
        self.assertEqual(processed, 1)
        self.assertEqual(failed, 0)
        
        # Check article status
        status = self.pipeline.get_article_status(self.test_article['url_hash'])
        self.assertEqual(status['status_text_processing'], 'SUCCESS')
        
        # Verify no more pending articles
        pending_articles = self.pipeline.get_pending_articles()
        self.assertEqual(len(pending_articles), 0)
    
    def test_article_chunks(self):
        """Test getting chunks for a processed article."""
        # Store and process the test article
        self.pipeline.store_articles([self.test_article])
        self.pipeline.process_articles()
        
        # Get chunks for the article
        chunks = self.pipeline.get_chunks_for_article(self.test_article['url_hash'])
        
        # Verify chunks exist and contain expected metadata
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['parent_url_hash'], self.test_article['url_hash'])
        self.assertEqual(chunks[0]['title'], self.test_article['title'])
        self.assertIn('text', chunks[0])
    
    def test_process_article_failure(self):
        """Test handling of article processing failure."""
        # Create an article with problematic content
        bad_article = self.test_article.copy()
        bad_article['url_hash'] = 'badarticle123'
        bad_article['raw_content'] = None  # This should cause a failure
        
        # Store the article
        self.pipeline.store_articles([bad_article])
        
        # Mock the clean_article_text method to raise an exception
        with patch.object(self.pipeline, 'clean_article_text', side_effect=Exception('Test error')):
            processed, failed = self.pipeline.process_articles()
            self.assertEqual(processed, 0)
            self.assertEqual(failed, 1)
        
        # Check article status
        status = self.pipeline.get_article_status(bad_article['url_hash'])
        self.assertEqual(status['status_text_processing'], 'FAILED')
    
    def test_url_hash_generation(self):
        """Test the URL hash generation."""
        url = 'https://example.com/test-article'
        url_hash = TextProcessingPipeline.generate_url_hash(url)
        
        # Verify hash is generated correctly
        self.assertEqual(len(url_hash), 64)  # SHA-256 produces 64 hex characters
        
        # Verify consistent hash for same URL
        url_hash2 = TextProcessingPipeline.generate_url_hash(url)
        self.assertEqual(url_hash, url_hash2)


if __name__ == '__main__':
    unittest.main()
