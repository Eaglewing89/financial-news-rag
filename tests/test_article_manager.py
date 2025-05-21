"""
Tests for the article manager module.

These tests validate the functionality of the ArticleManager class,
including database initialization, article storage, and status updates.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest

from financial_news_rag.article_manager import ArticleManager


class TestArticleManager(unittest.TestCase):
    """Test the ArticleManager class functions."""
    
    def setUp(self):
        """Set up a test database and article manager instance."""
        # Create a temporary directory for test database
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test_financial_news.db')
        
        # Create article manager with test database
        self.article_manager = ArticleManager(db_path=self.db_path)
        
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
        self.article_manager.close_connection()
        
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
        result = self.article_manager.store_articles([self.test_article])
        self.assertEqual(result, 1)
        
        # Check if article exists
        exists = self.article_manager.article_exists(self.test_article['url_hash'])
        self.assertTrue(exists)
        
        # Get pending articles
        pending_articles = self.article_manager.get_pending_articles()
        self.assertEqual(len(pending_articles), 1)
        self.assertEqual(pending_articles[0]['url_hash'], self.test_article['url_hash'])
        
        # Test get_article_by_hash
        article = self.article_manager.get_article_by_hash(self.test_article['url_hash'])
        self.assertIsNotNone(article)
        self.assertEqual(article['url'], self.test_article['url'])
    
    def test_store_article_replace_existing(self):
        """Test storing an article with replace_existing flag."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Modify the article
        modified_article = self.test_article.copy()
        modified_article['title'] = 'Modified Title'
        
        # Store again with replace_existing=False (should not update)
        self.article_manager.store_articles([modified_article], replace_existing=False)
        
        # Get the article and verify title wasn't changed
        article = self.article_manager.get_article_by_hash(self.test_article['url_hash'])
        self.assertEqual(article['title'], self.test_article['title'])
        
        # Store again with replace_existing=True (should update)
        self.article_manager.store_articles([modified_article], replace_existing=True)
        
        # Get the article and verify title was changed
        article = self.article_manager.get_article_by_hash(self.test_article['url_hash'])
        self.assertEqual(article['title'], 'Modified Title')
    
    def test_update_article_processing_status(self):
        """Test updating an article's processing status."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Update processing status
        self.article_manager.update_article_processing_status(
            self.test_article['url_hash'],
            processed_content='Processed content',
            status='SUCCESS'
        )
        
        # Get status and verify
        status = self.article_manager.get_article_status(self.test_article['url_hash'])
        self.assertEqual(status['status_text_processing'], 'SUCCESS')
        
        # Get the article and verify processed content
        article = self.article_manager.get_article_by_hash(self.test_article['url_hash'])
        self.assertEqual(article['processed_content'], 'Processed content')
    
    def test_update_article_embedding_status(self):
        """Test updating an article's embedding status."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Update embedding status
        self.article_manager.update_article_embedding_status(
            self.test_article['url_hash'],
            status='SUCCESS',
            embedding_model='test-model',
            vector_db_id='test-id-123'
        )
        
        # Get status and verify
        status = self.article_manager.get_article_status(self.test_article['url_hash'])
        self.assertEqual(status['status_embedding'], 'SUCCESS')
        self.assertEqual(status['embedding_model'], 'test-model')
        self.assertEqual(status['vector_db_id'], 'test-id-123')
    
    def test_get_processed_articles_for_embedding(self):
        """Test getting processed articles ready for embedding."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Update processing status to SUCCESS
        self.article_manager.update_article_processing_status(
            self.test_article['url_hash'],
            processed_content='Processed content',
            status='SUCCESS'
        )
        
        # Get articles for embedding
        articles = self.article_manager.get_processed_articles_for_embedding()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['url_hash'], self.test_article['url_hash'])
        self.assertEqual(articles[0]['processed_content'], 'Processed content')
    
    def test_log_api_call(self):
        """Test logging an API call."""
        # Log an API call
        log_id = self.article_manager.log_api_call(
            query_type='tag',
            query_value='EARNINGS',
            from_date='2025-05-01',
            to_date='2025-05-19',
            limit=10,
            articles_retrieved_count=5,
            oldest_article_date='2025-05-10',
            newest_article_date='2025-05-18',
            api_call_successful=True,
            http_status_code=200
        )
        
        # Verify log ID
        self.assertGreater(log_id, 0)
        
        # Verify log in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query_type, query_value FROM api_call_log WHERE log_id = ?", (log_id,))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 'tag')
        self.assertEqual(row[1], 'EARNINGS')
    
    def test_url_hash_generation(self):
        """Test the URL hash generation."""
        url = 'https://example.com/test-article'
        url_hash = ArticleManager.generate_url_hash(url)
        
        # Verify hash is generated correctly
        self.assertEqual(len(url_hash), 64)  # SHA-256 produces 64 hex characters
        
        # Verify consistent hash for same URL
        url_hash2 = ArticleManager.generate_url_hash(url)
        self.assertEqual(url_hash, url_hash2)


if __name__ == '__main__':
    unittest.main()
