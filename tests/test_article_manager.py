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
from unittest import mock

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
            'title': 'Test Article',
            'raw_content': '<p>This is a test article with <b>HTML</b> tags.</p> Click here to read more.',
            'url': 'https://example.com/test-article',
            'published_at': '2025-05-18T12:00:00+00:00',
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
        
        # Get pending articles
        pending_articles = self.article_manager.get_articles_by_processing_status(status='PENDING')
        self.assertEqual(len(pending_articles), 1)
        
        # The URL hash should be generated from the URL
        expected_url_hash = self.article_manager.generate_url_hash(self.test_article['url'])
        self.assertEqual(pending_articles[0]['url_hash'], expected_url_hash)
        
        # Test get_article_by_hash
        article = self.article_manager.get_article_by_hash(expected_url_hash)
        self.assertIsNotNone(article)
        self.assertEqual(article['url'], self.test_article['url'])
        self.assertIn('added_at', article)  # Should have an added_at timestamp
    
    def test_store_article_replace_existing(self):
        """Test storing an article with replace_existing flag."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Get the URL hash
        url_hash = ArticleManager.generate_url_hash(self.test_article['url'])
        
        # Modify the article
        modified_article = self.test_article.copy()
        modified_article['title'] = 'Modified Title'
        
        # Store again with replace_existing=False (should not update)
        self.article_manager.store_articles([modified_article], replace_existing=False)
        
        # Get the article and verify title wasn't changed
        article = self.article_manager.get_article_by_hash(url_hash)
        self.assertEqual(article['title'], self.test_article['title'])
        
        # Store again with replace_existing=True (should update)
        self.article_manager.store_articles([modified_article], replace_existing=True)
        
        # Get the article and verify title was changed
        article = self.article_manager.get_article_by_hash(url_hash)
        self.assertEqual(article['title'], 'Modified Title')
    
    def test_update_article_processing_status(self):
        """Test updating an article's processing status."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Get the URL hash
        url_hash = ArticleManager.generate_url_hash(self.test_article['url'])
        
        # Update processing status
        self.article_manager.update_article_processing_status(
            url_hash,
            processed_content='Processed content',
            status='SUCCESS'
        )
        
        # Get the article and verify processed content
        article = self.article_manager.get_article_by_hash(url_hash)
        self.assertEqual(article['status_text_processing'], 'SUCCESS')
        self.assertEqual(article['processed_content'], 'Processed content')
    
    def test_update_article_embedding_status(self):
        """Test updating an article's embedding status."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Get the URL hash
        url_hash = ArticleManager.generate_url_hash(self.test_article['url'])
        
        # Update embedding status
        self.article_manager.update_article_embedding_status(
            url_hash,
            status='SUCCESS',
            embedding_model='test-model',
            vector_db_id='test-id-123'
        )
        
        # Get the article and verify embedding status
        article = self.article_manager.get_article_by_hash(url_hash)
        self.assertEqual(article['status_embedding'], 'SUCCESS')
    
    def test_get_processed_articles_for_embedding(self):
        """Test getting processed articles ready for embedding."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Get the URL hash
        url_hash = ArticleManager.generate_url_hash(self.test_article['url'])
        
        # Update processing status to SUCCESS
        self.article_manager.update_article_processing_status(
            url_hash,
            processed_content='Processed content',
            status='SUCCESS'
        )
        
        # Get articles for embedding
        articles = self.article_manager.get_processed_articles_for_embedding()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['url_hash'], url_hash)
        self.assertEqual(articles[0]['processed_content'], 'Processed content')
    
    def test_log_api_call(self):
        """Test logging an API call."""
        # Create mock fetched articles
        mock_articles = [
            {
                "title": "Test Article 1", 
                "published_at": "2025-05-10T12:00:00Z",
                "url": "https://example.com/article1"
            },
            {
                "title": "Test Article 2", 
                "published_at": "2025-05-18T12:00:00Z",
                "url": "https://example.com/article2"
            }
        ]
        
        # Log an API call with mock fetched articles
        log_id = self.article_manager.log_api_call(
            query_type='tag',
            query_value='EARNINGS',
            from_date='2025-05-01',
            to_date='2025-05-19',
            limit=10,
            articles_retrieved_count=2,
            fetched_articles=mock_articles,
            api_call_successful=True,
            http_status_code=200
        )
        
        # Verify log ID
        self.assertGreater(log_id, 0)
        
        # Verify log in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query_type, query_value, oldest_article_date_in_batch, newest_article_date_in_batch 
            FROM api_call_log WHERE log_id = ?
        """, (log_id,))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 'tag')
        self.assertEqual(row[1], 'EARNINGS')
        self.assertEqual(row[2], '2025-05-10T12:00:00Z')  # oldest date
        self.assertEqual(row[3], '2025-05-18T12:00:00Z')  # newest date
    
    def test_log_api_call_empty_articles(self):
        """Test logging an API call with empty article list."""
        # Log an API call with empty fetched articles
        log_id = self.article_manager.log_api_call(
            query_type='tag',
            query_value='EARNINGS',
            from_date='2025-05-01',
            to_date='2025-05-19',
            limit=10,
            articles_retrieved_count=0,
            fetched_articles=[],
            api_call_successful=True,
            http_status_code=200
        )
        
        # Verify log ID
        self.assertGreater(log_id, 0)
        
        # Verify log in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT oldest_article_date_in_batch, newest_article_date_in_batch 
            FROM api_call_log WHERE log_id = ?
        """, (log_id,))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertIsNone(row[0])  # oldest date should be None
        self.assertIsNone(row[1])  # newest date should be None
    
    def test_log_api_call_missing_dates(self):
        """Test logging an API call with articles missing dates."""
        # Create mock fetched articles with missing dates
        mock_articles = [
            {
                "title": "Test Article 1",
                "url": "https://example.com/article1"
                # missing published_at
            },
            {
                "title": "Test Article 2",
                "published_at": "",  # empty date
                "url": "https://example.com/article2"
            },
            {
                "title": "Test Article 3",
                "published_at": None,  # None date
                "url": "https://example.com/article3"
            }
        ]
        
        # Log an API call with mock fetched articles
        log_id = self.article_manager.log_api_call(
            query_type='tag',
            query_value='EARNINGS',
            articles_retrieved_count=3,
            fetched_articles=mock_articles,
            api_call_successful=True
        )
        
        # Verify log ID
        self.assertGreater(log_id, 0)
        
        # Verify log in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT oldest_article_date_in_batch, newest_article_date_in_batch 
            FROM api_call_log WHERE log_id = ?
        """, (log_id,))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertIsNone(row[0])  # oldest date should be None
        self.assertIsNone(row[1])  # newest date should be None
    
    def test_url_hash_generation(self):
        """Test the URL hash generation."""
        # Test with a normal URL
        url = 'https://example.com/test-article'
        url_hash = ArticleManager.generate_url_hash(url)
        
        # Verify hash is generated correctly
        self.assertEqual(len(url_hash), 64)  # SHA-256 produces 64 hex characters
        
        # Verify consistent hash for same URL
        url_hash2 = ArticleManager.generate_url_hash(url)
        self.assertEqual(url_hash, url_hash2)
        
        # Test with empty URL
        empty_url_hash = ArticleManager.generate_url_hash('')
        self.assertEqual(empty_url_hash, '')
        
        # Test with None URL
        none_url_hash = ArticleManager.generate_url_hash(None)
        self.assertEqual(none_url_hash, '')
    
    def test_get_articles_by_processing_status(self):
        """Test getting articles by their processing status."""
        # Store multiple test articles with different statuses
        article1 = self.test_article.copy()
        article1['url'] = 'https://example.com/article1'
        
        article2 = self.test_article.copy()
        article2['url'] = 'https://example.com/article2'
        
        article3 = self.test_article.copy()
        article3['url'] = 'https://example.com/article3'
        
        # Store all articles
        self.article_manager.store_articles([article1, article2, article3])
        
        # Get URL hashes
        hash1 = ArticleManager.generate_url_hash(article1['url'])
        hash2 = ArticleManager.generate_url_hash(article2['url'])
        hash3 = ArticleManager.generate_url_hash(article3['url'])
        
        # Update processing status for articles
        self.article_manager.update_article_processing_status(
            hash1,
            processed_content='Processed content 1',
            status='SUCCESS'
        )
        
        self.article_manager.update_article_processing_status(
            hash2,
            processed_content='',
            status='FAILED'
        )
        
        # Article3 remains with default 'PENDING' status
        
        # Test getting articles with 'SUCCESS' status
        success_articles = self.article_manager.get_articles_by_processing_status(status='SUCCESS')
        self.assertEqual(len(success_articles), 1)
        self.assertEqual(success_articles[0]['url_hash'], hash1)
        self.assertEqual(success_articles[0]['status_text_processing'], 'SUCCESS')
        
        # Test getting articles with 'FAILED' status
        failed_articles = self.article_manager.get_articles_by_processing_status(status='FAILED')
        self.assertEqual(len(failed_articles), 1)
        self.assertEqual(failed_articles[0]['url_hash'], hash2)
        self.assertEqual(failed_articles[0]['status_text_processing'], 'FAILED')
        
        # Test getting articles with 'PENDING' status
        pending_articles = self.article_manager.get_articles_by_processing_status(status='PENDING')
        self.assertEqual(len(pending_articles), 1)
        self.assertEqual(pending_articles[0]['url_hash'], hash3)
        self.assertEqual(pending_articles[0]['status_text_processing'], 'PENDING')
        
        # Test limit parameter
        all_articles = self.article_manager.get_articles_by_processing_status(status='PENDING', limit=1)
        self.assertEqual(len(all_articles), 1)
        
        # Test structure of returned articles - ensure all fields are present
        article = success_articles[0]
        self.assertIn('url_hash', article)
        self.assertIn('title', article)
        self.assertIn('raw_content', article)
        self.assertIn('processed_content', article)
        self.assertIn('url', article)
        self.assertIn('published_at', article)
        self.assertIn('added_at', article)  # Check for added_at instead of fetched_at
        self.assertIn('status_text_processing', article)
        self.assertIn('status_embedding', article)
        self.assertIn('symbols', article)
        self.assertIn('tags', article)
        self.assertIn('sentiment', article)
        
        # Test JSON fields are properly parsed
        self.assertIsInstance(article['symbols'], list)
        self.assertIsInstance(article['tags'], list)
        self.assertIsInstance(article['sentiment'], dict)
    
    def test_get_database_statistics(self):
        """Test the get_database_statistics method."""
        # Prepare test data with different statuses
        article1 = self.test_article.copy()
        article1['url'] = 'https://example.com/article1'
        article1['source_query_tag'] = 'EARNINGS'
        article1['source_query_symbol'] = 'AAPL.US'
        
        article2 = self.test_article.copy()
        article2['url'] = 'https://example.com/article2'
        article2['source_query_tag'] = 'TECHNOLOGY'
        article2['source_query_symbol'] = 'MSFT.US'
        
        article3 = self.test_article.copy()
        article3['url'] = 'https://example.com/article3'
        article3['source_query_tag'] = 'EARNINGS'
        article3['source_query_symbol'] = 'GOOGL.US'
        
        # Store all articles
        self.article_manager.store_articles([article1, article2, article3])
        
        # Get URL hashes
        hash1 = ArticleManager.generate_url_hash(article1['url'])
        hash2 = ArticleManager.generate_url_hash(article2['url'])
        
        # Update processing status for articles
        self.article_manager.update_article_processing_status(
            hash1,
            processed_content='Processed content 1',
            status='SUCCESS'
        )
        
        self.article_manager.update_article_processing_status(
            hash2,
            processed_content='',
            status='FAILED'
        )
        
        # Update embedding status for articles
        self.article_manager.update_article_embedding_status(
            hash1,
            status='SUCCESS',
            embedding_model='test-model'
        )
        
        # Log test API calls
        self.article_manager.log_api_call(
            query_type='tag',
            query_value='EARNINGS',
            articles_retrieved_count=2,
            api_call_successful=True
        )
        
        self.article_manager.log_api_call(
            query_type='symbol',
            query_value='AAPL.US',
            articles_retrieved_count=1,
            api_call_successful=True
        )
        
        # Get database statistics
        stats = self.article_manager.get_database_statistics()
        
        # Verify statistics
        self.assertEqual(stats['total_articles'], 3)
        
        # Check text processing status counts
        self.assertEqual(stats['text_processing_status'].get('SUCCESS', 0), 1)
        self.assertEqual(stats['text_processing_status'].get('FAILED', 0), 1)
        self.assertEqual(stats['text_processing_status'].get('PENDING', 0), 1)
        
        # Check embedding status counts
        self.assertEqual(stats['embedding_status'].get('SUCCESS', 0), 1)
        self.assertEqual(stats['embedding_status'].get('PENDING', 0), 2)
        
        # Check tag counts
        self.assertEqual(stats['articles_by_tag'].get('EARNINGS', 0), 2)
        self.assertEqual(stats['articles_by_tag'].get('TECHNOLOGY', 0), 1)
        
        # Check symbol counts
        self.assertEqual(stats['articles_by_symbol'].get('AAPL.US', 0), 1)
        self.assertEqual(stats['articles_by_symbol'].get('MSFT.US', 0), 1)
        self.assertEqual(stats['articles_by_symbol'].get('GOOGL.US', 0), 1)
        
        # Check API call stats
        self.assertEqual(stats['api_calls']['total_calls'], 2)
        self.assertEqual(stats['api_calls']['total_articles_retrieved'], 3)
        
        # Check date range
        self.assertIsNotNone(stats['date_range']['oldest_article'])
        self.assertIsNotNone(stats['date_range']['newest_article'])
        
    def test_get_database_statistics_empty_db(self):
        """Test get_database_statistics on an empty database."""
        # Get statistics for empty database
        stats = self.article_manager.get_database_statistics()
        
        # Verify statistics
        self.assertEqual(stats['total_articles'], 0)
        self.assertEqual(stats['text_processing_status'], {})
        self.assertEqual(stats['embedding_status'], {})
        self.assertEqual(stats['articles_by_tag'], {})
        self.assertEqual(stats['articles_by_symbol'], {})
        self.assertEqual(stats['api_calls']['total_calls'], 0)
        self.assertEqual(stats['api_calls']['total_articles_retrieved'], 0)
    
    def test_get_database_statistics_handles_error(self):
        """Test that get_database_statistics handles database errors gracefully."""
        # Save the original connection
        original_conn = self.article_manager.conn
        
        try:
            # Create a mock connection that raises an error when cursor is called
            mock_conn = mock.MagicMock()
            mock_conn.cursor.side_effect = sqlite3.Error("Simulated database error")
            
            # Replace the connection
            self.article_manager.conn = mock_conn
            
            # Call the method that should catch the error
            result = self.article_manager.get_database_statistics()
            
            # Verify error handling
            self.assertIn('error', result)
            self.assertEqual(result.get('status'), 'FAILED')
        finally:
            # Restore the original connection
            self.article_manager.conn = original_conn
            self.assertIn('error', result)
            self.assertIn('Simulated database error', result['error'])
    
    def test_delete_article_by_hash_successful(self):
        """Test successful deletion of an article by its URL hash."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Get the URL hash
        url_hash = ArticleManager.generate_url_hash(self.test_article['url'])
        
        # Verify article exists before deletion
        article = self.article_manager.get_article_by_hash(url_hash)
        self.assertIsNotNone(article)
        
        # Delete the article
        result = self.article_manager.delete_article_by_hash(url_hash)
        
        # Assert deletion was successful
        self.assertTrue(result)
        
        # Verify article no longer exists
        article = self.article_manager.get_article_by_hash(url_hash)
        self.assertIsNone(article)
    
    def test_delete_article_by_hash_nonexistent(self):
        """Test deletion of a non-existent article."""
        # Generate a hash for a URL that doesn't exist in the database
        nonexistent_hash = ArticleManager.generate_url_hash("https://example.com/nonexistent")
        
        # Try to delete the non-existent article
        result = self.article_manager.delete_article_by_hash(nonexistent_hash)
        
        # Assert deletion failed (returns False)
        self.assertFalse(result)
    
    def test_delete_article_by_hash_with_error(self):
        """Test deletion with database error."""
        # Store the test article
        self.article_manager.store_articles([self.test_article])
        
        # Get the URL hash
        url_hash = ArticleManager.generate_url_hash(self.test_article['url'])
        
        # Mock the database connection to simulate an error
        with mock.patch.object(self.article_manager, 'conn') as mock_conn:
            mock_conn.cursor.side_effect = sqlite3.Error("Simulated database error")
            
            # Try to delete the article with a simulated error
            result = self.article_manager.delete_article_by_hash(url_hash)
            
            # Assert deletion failed due to the error
            self.assertFalse(result)
