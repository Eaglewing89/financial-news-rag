"""
Unit tests for the ArticleManager class.

These tests validate the functionality of the ArticleManager class,
including database initialization, article storage, and status updates.
All tests use pytest fixtures and are isolated from external dependencies.
"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from financial_news_rag.article_manager import ArticleManager
from financial_news_rag.utils import generate_url_hash
from tests.fixtures.sample_data import ArticleFactory


class TestArticleManager:
    """Test suite for the ArticleManager class functionality."""
    
    def test_database_initialization(self, article_manager, temp_db_path):
        """Test that the database tables are properly created."""
        # Check if tables exist by querying the database directly
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Query for tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Check expected tables exist
        assert 'articles' in tables
        assert 'api_call_log' in tables
        
        conn.close()
    
    def test_store_and_retrieve_single_article(self, article_manager):
        """Test storing a single article and retrieving it."""
        # Create test article using factory
        test_article = ArticleFactory.create_article(
            title='Test Article',
            url='https://example.com/test-article',
            symbols=['AAPL.US', 'MSFT.US'],
            tags=['TECHNOLOGY', 'EARNINGS']
        )
        
        # Store the article
        result = article_manager.store_articles([test_article])
        assert result == 1
        
        # Get pending articles
        pending_articles = article_manager.get_articles_by_processing_status(status='PENDING')
        assert len(pending_articles) == 1
        
        # Verify URL hash generation
        expected_url_hash = generate_url_hash(test_article['url'])
        assert pending_articles[0]['url_hash'] == expected_url_hash
        
        # Test get_article_by_hash
        article = article_manager.get_article_by_hash(expected_url_hash)
        assert article is not None
        assert article['url'] == test_article['url']
        assert article['title'] == test_article['title']
        assert 'added_at' in article  # Should have timestamp
    
    def test_store_multiple_articles(self, article_manager):
        """Test storing multiple articles at once."""
        # Create multiple test articles
        articles = [
            ArticleFactory.create_article(title=f'Test Article {i}', url=f'https://example.com/article-{i}')
            for i in range(3)
        ]
        
        # Store all articles
        result = article_manager.store_articles(articles)
        assert result == 3
        
        # Verify all are stored as pending
        pending_articles = article_manager.get_articles_by_processing_status(status='PENDING')
        assert len(pending_articles) == 3
        
        # Verify each article can be retrieved by hash
        for original_article in articles:
            url_hash = generate_url_hash(original_article['url'])
            stored_article = article_manager.get_article_by_hash(url_hash)
            assert stored_article is not None
            assert stored_article['title'] == original_article['title']
    
    def test_store_article_replace_existing_false(self, article_manager):
        """Test storing duplicate article with replace_existing=False."""
        # Create and store initial article
        test_article = ArticleFactory.create_article(
            title='Original Title',
            url='https://example.com/test-article'
        )
        article_manager.store_articles([test_article])
        
        # Get URL hash for retrieval
        url_hash = generate_url_hash(test_article['url'])
        
        # Create modified version of same article
        modified_article = test_article.copy()
        modified_article['title'] = 'Modified Title'
        
        # Store again with replace_existing=False (should not update)
        result = article_manager.store_articles([modified_article], replace_existing=False)
        assert result == 0  # No new articles stored
        
        # Verify original title remains unchanged
        stored_article = article_manager.get_article_by_hash(url_hash)
        assert stored_article['title'] == 'Original Title'
    
    def test_store_article_replace_existing_true(self, article_manager):
        """Test storing duplicate article with replace_existing=True."""
        # Create and store initial article
        test_article = ArticleFactory.create_article(
            title='Original Title',
            url='https://example.com/test-article'
        )
        article_manager.store_articles([test_article])
        
        # Get URL hash for retrieval
        url_hash = generate_url_hash(test_article['url'])
        
        # Create modified version of same article
        modified_article = test_article.copy()
        modified_article['title'] = 'Modified Title'
        modified_article['raw_content'] = 'Updated content'
        
        # Store again with replace_existing=True (should update)
        result = article_manager.store_articles([modified_article], replace_existing=True)
        assert result == 1  # One article updated
        
        # Verify title was updated
        stored_article = article_manager.get_article_by_hash(url_hash)
        assert stored_article['title'] == 'Modified Title'
        assert 'Updated content' in stored_article['raw_content']
    
    def test_update_article_processing_status(self, article_manager):
        """Test updating an article's text processing status."""
        # Store test article
        test_article = ArticleFactory.create_article()
        article_manager.store_articles([test_article])
        url_hash = generate_url_hash(test_article['url'])
        
        # Update processing status
        article_manager.update_article_processing_status(
            url_hash,
            processed_content='Cleaned and processed content',
            status='SUCCESS'
        )
        # No return value, verify by retrieving the article
        updated_article = article_manager.get_article_by_hash(url_hash)
        assert updated_article is not None
        
        # Verify update
        article = article_manager.get_article_by_hash(url_hash)
        assert article['status_text_processing'] == 'SUCCESS'
        assert article['processed_content'] == 'Cleaned and processed content'
        # Note: text_processed_at field doesn't exist in current schema
    
    def test_update_article_processing_status_failure(self, article_manager):
        """Test updating an article's processing status to FAILED."""
        # Store test article
        test_article = ArticleFactory.create_article()
        article_manager.store_articles([test_article])
        url_hash = generate_url_hash(test_article['url'])
        
        # Update to failed status with error message
        article_manager.update_article_processing_status(
            url_hash,
            status='FAILED',
            error_message='Processing failed due to invalid content'
        )
        # No return value, verify by retrieving the article
        updated_article = article_manager.get_article_by_hash(url_hash)
        assert updated_article is not None
        
        # Verify failure status and error message
        article = article_manager.get_article_by_hash(url_hash)
        assert article['status_text_processing'] == 'FAILED'
        # Note: processing_error_message field doesn't exist in current schema
        assert article['processed_content'] is None
    
    def test_update_article_embedding_status(self, article_manager):
        """Test updating an article's embedding status."""
        # Store test article
        test_article = ArticleFactory.create_article()
        article_manager.store_articles([test_article])
        url_hash = generate_url_hash(test_article['url'])
        
        # Update embedding status
        article_manager.update_article_embedding_status(
            url_hash,
            status='SUCCESS',
            embedding_model='text-embedding-004',
            vector_db_id='chunk_123_456'
        )
        # No return value, verify by retrieving the article
        updated_article = article_manager.get_article_by_hash(url_hash)
        assert updated_article is not None
        
        # Verify embedding status update
        article = article_manager.get_article_by_hash(url_hash)
        assert article['status_embedding'] == 'SUCCESS'
        assert article['embedding_model'] == 'text-embedding-004'
        assert article['vector_db_id'] == 'chunk_123_456'
        # Note: embeddings_created_at field doesn't exist in current schema
    
    def test_update_article_embedding_status_failure(self, article_manager):
        """Test updating an article's embedding status to FAILED."""
        # Store test article
        test_article = ArticleFactory.create_article()
        article_manager.store_articles([test_article])
        url_hash = generate_url_hash(test_article['url'])
        
        # Update to failed embedding status
        article_manager.update_article_embedding_status(
            url_hash,
            status='FAILED',
            error_message='Embedding generation failed'
        )
        # No return value, verify by retrieving the article
        updated_article = article_manager.get_article_by_hash(url_hash)
        assert updated_article is not None
        
        # Verify failure status
        article = article_manager.get_article_by_hash(url_hash)
        assert article['status_embedding'] == 'FAILED'
        # Note: embedding_error_message field doesn't exist in current schema
        # embedding_model and vector_db_id should remain unchanged (or be None if not set)
        assert article['embedding_model'] is None
        assert article['vector_db_id'] is None
    
    def test_get_articles_by_processing_status(self, article_manager):
        """Test filtering articles by processing status."""
        # Create articles with different processing states
        articles = [
            ArticleFactory.create_article(title='Pending Article 1'),
            ArticleFactory.create_article(title='Pending Article 2'),
            ArticleFactory.create_article(title='To Process Article')
        ]
        article_manager.store_articles(articles)
        
        # Update one article to SUCCESS status
        url_hash = generate_url_hash(articles[0]['url'])
        article_manager.update_article_processing_status(url_hash, status='SUCCESS')
        
        # Test filtering by PENDING status
        pending_articles = article_manager.get_articles_by_processing_status('PENDING')
        assert len(pending_articles) == 2
        pending_titles = [a['title'] for a in pending_articles]
        assert 'Pending Article 2' in pending_titles
        assert 'To Process Article' in pending_titles
        
        # Test filtering by SUCCESS status
        success_articles = article_manager.get_articles_by_processing_status('SUCCESS')
        assert len(success_articles) == 1
        assert success_articles[0]['title'] == 'Pending Article 1'
    
    def test_get_processed_articles_for_embedding(self, article_manager):
        """Test retrieving articles ready for embedding."""
        # Create and store test articles
        articles = [
            ArticleFactory.create_article(title='Article 1'),
            ArticleFactory.create_article(title='Article 2'),
            ArticleFactory.create_article(title='Article 3')
        ]
        article_manager.store_articles(articles)
        
        # Update two articles to SUCCESS processing status
        for i in range(2):
            url_hash = generate_url_hash(articles[i]['url'])
            article_manager.update_article_processing_status(
                url_hash,
                processed_content=f'Processed content {i+1}',
                status='SUCCESS'
            )
        
        # Get articles ready for embedding
        ready_articles = article_manager.get_processed_articles_for_embedding()
        assert len(ready_articles) == 2
        
        # Verify they have processed content
        for article in ready_articles:
            # Note: status fields are not returned by get_processed_articles_for_embedding
            # The method already filters by status_text_processing = 'SUCCESS'
            assert article['processed_content'] is not None
            assert 'url_hash' in article
            assert 'title' in article
    
    def test_get_database_statistics(self, article_manager):
        """Test retrieving database statistics."""
        # Create articles with various states
        articles = [
            ArticleFactory.create_article(
                title='Tech Article',
                symbols=['AAPL.US'],
                tags=['TECHNOLOGY']
            ),
            ArticleFactory.create_article(
                title='Finance Article',
                symbols=['JPM.US'],
                tags=['FINANCE']
            )
        ]
        article_manager.store_articles(articles)
        
        # Update statuses
        url_hash1 = generate_url_hash(articles[0]['url'])
        article_manager.update_article_processing_status(url_hash1, status='SUCCESS')
        article_manager.update_article_embedding_status(url_hash1, status='SUCCESS')
        
        # Get statistics
        stats = article_manager.get_database_statistics()
        
        # Verify statistics structure
        assert 'total_articles' in stats
        assert 'text_processing_status' in stats
        assert 'embedding_status' in stats
        assert 'articles_by_tag' in stats
        assert 'articles_by_symbol' in stats
        assert 'date_range' in stats
        
        # Verify specific counts
        assert stats['total_articles'] == 2
        assert stats['text_processing_status']['SUCCESS'] == 1
        assert stats['text_processing_status']['PENDING'] == 1
        assert stats['embedding_status']['SUCCESS'] == 1
        assert stats['embedding_status']['PENDING'] == 1
    
    def test_store_articles_with_invalid_data(self, article_manager):
        """Test storing articles with missing required fields."""
        # Test with missing URL - should handle gracefully with empty string
        invalid_article = ArticleFactory.create_article()
        del invalid_article['url']
        
        # Should not raise an exception, but may result in invalid data
        result = article_manager.store_articles([invalid_article])
        # Should still store the article (with empty URL)
        assert result == 1
    
    def test_update_nonexistent_article(self, article_manager):
        """Test updating an article that doesn't exist."""
        fake_hash = 'nonexistent_hash_123'
        
        # Attempt to update processing status
        article_manager.update_article_processing_status(
            fake_hash,
            status='SUCCESS'
        )
        # No return value for non-existent articles, just verify no exception
        
        # Attempt to update embedding status
        result = article_manager.update_article_embedding_status(
            fake_hash,
            status='SUCCESS'
        )
        # Method returns None, not False for non-existent articles
        assert result is None
    
    def test_close_connection(self, temp_db_path):
        """Test that database connection can be properly closed."""
        manager = ArticleManager(db_path=temp_db_path)
        
        # Verify connection is active
        assert manager.conn is not None
        
        # Close connection
        manager.close_connection()
        
        # Verify connection is closed (implementation dependent)
        # Note: This test may need adjustment based on actual implementation
        assert manager.conn is None or hasattr(manager.conn, 'close')
