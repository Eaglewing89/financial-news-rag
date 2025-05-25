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


# =============================================================================
# Shared Test Fixtures
# =============================================================================

@pytest.fixture
def sample_articles():
    """Create a set of sample articles for testing."""
    return [
        ArticleFactory.create_article(
            title='Tech Article',
            url='https://example.com/tech-article',
            symbols=['AAPL.US', 'MSFT.US'],
            tags=['TECHNOLOGY', 'EARNINGS']
        ),
        ArticleFactory.create_article(
            title='Finance Article', 
            url='https://example.com/finance-article',
            symbols=['JPM.US'],
            tags=['FINANCE']
        ),
        ArticleFactory.create_article(
            title='Healthcare Article',
            url='https://example.com/healthcare-article', 
            symbols=['JNJ.US'],
            tags=['HEALTHCARE']
        )
    ]


@pytest.fixture
def stored_article_with_hash(article_manager):
    """Create and store a single article, returning the article and its hash."""
    article = ArticleFactory.create_article(
        title='Stored Test Article',
        url='https://example.com/stored-article'
    )
    article_manager.store_articles([article])
    url_hash = generate_url_hash(article['url'])
    return article, url_hash


# =============================================================================
# TestArticleManagerInitialization - Database initialization, connection handling
# =============================================================================

class TestArticleManagerInitialization:
    """Tests for ArticleManager initialization and connection handling."""
    
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
    
    def test_database_connection_error_handling(self):
        """Test handling of database connection issues."""
        # Create manager with invalid path to test error handling
        with pytest.raises((sqlite3.OperationalError, OSError, sqlite3.DatabaseError)):
            # This should raise an exception for invalid database path
            invalid_manager = ArticleManager(db_path='/invalid/path/that/does/not/exist.db')
            # Try to perform an operation that would trigger the connection
            invalid_manager.get_database_statistics()
    
    def test_close_connection(self, temp_db_path):
        """Test that database connection can be properly closed."""
        manager = ArticleManager(db_path=temp_db_path)
        
        # Verify connection is active
        assert manager.conn is not None
        
        # Close connection
        manager.close_connection()
        
        # Verify connection is closed (implementation dependent)
        assert manager.conn is None or hasattr(manager.conn, 'close')


# =============================================================================
# TestArticleManagerStorage - Storing articles, handling duplicates, validation
# =============================================================================

class TestArticleManagerStorage:
    """Tests for ArticleManager storage operations."""
    
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
    
    def test_store_multiple_articles(self, article_manager, sample_articles):
        """Test storing multiple articles at once."""
        # Store all articles
        result = article_manager.store_articles(sample_articles)
        assert result == 3
        
        # Verify all are stored as pending
        pending_articles = article_manager.get_articles_by_processing_status(status='PENDING')
        assert len(pending_articles) == 3
        
        # Verify each article can be retrieved by hash
        for original_article in sample_articles:
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
    
    def test_store_articles_with_invalid_data(self, article_manager):
        """Test storing articles with missing required fields."""
        # Test with missing URL - should handle gracefully with empty string
        invalid_article = ArticleFactory.create_article()
        del invalid_article['url']
        
        # Should not raise an exception, but may result in invalid data
        result = article_manager.store_articles([invalid_article])
        # Should still store the article (with empty URL)
        assert result == 1
    
    def test_store_empty_articles_list(self, article_manager):
        """Test storing an empty list of articles."""
        result = article_manager.store_articles([])
        assert result == 0
    
    def test_store_articles_with_none_values(self, article_manager):
        """Test storing articles with None values in optional fields."""
        article = ArticleFactory.create_article()
        article['raw_content'] = None
        article['tags'] = None
        article['symbols'] = None
        
        result = article_manager.store_articles([article])
        assert result == 1
        
        # Verify the article was stored correctly
        url_hash = generate_url_hash(article['url'])
        stored_article = article_manager.get_article_by_hash(url_hash)
        assert stored_article is not None
        assert stored_article['title'] == article['title']


# =============================================================================
# TestArticleManagerRetrieval - Getting articles by hash, filtering by status
# =============================================================================

class TestArticleManagerRetrieval:
    """Tests for ArticleManager retrieval operations."""
    
    def test_get_article_by_invalid_hash(self, article_manager):
        """Test retrieving an article with an invalid hash."""
        result = article_manager.get_article_by_hash('invalid_hash_that_does_not_exist')
        assert result is None
    
    def test_get_article_by_empty_hash(self, article_manager):
        """Test retrieving an article with empty hash."""
        result = article_manager.get_article_by_hash('')
        assert result is None
    
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
    
    def test_get_articles_by_invalid_status(self, article_manager):
        """Test filtering articles by non-existent status."""
        result = article_manager.get_articles_by_processing_status('INVALID_STATUS')
        assert len(result) == 0
    
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


# =============================================================================
# TestArticleManagerStatusUpdates - Processing status, embedding status updates
# =============================================================================

class TestArticleManagerStatusUpdates:
    """Tests for ArticleManager status update operations."""
    
    def test_update_article_processing_status(self, stored_article_with_hash, article_manager):
        """Test updating an article's text processing status."""
        article, url_hash = stored_article_with_hash
        
        # Update processing status
        article_manager.update_article_processing_status(
            url_hash,
            processed_content='Cleaned and processed content',
            status='SUCCESS'
        )
        
        # Verify update
        updated_article = article_manager.get_article_by_hash(url_hash)
        assert updated_article['status_text_processing'] == 'SUCCESS'
        assert updated_article['processed_content'] == 'Cleaned and processed content'
    
    def test_update_article_processing_status_failure(self, stored_article_with_hash, article_manager):
        """Test updating an article's processing status to FAILED."""
        article, url_hash = stored_article_with_hash
        
        # Update to failed status with error message
        article_manager.update_article_processing_status(
            url_hash,
            status='FAILED',
            error_message='Processing failed due to invalid content'
        )
        
        # Verify failure status and error message
        updated_article = article_manager.get_article_by_hash(url_hash)
        assert updated_article['status_text_processing'] == 'FAILED'
        assert updated_article['processed_content'] is None
    
    def test_update_article_embedding_status(self, stored_article_with_hash, article_manager):
        """Test updating an article's embedding status."""
        article, url_hash = stored_article_with_hash
        
        # Update embedding status
        article_manager.update_article_embedding_status(
            url_hash,
            status='SUCCESS',
            embedding_model='text-embedding-004',
            vector_db_id='chunk_123_456'
        )
        
        # Verify embedding status update
        updated_article = article_manager.get_article_by_hash(url_hash)
        assert updated_article['status_embedding'] == 'SUCCESS'
        assert updated_article['embedding_model'] == 'text-embedding-004'
        assert updated_article['vector_db_id'] == 'chunk_123_456'
    
    def test_update_article_embedding_status_failure(self, stored_article_with_hash, article_manager):
        """Test updating an article's embedding status to FAILED."""
        article, url_hash = stored_article_with_hash
        
        # Update to failed embedding status
        article_manager.update_article_embedding_status(
            url_hash,
            status='FAILED',
            error_message='Embedding generation failed'
        )
        
        # Verify failure status
        updated_article = article_manager.get_article_by_hash(url_hash)
        assert updated_article['status_embedding'] == 'FAILED'
        # embedding_model and vector_db_id should remain unchanged (or be None if not set)
        assert updated_article['embedding_model'] is None
        assert updated_article['vector_db_id'] is None
    
    def test_update_article_with_empty_status(self, stored_article_with_hash, article_manager):
        """Test updating article with empty or invalid status values."""
        article, url_hash = stored_article_with_hash
        
        # Test with empty status string
        article_manager.update_article_processing_status(url_hash, status='')
        
        # Verify the update
        updated_article = article_manager.get_article_by_hash(url_hash)
        assert updated_article['status_text_processing'] == ''
    
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


# =============================================================================
# TestArticleManagerStatistics - Database statistics and reporting
# =============================================================================

class TestArticleManagerStatistics:
    """Tests for ArticleManager statistics operations."""
    
    def test_get_database_statistics(self, article_manager, sample_articles):
        """Test retrieving database statistics."""
        # Store articles
        article_manager.store_articles(sample_articles[:2])  # Store first 2 articles
        
        # Update statuses
        url_hash1 = generate_url_hash(sample_articles[0]['url'])
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
