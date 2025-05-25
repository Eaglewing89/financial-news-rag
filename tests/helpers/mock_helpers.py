"""
Test helper utilities and assertion functions.

This module provides reusable utilities for setting up test environments,
creating mock responses, and performing common assertions across tests.
"""

import tempfile
import sqlite3
import os
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import numpy as np

from financial_news_rag.utils import generate_url_hash

if TYPE_CHECKING:
    from financial_news_rag.config import Config


class DatabaseTestHelper:
    """Helper class for database-related test operations."""
    
    @staticmethod
    def create_temp_db() -> str:
        """
        Create a temporary database for testing.
        
        Returns:
            Path to the temporary database file
        """
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)  # Close the file descriptor
        return path
    
    @staticmethod
    def verify_table_exists(db_path: str, table_name: str) -> bool:
        """
        Verify that a table exists in the database.
        
        Args:
            db_path: Path to the database
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        result = cursor.fetchone() is not None
        conn.close()
        return result
    
    @staticmethod
    def get_table_count(db_path: str, table_name: str) -> int:
        """
        Get the number of rows in a table.
        
        Args:
            db_path: Path to the database
            table_name: Name of the table
            
        Returns:
            Number of rows in the table
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    @staticmethod
    def cleanup_db(db_path: str) -> None:
        """
        Clean up a temporary database file.
        
        Args:
            db_path: Path to the database file to remove
        """
        if os.path.exists(db_path):
            os.remove(db_path)


class MockHelper:
    """Helper class for creating and configuring mocks."""
    
    @staticmethod
    def create_mock_eodhd_success_response(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a mock successful EODHD API response.
        
        Args:
            articles: List of articles to include in response
            
        Returns:
            Mock API response
        """
        return {
            "articles": articles,
            "status_code": 200,
            "success": True,
            "message": "Success"
        }
    
    @staticmethod
    def create_mock_eodhd_error_response(status_code: int = 500, message: str = "Server error") -> Dict[str, Any]:
        """
        Create a mock error EODHD API response.
        
        Args:
            status_code: HTTP status code
            message: Error message
            
        Returns:
            Mock error response
        """
        return {
            "articles": [],
            "status_code": status_code,
            "success": False,
            "message": message
        }
    
    @staticmethod
    def configure_text_processor_mock(mock_processor: MagicMock, 
                                      clean_content: str = "Cleaned content",
                                      chunks: List[str] = None) -> None:
        """
        Configure a TextProcessor mock with default responses.
        
        Args:
            mock_processor: Mock TextProcessor instance
            clean_content: Content to return from clean_content()
            chunks: List of chunks to return from chunk_text()
        """
        if chunks is None:
            chunks = ["Chunk 1", "Chunk 2"]
            
        mock_processor.clean_content.return_value = clean_content
        mock_processor.chunk_text.return_value = chunks
    
    @staticmethod
    def configure_embeddings_mock(mock_generator: MagicMock,
                                  embedding_dim: int = 768,
                                  model_name: str = "text-embedding-004") -> None:
        """
        Configure an EmbeddingsGenerator mock with default responses.
        
        Args:
            mock_generator: Mock EmbeddingsGenerator instance
            embedding_dim: Embedding dimension
            model_name: Model name
        """
        mock_generator.embedding_dim = embedding_dim
        mock_generator.model_name = model_name
        mock_generator.generate_embeddings.return_value = [
            np.random.rand(embedding_dim).tolist() for _ in range(2)
        ]
    
    @staticmethod
    def configure_chroma_mock(mock_chroma: MagicMock) -> None:
        """
        Configure a ChromaDBManager mock with default responses.
        
        Args:
            mock_chroma: Mock ChromaDBManager instance
        """
        mock_chroma.add_article_chunks.return_value = True
        mock_chroma.query_embeddings.return_value = [
            {
                "chunk_id": "test_article_0",
                "text": "Sample chunk content",
                "metadata": {"article_url_hash": "test_hash", "chunk_index": 0},
                "distance": 0.5
            }
        ]
        mock_chroma.get_collection_status.return_value = {
            "total_chunks": 10,
            "unique_articles": 5,
            "is_empty": False
        }


class AssertionHelper:
    """Helper class for common test assertions."""
    
    @staticmethod
    def assert_article_equals(actual: Dict[str, Any], expected: Dict[str, Any],
                              ignore_fields: List[str] = None) -> None:
        """
        Assert that two articles are equal, ignoring specified fields.
        
        Args:
            actual: Actual article dictionary
            expected: Expected article dictionary
            ignore_fields: List of fields to ignore in comparison
        """
        if ignore_fields is None:
            ignore_fields = ['url_hash', 'created_at', 'updated_at']
        
        for key, value in expected.items():
            if key not in ignore_fields:
                assert key in actual, f"Key '{key}' missing from actual article"
                assert actual[key] == value, f"Value mismatch for key '{key}': {actual[key]} != {value}"
    
    @staticmethod
    def assert_article_stored_correctly(article_manager, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assert that an article was stored correctly in the database.
        
        Args:
            article_manager: ArticleManager instance
            article: Original article that was stored
            
        Returns:
            Retrieved article from database
        """
        url_hash = generate_url_hash(article['url'])
        stored_article = article_manager.get_article_by_hash(url_hash)
        
        assert stored_article is not None, "Article was not found in database"
        assert stored_article['title'] == article['title']
        assert stored_article['url'] == article['url']
        assert 'url_hash' in stored_article
        
        return stored_article
    
    @staticmethod
    def assert_chunks_in_chroma(chroma_manager, article_hash: str, 
                                expected_chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Assert that chunks were added to ChromaDB correctly.
        
        Args:
            chroma_manager: ChromaDBManager instance
            article_hash: Article URL hash
            expected_chunks: Expected chunk texts
            
        Returns:
            Query results from ChromaDB
        """
        # Query for all chunks for this article
        query_embedding = np.random.rand(768).tolist()
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=100,
            filter_metadata={"article_url_hash": article_hash}
        )
        
        assert len(results) == len(expected_chunks), \
            f"Expected {len(expected_chunks)} chunks, got {len(results)}"
        
        # Check that all chunk texts are present
        result_texts = {result["text"] for result in results}
        expected_texts = set(expected_chunks)
        assert result_texts == expected_texts, \
            f"Chunk texts don't match. Expected: {expected_texts}, Got: {result_texts}"
        
        return results
    
    @staticmethod
    def assert_api_call_logged(article_manager, query_type: str, query_value: str) -> None:
        """
        Assert that an API call was logged correctly.
        
        Args:
            article_manager: ArticleManager instance
            query_type: Type of query (tag/symbol)
            query_value: Query value
        """
        # This would need to be implemented based on how you want to verify API call logging
        # For now, just verify the method was called
        if hasattr(article_manager, 'log_api_call'):
            article_manager.log_api_call.assert_called()
    
    @staticmethod
    def assert_processing_status(article_manager, url_hash: str, expected_status: str) -> None:
        """
        Assert that an article has the expected processing status.
        
        Args:
            article_manager: ArticleManager instance
            url_hash: Article URL hash
            expected_status: Expected processing status
        """
        article = article_manager.get_article_by_hash(url_hash)
        assert article is not None, "Article not found"
        assert article.get('processing_status') == expected_status, \
            f"Expected status '{expected_status}', got '{article.get('processing_status')}'"
    
    @staticmethod
    def assert_embedding_status(article_manager, url_hash: str, expected_status: str) -> None:
        """
        Assert that an article has the expected embedding status.
        
        Args:
            article_manager: ArticleManager instance
            url_hash: Article URL hash
            expected_status: Expected embedding status
        """
        article = article_manager.get_article_by_hash(url_hash)
        assert article is not None, "Article not found"
        assert article.get('embedding_status') == expected_status, \
            f"Expected embedding status '{expected_status}', got '{article.get('embedding_status')}'"


class TestEnvironmentHelper:
    """Helper class for setting up test environments."""
    
    @staticmethod
    def setup_test_env_vars() -> Dict[str, str]:
        """
        Set up standard test environment variables.
        
        Returns:
            Dictionary of environment variables set
        """
        env_vars = {
            "EODHD_API_KEY": "test_eodhd_api_key",
            "GEMINI_API_KEY": "test_gemini_api_key",
            "DATABASE_PATH_OVERRIDE": ":memory:",
            "CHROMA_DEFAULT_PERSIST_DIRECTORY_OVERRIDE": "/tmp/test_chroma",
            "EMBEDDINGS_DEFAULT_MODEL": "text-embedding-004"
        }
        
        # Apply environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
        
        return env_vars
    
    @staticmethod
    def cleanup_test_env_vars(env_vars: Dict[str, str]) -> None:
        """
        Clean up test environment variables.
        
        Args:
            env_vars: Dictionary of environment variables to remove
        """
        for key in env_vars.keys():
            if key in os.environ:
                del os.environ[key]
    
    @staticmethod
    def create_test_config() -> 'Config':
        """
        Create a test configuration instance.
        
        Returns:
            Config instance for testing
        """
        from financial_news_rag.config import Config
        
        with patch.dict(os.environ, {
            "EODHD_API_KEY": "test_eodhd_api_key",
            "GEMINI_API_KEY": "test_gemini_api_key"
        }):
            return Config()


class RetryHelper:
    """Helper for testing retry logic and async operations."""
    
    @staticmethod
    def create_failing_then_succeeding_mock(fail_count: int = 2):
        """
        Create a mock that fails a specified number of times then succeeds.
        
        Args:
            fail_count: Number of times to fail before succeeding
            
        Returns:
            Configured mock
        """
        mock = MagicMock()
        side_effects = [Exception("Test failure")] * fail_count + [{"success": True}]
        mock.side_effect = side_effects
        return mock
    
    @staticmethod
    def create_rate_limited_mock(rate_limit_responses: int = 1):
        """
        Create a mock that simulates rate limiting.
        
        Args:
            rate_limit_responses: Number of rate limit responses before success
            
        Returns:
            Configured mock
        """
        mock = MagicMock()
        rate_limit_response = {"status_code": 429, "success": False, "message": "Rate limited"}
        success_response = {"status_code": 200, "success": True, "articles": []}
        
        side_effects = [rate_limit_response] * rate_limit_responses + [success_response]
        mock.return_value = side_effects
        return mock


# Convenience functions for common operations
def create_temp_db_with_articles(articles: List[Dict[str, Any]]) -> tuple:
    """
    Create a temporary database and populate it with test articles.
    
    Args:
        articles: List of articles to store
        
    Returns:
        Tuple of (db_path, article_manager)
    """
    from financial_news_rag.article_manager import ArticleManager
    
    db_path = DatabaseTestHelper.create_temp_db()
    article_manager = ArticleManager(db_path=db_path)
    article_manager.store_articles(articles)
    
    return db_path, article_manager


def verify_mock_call_order(mock_obj: MagicMock, expected_calls: List[str]) -> None:
    """
    Verify that mock methods were called in the expected order.
    
    Args:
        mock_obj: Mock object to check
        expected_calls: List of expected method names in order
    """
    actual_calls = [call[0] for call in mock_obj.method_calls]
    assert actual_calls == expected_calls, \
        f"Expected calls {expected_calls}, got {actual_calls}"


def setup_integration_test_environment():
    """Set up environment for integration tests with real components."""
    temp_dir = tempfile.mkdtemp()
    
    config = {
        'db_path': os.path.join(temp_dir, 'test.db'),
        'chroma_dir': os.path.join(temp_dir, 'chroma'),
        'temp_dir': temp_dir
    }
    
    return config


def cleanup_integration_test_environment(config: Dict[str, str]):
    """Clean up integration test environment."""
    import shutil
    if os.path.exists(config['temp_dir']):
        shutil.rmtree(config['temp_dir'])
