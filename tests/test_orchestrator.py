"""
Tests for the FinancialNewsRAG orchestrator class.

This module contains unit tests for the FinancialNewsRAG class which integrates
all components of the financial-news-rag system.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

import pytest

# Import the class to test
from financial_news_rag.orchestrator import FinancialNewsRAG


class TestFinancialNewsRAG:
    """Test suite for the FinancialNewsRAG orchestrator class."""
    
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Set up test fixtures."""
        # Mock environment variables
        monkeypatch.setenv("EODHD_API_KEY", "test_eodhd_api_key")
        monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_api_key")
        
        # Create patches for all dependencies
        self.eodhd_patch = patch('financial_news_rag.orchestrator.EODHDClient')
        self.article_manager_patch = patch('financial_news_rag.orchestrator.ArticleManager')
        self.text_processor_patch = patch('financial_news_rag.orchestrator.TextProcessor')
        self.embeddings_generator_patch = patch('financial_news_rag.orchestrator.EmbeddingsGenerator')
        self.chroma_manager_patch = patch('financial_news_rag.orchestrator.ChromaDBManager')
        self.reranker_patch = patch('financial_news_rag.orchestrator.ReRanker')
        
        # Start patches and create mocks
        self.mock_eodhd = self.eodhd_patch.start()
        self.mock_article_manager = self.article_manager_patch.start()
        self.mock_text_processor = self.text_processor_patch.start()
        self.mock_embeddings_generator = self.embeddings_generator_patch.start()
        self.mock_chroma_manager = self.chroma_manager_patch.start()
        self.mock_reranker = self.reranker_patch.start()
        
        # Configure the embedding dimension mock
        self.mock_embeddings_instance = self.mock_embeddings_generator.return_value
        self.mock_embeddings_instance.embedding_dim = 768
        self.mock_embeddings_instance.model_name = "text-embedding-004"
        
        # Create the orchestrator instance
        self.orchestrator = FinancialNewsRAG()
        
        yield
        
        # Stop all patches
        self.eodhd_patch.stop()
        self.article_manager_patch.stop()
        self.text_processor_patch.stop()
        self.embeddings_generator_patch.stop()
        self.chroma_manager_patch.stop()
        self.reranker_patch.stop()
    
    def test_init(self):
        """Test the constructor initializes all components correctly."""
        # Check that all clients were initialized
        self.mock_eodhd.assert_called_once()
        self.mock_article_manager.assert_called_once()
        self.mock_text_processor.assert_called_once()
        self.mock_embeddings_generator.assert_called_once()
        self.mock_chroma_manager.assert_called_once()
        self.mock_reranker.assert_called_once()
    
    def test_fetch_and_store_articles_with_tag(self):
        """Test fetching and storing articles by tag."""
        # Mock articles return value with the source_query_tag and raw_content already included
        mock_articles = [
            {
                "title": "Test Article 1", 
                "published_at": "2023-01-01T12:00:00Z",
                "raw_content": "Test content 1",
                "source_query_tag": "TECHNOLOGY"
            },
            {
                "title": "Test Article 2", 
                "published_at": "2023-01-02T12:00:00Z",
                "raw_content": "Test content 2",
                "source_query_tag": "TECHNOLOGY"
            }
        ]
        
        # Configure mocks
        self.orchestrator.eodhd_client.fetch_news.return_value = mock_articles
        self.orchestrator.article_manager.store_articles.return_value = 2
        
        # Call the method with a tag
        result = self.orchestrator.fetch_and_store_articles(tag="TECHNOLOGY")
        
        # Assertions
        self.orchestrator.eodhd_client.fetch_news.assert_called_once_with(
            tag="TECHNOLOGY", 
            from_date=None, 
            to_date=None, 
            limit=50
        )
        
        # Check that articles were properly passed to store_articles
        self.orchestrator.article_manager.store_articles.assert_called_once()
        stored_articles = self.orchestrator.article_manager.store_articles.call_args[0][0]
        
        # Verify source_query_tag and raw_content came from EODHDClient
        assert len(stored_articles) == 2
        assert stored_articles[0].get("source_query_tag") == "TECHNOLOGY"
        assert "raw_content" in stored_articles[0]
        
        # Check that API call was logged
        self.orchestrator.article_manager.log_api_call.assert_called_once()
        log_args = self.orchestrator.article_manager.log_api_call.call_args[1]
        assert log_args["query_type"] == "tag"
        assert log_args["query_value"] == "TECHNOLOGY"
        assert log_args["fetched_articles"] == mock_articles
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_fetched"] == 2
        assert result["articles_stored"] == 2
    
    def test_fetch_and_store_articles_with_symbol(self):
        """Test fetching and storing articles by symbol."""
        # Mock articles return value with the source_query_symbol and raw_content already included
        mock_articles = [
            {
                "title": "Test Article 1", 
                "published_at": "2023-01-01T12:00:00Z",
                "raw_content": "Test content 1",
                "source_query_symbol": "AAPL"
            },
            {
                "title": "Test Article 2", 
                "published_at": "2023-01-02T12:00:00Z",
                "raw_content": "Test content 2",
                "source_query_symbol": "AAPL"
            }
        ]
        
        # Configure mocks
        self.orchestrator.eodhd_client.fetch_news.return_value = mock_articles
        self.orchestrator.article_manager.store_articles.return_value = 2
        
        # Call the method with a symbol
        result = self.orchestrator.fetch_and_store_articles(symbol="AAPL")
        
        # Assertions
        self.orchestrator.eodhd_client.fetch_news.assert_called_once_with(
            symbols="AAPL", 
            from_date=None, 
            to_date=None, 
            limit=50
        )
        
        # Check that articles were properly passed to store_articles
        self.orchestrator.article_manager.store_articles.assert_called_once()
        stored_articles = self.orchestrator.article_manager.store_articles.call_args[0][0]
        
        # Verify source_query_symbol and raw_content came from EODHDClient
        assert len(stored_articles) == 2
        assert stored_articles[0].get("source_query_symbol") == "AAPL"
        assert "raw_content" in stored_articles[0]
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_fetched"] == 2
        assert result["articles_stored"] == 2
    
    def test_fetch_and_store_articles_with_multiple_symbols(self):
        """Test fetching and storing articles with multiple symbols."""
        # Mock articles return value for each symbol
        mock_articles_aapl = [
            {
                "title": "AAPL Article", 
                "published_at": "2023-01-01T12:00:00Z",
                "raw_content": "AAPL content",
                "source_query_symbol": "AAPL"
            }
        ]
        mock_articles_msft = [
            {
                "title": "MSFT Article", 
                "published_at": "2023-01-02T12:00:00Z",
                "raw_content": "MSFT content",
                "source_query_symbol": "MSFT"
            }
        ]
        
        # Configure mock to return different values on each call
        self.orchestrator.eodhd_client.fetch_news.side_effect = [mock_articles_aapl, mock_articles_msft]
        self.orchestrator.article_manager.store_articles.return_value = 1
        
        # Call the method with comma-separated symbols
        result = self.orchestrator.fetch_and_store_articles(symbol="AAPL,MSFT")
        
        # Assertions
        assert self.orchestrator.eodhd_client.fetch_news.call_count == 2
        # First call with AAPL
        assert self.orchestrator.eodhd_client.fetch_news.call_args_list[0][1]["symbols"] == "AAPL"
        # Second call with MSFT
        assert self.orchestrator.eodhd_client.fetch_news.call_args_list[1][1]["symbols"] == "MSFT"
        
        # Check that store_articles was called twice
        assert self.orchestrator.article_manager.store_articles.call_count == 2
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_fetched"] == 2
        assert result["articles_stored"] == 2
    
    def test_process_articles_by_status_pending(self):
        """Test processing articles with pending status."""
        # Mock pending articles
        mock_articles = [
            {"url_hash": "hash1", "raw_content": "Test content 1"},
            {"url_hash": "hash2", "raw_content": "Test content 2"},
            {"url_hash": "hash3", "raw_content": ""}  # Empty content to test failure case
        ]
        
        # Configure mocks
        self.orchestrator.article_manager.get_articles_by_processing_status.return_value = mock_articles
        
        # Mock the process_and_validate_content method
        self.orchestrator.text_processor.process_and_validate_content.side_effect = [
            {"status": "SUCCESS", "reason": "", "content": "Processed content 1"},
            {"status": "SUCCESS", "reason": "", "content": "Processed content 2"},
            {"status": "FAILED", "reason": "Empty raw content", "content": ""}
        ]
        
        # Call the method with default status='PENDING'
        result = self.orchestrator.process_articles_by_status()
        
        # Assertions
        self.orchestrator.article_manager.get_articles_by_processing_status.assert_called_once_with(status='PENDING', limit=100)
        assert self.orchestrator.text_processor.process_and_validate_content.call_count == 3
        
        # Check updates on articles
        update_calls = self.orchestrator.article_manager.update_article_processing_status.call_args_list
        
        # First article should be successful
        assert update_calls[0][0][0] == "hash1"  # First positional arg is url_hash
        assert update_calls[0][1]["status"] == "SUCCESS"
        assert update_calls[0][1]["processed_content"] == "Processed content 1"
        
        # Second article should be successful
        assert update_calls[1][0][0] == "hash2"  # First positional arg is url_hash
        assert update_calls[1][1]["status"] == "SUCCESS"
        assert update_calls[1][1]["processed_content"] == "Processed content 2"
        
        # Third article should fail due to empty content
        assert update_calls[2][0][0] == "hash3"  # First positional arg is url_hash
        assert update_calls[2][1]["status"] == "FAILED"
        assert update_calls[2][1]["error_message"] == "Empty raw content"
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_processed"] == 2
        assert result["articles_failed"] == 1
    
    def test_process_articles_by_status_failed(self):
        """Test processing articles with failed status."""
        # Mock failed articles
        mock_articles = [
            {"url_hash": "hash1", "raw_content": "Test content 1"},
            {"url_hash": "hash2", "raw_content": ""}  # Empty content to test failure case
        ]
        
        # Configure mocks
        self.orchestrator.article_manager.get_articles_by_processing_status.return_value = mock_articles
        
        # Mock the process_and_validate_content method
        self.orchestrator.text_processor.process_and_validate_content.side_effect = [
            {"status": "SUCCESS", "reason": "", "content": "Processed content 1"},
            {"status": "FAILED", "reason": "Empty raw content", "content": ""}
        ]
        
        # Call the method with status='FAILED'
        result = self.orchestrator.process_articles_by_status(status='FAILED')
        
        # Assertions
        self.orchestrator.article_manager.get_articles_by_processing_status.assert_called_once_with(
            status='FAILED', limit=100
        )
        assert self.orchestrator.text_processor.process_and_validate_content.call_count == 2
        
        # Check updates on articles
        update_calls = self.orchestrator.article_manager.update_article_processing_status.call_args_list
        
        # First article should be successful
        assert update_calls[0][0][0] == "hash1"  # First positional arg is url_hash
        assert update_calls[0][1]["status"] == "SUCCESS"
        assert update_calls[0][1]["processed_content"] == "Processed content 1"
        
        # Second article should fail due to empty content
        assert update_calls[1][0][0] == "hash2"  # First positional arg is url_hash
        assert update_calls[1][1]["status"] == "FAILED"
        assert update_calls[1][1]["error_message"] == "Empty raw content"
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_processed"] == 1
        assert result["articles_failed"] == 1
    
    def test_embed_processed_articles(self):
        """Test embedding processed articles for both PENDING and FAILED statuses."""
        # Create mock processed articles
        mock_articles_pending = [
            {
                "url_hash": "hash_pending_1", 
                "processed_content": "Processed content pending 1",
                "published_at": "2023-01-01T12:00:00Z",
                "source_query_tag": "TECHNOLOGY"
            },
            {
                "url_hash": "hash_pending_2", 
                "processed_content": "Processed content pending 2",
                "published_at": "2023-01-02T12:00:00Z",
                "source_query_symbol": "AAPL"
            }
        ]
        mock_articles_failed = [
            {
                "url_hash": "hash_failed_1", 
                "processed_content": "Processed content failed 1",
                "published_at": "2023-01-03T12:00:00Z"
            }
        ]
        
        # Mock chunks
        mock_chunks_pending_1 = ["Chunk P1.1", "Chunk P1.2"]
        mock_chunks_pending_2 = ["Chunk P2.1", "Chunk P2.2"]
        mock_chunks_failed_1 = ["Chunk F1.1", "Chunk F1.2"]
        
        # Mock embeddings
        mock_embeddings_pending_1 = [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings_pending_2 = [[0.5, 0.6], [0.7, 0.8]]
        mock_embeddings_failed_1 = [[0.9, 1.0], [1.1, 1.2]]
        
        self.mock_embeddings_instance.embedding_dim = 2  # For simplicity

        # --- Test for PENDING status ---
        self.orchestrator.article_manager.reset_mock()
        self.orchestrator.text_processor.reset_mock()
        self.orchestrator.embeddings_generator.reset_mock() # Reset the main mock for the class
        self.mock_embeddings_instance.reset_mock() # Reset the instance mock too, for good measure
        self.orchestrator.chroma_manager.reset_mock()
        self.orchestrator.article_manager.update_article_embedding_status.reset_mock()

        # Configure mocks for PENDING
        self.orchestrator.article_manager.get_processed_articles_for_embedding.return_value = mock_articles_pending
        self.orchestrator.text_processor.split_into_chunks.side_effect = [mock_chunks_pending_1, mock_chunks_pending_2]
        # Mock generate_and_verify_embeddings on the actual EmbeddingsGenerator instance used by the orchestrator
        self.mock_embeddings_instance.generate_and_verify_embeddings.side_effect = [
            {"embeddings": mock_embeddings_pending_1, "all_valid": True},
            {"embeddings": mock_embeddings_pending_2, "all_valid": True}
        ]
        self.orchestrator.chroma_manager.add_article_chunks.return_value = True
        
        # Call the method for PENDING status
        result_pending = self.orchestrator.embed_processed_articles(status='PENDING')
        
        # Assertions for PENDING
        self.orchestrator.article_manager.get_processed_articles_for_embedding.assert_called_once_with(status='PENDING', limit=100)
        assert self.orchestrator.text_processor.split_into_chunks.call_count == 2
        assert self.mock_embeddings_instance.generate_and_verify_embeddings.call_count == 2
        assert self.orchestrator.chroma_manager.add_article_chunks.call_count == 2
        self.orchestrator.chroma_manager.delete_embeddings_by_article.assert_not_called() # Should not be called for PENDING
        
        # Check that add_article_chunks was called with the correct arguments
        add_chunks_calls = self.orchestrator.chroma_manager.add_article_chunks.call_args_list
        
        # First article
        first_call_args = add_chunks_calls[0][0]  # Positional arguments of first call
        first_call_kwargs = add_chunks_calls[0][1]  # Keyword arguments of first call
        assert first_call_args[0] == "hash_pending_1"  # article_url_hash
        assert first_call_args[1] == mock_chunks_pending_1  # chunk_texts
        assert first_call_args[2] == mock_embeddings_pending_1  # chunk_vectors
        assert first_call_args[3] == {  # article_data
            'published_at': '2023-01-01T12:00:00Z',
            'source_query_tag': 'TECHNOLOGY',
            'source_query_symbol': None
        }
        
        # Second article
        second_call_args = add_chunks_calls[1][0]  # Positional arguments of second call
        second_call_kwargs = add_chunks_calls[1][1]  # Keyword arguments of second call
        assert second_call_args[0] == "hash_pending_2"  # article_url_hash
        assert second_call_args[1] == mock_chunks_pending_2  # chunk_texts
        assert second_call_args[2] == mock_embeddings_pending_2  # chunk_vectors
        assert second_call_args[3] == {  # article_data
            'published_at': '2023-01-02T12:00:00Z',
            'source_query_tag': None,
            'source_query_symbol': 'AAPL'
        }
        
        # Check that embedding status was updated for PENDING
        update_calls_pending = self.orchestrator.article_manager.update_article_embedding_status.call_args_list
        assert len(update_calls_pending) == 2
        assert update_calls_pending[0][1]["url_hash"] == "hash_pending_1"
        assert update_calls_pending[0][1]["status"] == "SUCCESS"
        assert update_calls_pending[0][1]["embedding_model"] == self.mock_embeddings_instance.model_name
        assert update_calls_pending[1][1]["url_hash"] == "hash_pending_2"
        assert update_calls_pending[1][1]["status"] == "SUCCESS"
        
        # Check result for PENDING
        assert result_pending["status"] == "SUCCESS"
        assert result_pending["articles_embedding_succeeded"] == 2
        assert result_pending["articles_failed"] == 0
    
        # --- Test for FAILED status ---
        # Reset all top-level mocks and the instance mock.
        # This resets their direct call counts and their children's call counts.
        self.orchestrator.article_manager.reset_mock()
        self.orchestrator.text_processor.reset_mock()
        self.orchestrator.embeddings_generator.reset_mock() # Class mock
        self.mock_embeddings_instance.reset_mock()         # Instance mock (which is orchestrator.embeddings_generator)
        self.orchestrator.chroma_manager.reset_mock()
        # Explicitly reset call count for update_article_embedding_status for focused assertions
        self.orchestrator.article_manager.update_article_embedding_status.reset_mock()

        # Explicitly clear side_effect and return_value from methods that used them,
        # then reconfigure them for the FAILED test phase.

        # For text_processor.split_into_chunks
        split_chunks_mock = self.orchestrator.text_processor.split_into_chunks
        split_chunks_mock.reset_mock(return_value=True, side_effect=True) # Clear prior side_effect
        
        # For embeddings_generator.generate_and_verify_embeddings
        gve_mock = self.mock_embeddings_instance.generate_and_verify_embeddings
        gve_mock.reset_mock(return_value=True, side_effect=True)          # Clear prior side_effect
    
        # Configure mocks for FAILED
        self.orchestrator.article_manager.get_processed_articles_for_embedding.return_value = mock_articles_failed
        
        split_chunks_mock.return_value = mock_chunks_failed_1             # Set new return_value for split_chunks
        
        gve_mock.return_value = {                                         # Set new return_value for gve_mock
            "embeddings": mock_embeddings_failed_1, "all_valid": True
        }
        self.orchestrator.chroma_manager.delete_embeddings_by_article.return_value = True
        self.orchestrator.chroma_manager.add_article_chunks.return_value = True
    
        # Call the method for FAILED status
        result_failed = self.orchestrator.embed_processed_articles(status='FAILED')
    
        # Assertions for FAILED
        self.orchestrator.article_manager.get_processed_articles_for_embedding.assert_called_once_with(status='FAILED', limit=100)
        split_chunks_mock.assert_called_once_with("Processed content failed 1")
        gve_mock.assert_called_once_with(mock_chunks_failed_1)
    
        self.orchestrator.chroma_manager.delete_embeddings_by_article.assert_called_once_with("hash_failed_1")
        self.orchestrator.chroma_manager.add_article_chunks.assert_called_once()
        
        # Check that add_article_chunks was called with the correct arguments
        add_chunks_call_args = self.orchestrator.chroma_manager.add_article_chunks.call_args[0]  # Positional arguments
        assert add_chunks_call_args[0] == "hash_failed_1"  # article_url_hash
        assert add_chunks_call_args[1] == mock_chunks_failed_1  # chunk_texts
        assert add_chunks_call_args[2] == mock_embeddings_failed_1  # chunk_vectors
        assert add_chunks_call_args[3] == {  # article_data
            'published_at': '2023-01-03T12:00:00Z',
            'source_query_tag': None,
            'source_query_symbol': None
        }

        # Check that embedding status was updated for FAILED
        update_calls_failed = self.orchestrator.article_manager.update_article_embedding_status.call_args_list
        assert len(update_calls_failed) == 1
        assert update_calls_failed[0][1]["url_hash"] == "hash_failed_1"
        assert update_calls_failed[0][1]["status"] == "SUCCESS"
        assert update_calls_failed[0][1]["embedding_model"] == self.mock_embeddings_instance.model_name

        # Check result for FAILED
        assert result_failed["status"] == "SUCCESS"
        assert result_failed["articles_embedding_succeeded"] == 1
        assert result_failed["articles_failed"] == 0
    
    def test_get_article_database_status(self):
        """Test getting article database statistics."""
        # Mock the get_database_statistics method in article_manager
        mock_stats = {
            "total_articles": 100,
            "text_processing_status": {
                "PENDING": 50,
                "SUCCESS": 40,
                "FAILED": 10
            },
            "embedding_status": {
                "PENDING": 60,
                "SUCCESS": 30,
                "FAILED": 10
            },
            "articles_by_tag": {
                "TECHNOLOGY": 30,
                "FINANCE": 20
            },
            "articles_by_symbol": {
                "AAPL": 25,
                "MSFT": 15
            },
            "date_range": {
                "oldest_article": "2023-01-01",
                "newest_article": "2023-01-31"
            },
            "api_calls": {
                "total_calls": 5,
                "total_articles_retrieved": 60
            }
        }
        
        self.orchestrator.article_manager.get_database_statistics = MagicMock(return_value=mock_stats)
        
        # Call the method
        result = self.orchestrator.get_article_database_status()
        
        # Assertions
        assert self.orchestrator.article_manager.get_database_statistics.call_count == 1
        
        # Check result structure
        assert result["total_articles"] == 100
        assert "text_processing_status" in result
        assert result["text_processing_status"]["PENDING"] == 50
        assert result["text_processing_status"]["SUCCESS"] == 40
        assert result["text_processing_status"]["FAILED"] == 10
        assert "embedding_status" in result
        assert "articles_by_tag" in result
        assert "articles_by_symbol" in result
        assert "date_range" in result
        assert "api_calls" in result
    
    def test_get_vector_database_status(self):
        """Test getting vector database statistics."""
        # Configure mock
        self.orchestrator.chroma_manager.get_collection_status.return_value = {
            "collection_name": "financial_news_embeddings",
            "total_chunks": 500,
            "unique_articles": 100,
            "persist_directory": "/path/to/chroma"
        }
        
        # Call the method
        result = self.orchestrator.get_vector_database_status()
        
        # Assertions
        self.orchestrator.chroma_manager.get_collection_status.assert_called_once()
        
        # Check result
        assert result["collection_name"] == "financial_news_embeddings"
        assert result["total_chunks"] == 500
        assert result["unique_articles"] == 100
    
    def test_search_articles(self):
        """Test searching for articles."""
        # Mock query embedding
        mock_query_embedding = [0.1, 0.2, 0.3]
        
        # Mock ChromaDB results
        mock_chroma_results = [
            {
                "chunk_id": "hash1_0",
                "text": "Chunk text 1",
                "distance": 0.1,
                "metadata": {"article_url_hash": "hash1", "chunk_index": 0}
            },
            {
                "chunk_id": "hash1_1",
                "text": "Chunk text 2",
                "distance": 0.2,
                "metadata": {"article_url_hash": "hash1", "chunk_index": 1}
            },
            {
                "chunk_id": "hash2_0",
                "text": "Chunk text 3",
                "distance": 0.3,
                "metadata": {"article_url_hash": "hash2", "chunk_index": 0}
            }
        ]
        
        # Mock articles
        mock_article1 = {
            "url_hash": "hash1",
            "title": "Article 1",
            "processed_content": "Processed content 1",
            "url": "http://example.com/1"
        }
        
        mock_article2 = {
            "url_hash": "hash2",
            "title": "Article 2",
            "processed_content": "Processed content 2",
            "url": "http://example.com/2"
        }
        
        # Configure mocks
        self.orchestrator.embeddings_generator.generate_embeddings.return_value = [mock_query_embedding]
        self.orchestrator.chroma_manager.query_embeddings.return_value = mock_chroma_results
        self.orchestrator.article_manager.get_article_by_hash.side_effect = [mock_article1, mock_article2]
        
        # Call the method without reranking
        result = self.orchestrator.search_articles("test query", n_results=2, rerank=False)
        
        # Assertions
        self.orchestrator.embeddings_generator.generate_embeddings.assert_called_once_with(["test query"])
        self.orchestrator.chroma_manager.query_embeddings.assert_called_once()
        assert self.orchestrator.article_manager.get_article_by_hash.call_count == 2
        
        # Check result (should be ordered by similarity)
        assert len(result) == 2
        assert result[0]["url_hash"] == "hash1"  # hash1 has better similarity (lower distance)
        assert result[1]["url_hash"] == "hash2"
        assert "similarity_score" in result[0]
        assert "similarity_score" in result[1]
        assert result[0]["similarity_score"] > result[1]["similarity_score"]
        
        # Test with reranking
        self.orchestrator.embeddings_generator.generate_embeddings.reset_mock()
        self.orchestrator.chroma_manager.query_embeddings.reset_mock()
        self.orchestrator.article_manager.get_article_by_hash.reset_mock()
        
        # Mock reranker
        mock_reranked_articles = [
            {
                "url_hash": "hash2",
                "title": "Article 2",
                "processed_content": "Processed content 2",
                "url": "http://example.com/2",
                "similarity_score": 0.85,
                "rerank_score": 0.9
            },
            {
                "url_hash": "hash1",
                "title": "Article 1",
                "processed_content": "Processed content 1",
                "url": "http://example.com/1",
                "similarity_score": 0.95,
                "rerank_score": 0.8
            }
        ]
        
        self.orchestrator.embeddings_generator.generate_embeddings.return_value = [mock_query_embedding]
        self.orchestrator.chroma_manager.query_embeddings.return_value = mock_chroma_results
        self.orchestrator.article_manager.get_article_by_hash.side_effect = [mock_article1, mock_article2]
        self.orchestrator.reranker.rerank_articles.return_value = mock_reranked_articles
        
        # Call the method with reranking
        result = self.orchestrator.search_articles("test query", n_results=2, rerank=True)
        
        # Assertions
        self.orchestrator.reranker.rerank_articles.assert_called_once()
        
        # Check result (should be ordered by rerank score)
        assert len(result) == 2
        assert result[0]["url_hash"] == "hash2"  # Reranked order
        assert result[1]["url_hash"] == "hash1"
        assert "rerank_score" in result[0]
    
    def test_delete_article_data(self):
        """Test deleting an article and its embeddings."""
        # Configure mocks
        self.orchestrator.chroma_manager.delete_embeddings_by_article.return_value = True
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        self.orchestrator.article_manager.conn.cursor.return_value = mock_cursor
        
        # Call the method
        result = self.orchestrator.delete_article_data("test_hash")
        
        # Assertions
        self.orchestrator.chroma_manager.delete_embeddings_by_article.assert_called_once_with("test_hash")
        mock_cursor.execute.assert_called_once()
        self.orchestrator.article_manager.conn.commit.assert_called_once()
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["article_deleted"] is True
        assert result["embeddings_deleted"] is True
    
    def test_close(self):
        """Test closing database connections."""
        # Call the method
        self.orchestrator.close()
        
        # Assertions
        self.orchestrator.article_manager.close_connection.assert_called_once()
