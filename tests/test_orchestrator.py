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
        # Mock articles return value
        mock_articles = [
            {"title": "Test Article 1", "published_at": "2023-01-01T12:00:00Z"},
            {"title": "Test Article 2", "published_at": "2023-01-02T12:00:00Z"}
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
        
        # Check that articles were properly processed before storing
        self.orchestrator.article_manager.store_articles.assert_called_once()
        stored_articles = self.orchestrator.article_manager.store_articles.call_args[0][0]
        
        # Verify source_query_tag and raw_content were added
        assert len(stored_articles) == 2
        assert stored_articles[0].get("source_query_tag") == "TECHNOLOGY"
        assert "raw_content" in stored_articles[0]
        
        # Check that API call was logged
        self.orchestrator.article_manager.log_api_call.assert_called_once()
        log_args = self.orchestrator.article_manager.log_api_call.call_args[1]
        assert log_args["query_type"] == "tag"
        assert log_args["query_value"] == "TECHNOLOGY"
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_fetched"] == 2
        assert result["articles_stored"] == 2
    
    def test_fetch_and_store_articles_with_symbol(self):
        """Test fetching and storing articles by symbol."""
        # Mock articles return value
        mock_articles = [
            {"title": "Test Article 1", "published_at": "2023-01-01T12:00:00Z"},
            {"title": "Test Article 2", "published_at": "2023-01-02T12:00:00Z"}
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
        
        # Check that articles were properly processed before storing
        self.orchestrator.article_manager.store_articles.assert_called_once()
        stored_articles = self.orchestrator.article_manager.store_articles.call_args[0][0]
        
        # Verify source_query_symbol and raw_content were added
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
        mock_articles_aapl = [{"title": "AAPL Article", "published_at": "2023-01-01T12:00:00Z"}]
        mock_articles_msft = [{"title": "MSFT Article", "published_at": "2023-01-02T12:00:00Z"}]
        
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
    
    def test_process_pending_articles(self):
        """Test processing pending articles."""
        # Mock pending articles
        mock_articles = [
            {"url_hash": "hash1", "raw_content": "Test content 1"},
            {"url_hash": "hash2", "raw_content": "Test content 2"},
            {"url_hash": "hash3", "raw_content": ""}  # Empty content to test failure case
        ]
        
        # Configure mocks
        self.orchestrator.article_manager.get_pending_articles.return_value = mock_articles
        self.orchestrator.text_processor.clean_article_text.side_effect = ["Processed content 1", "Processed content 2"]
        
        # Call the method
        result = self.orchestrator.process_pending_articles()
        
        # Assertions
        self.orchestrator.article_manager.get_pending_articles.assert_called_once()
        assert self.orchestrator.text_processor.clean_article_text.call_count == 2
        
        # Check updates on articles
        update_calls = self.orchestrator.article_manager.update_article_processing_status.call_args_list
        
        # First article should be successful
        assert update_calls[0][0][0] == "hash1"  # First positional arg is url_hash
        assert update_calls[0][1]["status"] == "SUCCESS"
        assert "processed_content" in update_calls[0][1]
        
        # Second article should be successful
        assert update_calls[1][0][0] == "hash2"  # First positional arg is url_hash
        assert update_calls[1][1]["status"] == "SUCCESS"
        
        # Third article should fail due to empty content
        assert update_calls[2][0][0] == "hash3"  # First positional arg is url_hash
        assert update_calls[2][1]["status"] == "FAILED"
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_processed"] == 2
        assert result["articles_failed"] == 1
    
    def test_reprocess_failed_articles(self):
        """Test reprocessing articles with failed text processing."""
        # Mock failed articles
        mock_articles = [
            {"url_hash": "hash1", "raw_content": "Test content 1"},
            {"url_hash": "hash2", "raw_content": ""}  # Empty content to test failure case
        ]
        
        # Configure mocks
        self.orchestrator.get_failed_text_processing_articles = MagicMock(return_value=mock_articles)
        self.orchestrator.text_processor.clean_article_text.return_value = "Processed content 1"
        
        # Call the method
        result = self.orchestrator.reprocess_failed_articles()
        
        # Assertions
        self.orchestrator.get_failed_text_processing_articles.assert_called_once()
        self.orchestrator.text_processor.clean_article_text.assert_called_once_with("Test content 1")
        
        # Check updates on articles
        update_calls = self.orchestrator.article_manager.update_article_processing_status.call_args_list
        
        # First article should be successful
        assert update_calls[0][0][0] == "hash1"  # First positional arg is url_hash
        assert update_calls[0][1]["status"] == "SUCCESS"
        
        # Second article should fail due to empty content
        assert update_calls[1][0][0] == "hash2"  # First positional arg is url_hash
        assert update_calls[1][1]["status"] == "FAILED"
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_reprocessed"] == 1
        assert result["articles_failed"] == 1
    
    def test_embed_processed_articles(self):
        """Test embedding processed articles."""
        # Create mock processed articles
        mock_articles = [
            {
                "url_hash": "hash1", 
                "processed_content": "Processed content 1",
                "published_at": "2023-01-01T12:00:00Z",
                "source_query_tag": "TECHNOLOGY"
            },
            {
                "url_hash": "hash2", 
                "processed_content": "Processed content 2",
                "published_at": "2023-01-02T12:00:00Z",
                "source_query_symbol": "AAPL"
            }
        ]
        
        # Mock chunks
        mock_chunks1 = ["Chunk 1.1", "Chunk 1.2"]
        mock_chunks2 = ["Chunk 2.1", "Chunk 2.2"]
        
        # Mock embeddings
        mock_embeddings1 = [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings2 = [[0.5, 0.6], [0.7, 0.8]]
        
        # Configure mocks
        self.orchestrator.article_manager.get_processed_articles_for_embedding.return_value = mock_articles
        self.orchestrator.text_processor.split_into_chunks.side_effect = [mock_chunks1, mock_chunks2]
        self.orchestrator.embeddings_generator.generate_embeddings.side_effect = [mock_embeddings1, mock_embeddings2]
        self.orchestrator.chroma_manager.add_embeddings.return_value = True
        self.mock_embeddings_instance.embedding_dim = 2  # For simplicity
        
        # Call the method
        result = self.orchestrator.embed_processed_articles()
        
        # Assertions
        self.orchestrator.article_manager.get_processed_articles_for_embedding.assert_called_once()
        assert self.orchestrator.text_processor.split_into_chunks.call_count == 2
        assert self.orchestrator.embeddings_generator.generate_embeddings.call_count == 2
        assert self.orchestrator.chroma_manager.add_embeddings.call_count == 2
        
        # Check that embedding status was updated
        update_calls = self.orchestrator.article_manager.update_article_embedding_status.call_args_list
        
        # First article
        assert update_calls[0][1]["url_hash"] == "hash1"
        assert update_calls[0][1]["status"] == "SUCCESS"
        assert update_calls[0][1]["embedding_model"] == self.mock_embeddings_instance.model_name
        
        # Second article
        assert update_calls[1][1]["url_hash"] == "hash2"
        assert update_calls[1][1]["status"] == "SUCCESS"
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_embedded"] == 2
        assert result["articles_failed"] == 0
    
    def test_re_embed_failed_articles(self):
        """Test re-embedding articles with failed embedding status."""
        # Create mock failed articles
        mock_articles = [
            {
                "url_hash": "hash1", 
                "processed_content": "Processed content 1",
                "published_at": "2023-01-01T12:00:00Z"
            }
        ]
        
        # Mock chunks and embeddings
        mock_chunks = ["Chunk 1.1", "Chunk 1.2"]
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        # Configure mocks
        self.orchestrator.get_failed_embedding_articles = MagicMock(return_value=mock_articles)
        self.orchestrator.text_processor.split_into_chunks.return_value = mock_chunks
        self.orchestrator.embeddings_generator.generate_embeddings.return_value = mock_embeddings
        self.orchestrator.chroma_manager.delete_embeddings_by_article.return_value = True
        self.orchestrator.chroma_manager.add_embeddings.return_value = True
        self.mock_embeddings_instance.embedding_dim = 2  # For simplicity
        
        # Call the method
        result = self.orchestrator.re_embed_failed_articles()
        
        # Assertions
        self.orchestrator.get_failed_embedding_articles.assert_called_once()
        self.orchestrator.text_processor.split_into_chunks.assert_called_once_with("Processed content 1")
        self.orchestrator.embeddings_generator.generate_embeddings.assert_called_once_with(mock_chunks)
        self.orchestrator.chroma_manager.delete_embeddings_by_article.assert_called_once_with("hash1")
        self.orchestrator.chroma_manager.add_embeddings.assert_called_once()
        
        # Check that embedding status was updated
        update_call = self.orchestrator.article_manager.update_article_embedding_status.call_args[1]
        assert update_call["url_hash"] == "hash1"
        assert update_call["status"] == "SUCCESS"
        
        # Check result
        assert result["status"] == "SUCCESS"
        assert result["articles_reembedded"] == 1
        assert result["articles_failed"] == 0
    
    def test_get_article_database_status(self):
        """Test getting article database statistics."""
        # Mock cursor and connection
        mock_cursor = MagicMock()
        self.orchestrator.article_manager.conn.cursor.return_value = mock_cursor
        
        # Set up mock query results
        mock_cursor.fetchone.side_effect = [
            (100,),  # Total articles
            ("2023-01-01", "2023-01-31"),  # Date range
            (5,),  # Total API calls
            (60,)  # Total articles retrieved
        ]
        mock_cursor.fetchall.side_effect = [
            [("PENDING", 50), ("SUCCESS", 40), ("FAILED", 10)],  # Text processing status
            [("PENDING", 60), ("SUCCESS", 30), ("FAILED", 10)],  # Embedding status
            [("TECHNOLOGY", 30), ("FINANCE", 20)],  # Tags
            [("AAPL", 25), ("MSFT", 15)]  # Symbols
        ]
        
        # Call the method
        result = self.orchestrator.get_article_database_status()
        
        # Assertions
        assert self.orchestrator.article_manager.conn.cursor.call_count == 1
        assert mock_cursor.execute.call_count == 8
        
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
