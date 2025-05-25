"""
Integration tests for the FinancialNewsRAG orchestrator class.

These tests verify that the orchestrator correctly integrates all components
and handles real interactions between ArticleManager, TextProcessor,
EmbeddingsGenerator, ChromaDBManager, and ReRanker.

External APIs (EODHD, Gemini) are mocked to avoid costs and ensure reliability,
but all internal component interactions are tested with real implementations.
"""

import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

# Import the orchestrator and components
from financial_news_rag.orchestrator import FinancialNewsRAG
from financial_news_rag.article_manager import ArticleManager
from financial_news_rag.text_processor import TextProcessor
from financial_news_rag.embeddings import EmbeddingsGenerator
from financial_news_rag.chroma_manager import ChromaDBManager
from financial_news_rag.reranker import ReRanker
from financial_news_rag.config import Config

# Import test utilities
from tests.fixtures.sample_data import ArticleFactory
from tests.helpers.assertions import assert_article_structure, assert_search_result_structure


class TestFinancialNewsRAGIntegration:
    """
    Integration tests for the FinancialNewsRAG orchestrator.
    
    These tests verify complete workflows and real component interactions,
    while mocking only external APIs for cost and reliability reasons.
    """
    
    @pytest.fixture(autouse=True)
    def setup_integration_environment(self, temp_directory):
        """Set up a complete integration test environment with real databases."""
        # Create temporary directories for databases
        self.test_db_path = os.path.join(temp_directory, "test_integration.db")
        self.test_chroma_dir = os.path.join(temp_directory, "test_chroma")
        
        # Set up environment variables for test configuration
        self.env_patcher = patch.dict(os.environ, {
            "EODHD_API_KEY": "test_eodhd_api_key",
            "GEMINI_API_KEY": "test_gemini_api_key",
            "DATABASE_PATH_OVERRIDE": self.test_db_path,
            "CHROMA_DEFAULT_PERSIST_DIRECTORY_OVERRIDE": self.test_chroma_dir,
            "CHROMA_DEFAULT_COLLECTION_NAME_OVERRIDE": "test_integration_collection",
        })
        self.env_patcher.start()
        
        # Mock external APIs but keep internal components real
        self.eodhd_api_patcher = patch('financial_news_rag.eodhd.requests.get')
        self.gemini_embedding_patcher = patch('financial_news_rag.embeddings.genai.Client')
        self.gemini_chat_patcher = patch('financial_news_rag.reranker.genai.Client')
        
        self.mock_eodhd_api = self.eodhd_api_patcher.start()
        self.mock_gemini_embedding = self.gemini_embedding_patcher.start()
        self.mock_gemini_chat = self.gemini_chat_patcher.start()
        
        # Configure realistic API responses
        self._configure_api_mocks()
        
        yield
        
        # Cleanup
        self.env_patcher.stop()
        self.eodhd_api_patcher.stop()
        self.gemini_embedding_patcher.stop()
        self.gemini_chat_patcher.stop()
    
    def _configure_api_mocks(self):
        """Configure realistic responses for external API mocks."""
        # Use current timestamps to ensure mock articles appear "fresh"
        from datetime import datetime, timezone
        current_time = datetime.now(timezone.utc)
        recent_time = current_time.replace(hour=current_time.hour-1)  # 1 hour ago
        
        # Mock EODHD API response - create proper mock response object for requests.get()
        raw_eodhd_articles = [
            {
                'title': 'Apple Reports Strong Q4 Earnings',
                'link': 'https://example.com/apple-earnings-2024',
                'date': current_time.isoformat(),
                'content': '<p>Apple Inc. reported strong fourth quarter earnings...</p><p>Revenue exceeded expectations at $89.5 billion.</p>',
                'symbols': ['AAPL.US'],
                'tags': ['earnings', 'technology'],
                'sentiment': {}
            },
            {
                'title': 'Microsoft Azure Growth Continues',
                'link': 'https://example.com/microsoft-azure-2024',
                'date': recent_time.isoformat(),
                'content': '<p>Microsoft continues to see strong growth in Azure cloud services...</p><p>Enterprise adoption driving revenue growth.</p>',
                'symbols': ['MSFT.US'],
                'tags': ['technology', 'cloud'],
                'sentiment': {}
            }
        ]
        
        # Create a smarter mock that respects the limit parameter
        def mock_eodhd_response(*args, **kwargs):
            # Extract the limit from the request params
            params = kwargs.get('params', {})
            limit = params.get('limit', len(raw_eodhd_articles))
            
            # Respect the limit parameter - return up to 'limit' articles
            limited_articles = raw_eodhd_articles[:limit]
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "valid json response"
            mock_response.json.return_value = limited_articles
            mock_response.raise_for_status.return_value = None
            return mock_response
        
        self.mock_eodhd_api.side_effect = mock_eodhd_response
        
        # Mock Gemini chat client for reranking
        mock_chat_client = MagicMock()
        mock_chat_models = MagicMock()
        mock_chat_response = MagicMock()
        mock_chat_response.text = '1\n2'  # Simple rerank order
        mock_chat_models.generate_content.return_value = mock_chat_response
        mock_chat_client.models = mock_chat_models
        self.mock_gemini_chat.return_value = mock_chat_client
    
    @pytest.fixture
    def orchestrator(self):
        """Create a FinancialNewsRAG orchestrator with real components."""
        orchestrator = FinancialNewsRAG(
            db_path=self.test_db_path,
            chroma_persist_dir=self.test_chroma_dir,
            chroma_collection_name="test_integration_collection"
        )
        
        # Directly patch the embed_content method on the orchestrator's embedding generator client
        # This is a more targeted approach that ensures the mock works correctly
        import numpy as np
        
        # Create a simple, deterministic embedding that's always the same
        base_embedding = np.random.rand(768)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        embedding_values = base_embedding.tolist()
        
        # Create a simple class to hold embedding values
        class MockEmbedding:
            def __init__(self, values):
                self.values = values
        
        # Create the mock response structure directly
        mock_response = MagicMock()
        mock_embedding = MockEmbedding(embedding_values)
        mock_response.embeddings = [mock_embedding]
        
        # Directly patch the embed_content method on the already-created client
        def mock_embed_content(*args, **kwargs):
            return mock_response
        
        # Apply the patch directly to the orchestrator's client
        orchestrator.embeddings_generator.client.models.embed_content = mock_embed_content
        
        return orchestrator
    
    def test_orchestrator_initialization_integration(self, orchestrator):
        """Test that the orchestrator initializes all real components correctly."""
        # Verify orchestrator was created
        assert orchestrator is not None
        
        # Verify all components are real instances (not mocks)
        assert isinstance(orchestrator.article_manager, ArticleManager)
        assert isinstance(orchestrator.text_processor, TextProcessor)
        assert isinstance(orchestrator.embeddings_generator, EmbeddingsGenerator)
        assert isinstance(orchestrator.chroma_manager, ChromaDBManager)
        assert isinstance(orchestrator.reranker, ReRanker)
        assert isinstance(orchestrator.config, Config)
        
        # Verify database paths are set correctly
        assert orchestrator.article_manager.db_path == self.test_db_path
        assert orchestrator.chroma_manager.persist_directory == self.test_chroma_dir
        
        # Verify API keys are configured
        assert orchestrator.eodhd_api_key == "test_eodhd_api_key"
        assert orchestrator.gemini_api_key == "test_gemini_api_key"
    
    def test_complete_rag_workflow_integration(self, orchestrator):
        """Test the complete RAG workflow end-to-end."""
        # Step 1: Fetch articles
        fetch_result = orchestrator.fetch_and_store_articles(tag="TECHNOLOGY")
        
        # Debug: Print details about the fetch result
        assert fetch_result["status"] == "SUCCESS"
        assert fetch_result["articles_stored"] == 2
        assert fetch_result["articles_fetched"] == 2
        assert len(fetch_result["errors"]) == 0
        
        # Verify articles were stored in database
        db_status = orchestrator.get_article_database_status()
        assert db_status["total_articles"] == 2
        
        # Step 2: Process articles
        process_result = orchestrator.process_articles_by_status(status="PENDING")
        
        assert process_result["status"] == "SUCCESS"
        assert process_result["articles_processed"] == 2
        assert process_result["articles_failed"] == 0
        
        # Verify processing status was updated
        db_status_after_processing = orchestrator.get_article_database_status()
        assert db_status_after_processing["text_processing_status"]["SUCCESS"] == 2
        assert db_status_after_processing["text_processing_status"].get("PENDING", 0) == 0
        
        # Step 3: Generate and store embeddings
        embedding_result = orchestrator.embed_processed_articles(status="PENDING")
        
        assert embedding_result["status"] == "SUCCESS"
        assert embedding_result["articles_embedding_succeeded"] == 2
        assert embedding_result["articles_failed"] == 0
        
        # Verify embeddings were stored in vector database
        vector_status = orchestrator.get_vector_database_status()
        assert vector_status["total_chunks"] > 0
        assert vector_status["unique_articles"] == 2
        
        # Verify embedding status was updated
        db_status_after_embedding = orchestrator.get_article_database_status()
        assert db_status_after_embedding["embedding_status"]["SUCCESS"] == 2
        
        # Step 4: Search for articles
        search_results = orchestrator.search_articles(
            query="Apple earnings technology",
            n_results=2,
            rerank=False
        )
        
        assert len(search_results) == 2
        for result in search_results:
            assert_search_result_structure(result)
            assert "similarity_score" in result
            assert result["similarity_score"] > 0
        
        # Test search with reranking
        reranked_results = orchestrator.search_articles(
            query="Microsoft cloud services",
            n_results=2,
            rerank=True
        )
        
        assert len(reranked_results) == 2
        for result in reranked_results:
            assert_search_result_structure(result)
            assert "similarity_score" in result
            assert "rerank_score" in result
    
    def test_article_lifecycle_management_integration(self, orchestrator):
        """Test complete article lifecycle including deletion."""
        # Add some articles
        fetch_result = orchestrator.fetch_and_store_articles(symbol="AAPL", limit=1)
        assert fetch_result["articles_stored"] == 1
        
        # Process and embed
        orchestrator.process_articles_by_status(status="PENDING")
        orchestrator.embed_processed_articles(status="PENDING")
        
        # Verify articles exist in both databases
        db_status = orchestrator.get_article_database_status()
        vector_status = orchestrator.get_vector_database_status()
        assert db_status["total_articles"] == 1
        assert vector_status["unique_articles"] == 1
        
        # Test deletion of old articles (should delete nothing since articles are new)
        delete_result = orchestrator.delete_articles_older_than(days=1)
        assert delete_result["status"] in ["SUCCESS", "FAILED"]  # No articles to delete
        assert delete_result["deleted_from_sqlite"] == 0
        assert delete_result["deleted_from_chroma"] == 0
        
        # Test deletion with a large number of days (should still delete nothing)
        delete_result_large = orchestrator.delete_articles_older_than(days=365)
        assert delete_result_large["deleted_from_sqlite"] == 0
        assert delete_result_large["deleted_from_chroma"] == 0
    
    def test_error_handling_integration(self, orchestrator):
        """Test error handling in integrated workflows."""
        # Test with invalid tag/symbol combination
        with pytest.raises(ValueError, match="tag and symbol parameters are mutually exclusive"):
            orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", symbol="AAPL")
        
        # Test with no tag or symbol
        with pytest.raises(ValueError, match="Either tag or symbol must be provided"):
            orchestrator.fetch_and_store_articles()
        
        # Test processing when no articles exist
        process_result = orchestrator.process_articles_by_status(status="PENDING")
        assert process_result["status"] == "SUCCESS"
        assert process_result["articles_processed"] == 0
        assert process_result["articles_failed"] == 0
        
        # Test embedding when no processed articles exist
        embedding_result = orchestrator.embed_processed_articles(status="PENDING")
        assert embedding_result["status"] == "SUCCESS"
        assert embedding_result["articles_embedding_succeeded"] == 0
        assert embedding_result["articles_failed"] == 0
    
    def test_database_consistency_integration(self, orchestrator):
        """Test that operations maintain consistency between SQLite and ChromaDB."""
        # Add and process articles
        orchestrator.fetch_and_store_articles(tag="FINANCE", limit=1)
        orchestrator.process_articles_by_status(status="PENDING")
        orchestrator.embed_processed_articles(status="PENDING")
        
        # Get initial states
        initial_db_status = orchestrator.get_article_database_status()
        initial_vector_status = orchestrator.get_vector_database_status()
        
        # Verify consistency
        assert initial_db_status["total_articles"] == 1
        assert initial_vector_status["unique_articles"] == 1
        assert initial_db_status["embedding_status"]["SUCCESS"] == 1
        
        # Test that search returns consistent results
        search_results = orchestrator.search_articles("finance banking", n_results=5)
        assert len(search_results) <= initial_vector_status["unique_articles"]
    
    def test_search_functionality_integration(self, orchestrator):
        """Test comprehensive search functionality with real components."""
        # Add diverse articles
        orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", limit=1)
        
        # Process and embed
        orchestrator.process_articles_by_status(status="PENDING")
        orchestrator.embed_processed_articles(status="PENDING")
        
        # Test basic search
        basic_results = orchestrator.search_articles("technology", n_results=1)
        assert len(basic_results) == 1
        assert_search_result_structure(basic_results[0])
        
        # Test search with date filters
        date_results = orchestrator.search_articles(
            "technology",
            n_results=1,
            from_date_str="2024-01-01",
            to_date_str="2024-12-31"
        )
        assert len(date_results) <= 1
        
        # Test search with no results
        no_results = orchestrator.search_articles("completely_irrelevant_query", n_results=5)
        assert isinstance(no_results, list)  # Should return empty list, not error
    
    def test_component_interaction_integration(self, orchestrator):
        """Test that components interact correctly through the orchestrator."""
        # Fetch articles
        fetch_result = orchestrator.fetch_and_store_articles(symbol="MSFT", limit=1)
        assert fetch_result["status"] == "SUCCESS"
        
        # Verify ArticleManager stored the articles
        db_status = orchestrator.get_article_database_status()
        assert db_status["total_articles"] == 1
        assert "MSFT" in str(db_status.get("articles_by_symbol", {}))
        
        # Process articles (TextProcessor interaction)
        process_result = orchestrator.process_articles_by_status(status="PENDING")
        assert process_result["articles_processed"] == 1
        
        # Verify TextProcessor updated the content
        updated_db_status = orchestrator.get_article_database_status()
        assert updated_db_status["text_processing_status"]["SUCCESS"] == 1
        
        # Generate embeddings (EmbeddingsGenerator + ChromaDBManager interaction)
        embedding_result = orchestrator.embed_processed_articles(status="PENDING")
        assert embedding_result["articles_embedding_succeeded"] == 1
        
        # Verify ChromaDBManager stored the embeddings
        vector_status = orchestrator.get_vector_database_status()
        assert vector_status["total_chunks"] > 0
        assert vector_status["unique_articles"] == 1
        
        # Test ReRanker interaction through search
        search_results = orchestrator.search_articles("Microsoft", n_results=1, rerank=True)
        assert len(search_results) == 1
        assert "rerank_score" in search_results[0]
    
    def test_configuration_inheritance_integration(self, orchestrator):
        """Test that configuration is properly inherited by all components."""
        # Verify config is passed to components
        assert orchestrator.config is not None
        
        # Check that components use the configuration
        assert orchestrator.article_manager.db_path == self.test_db_path
        assert orchestrator.chroma_manager.persist_directory == self.test_chroma_dir
        assert orchestrator.chroma_manager.collection_name == "test_integration_collection"
        
        # Verify API keys are configured at the orchestrator level
        assert orchestrator.eodhd_client.api_key == "test_eodhd_api_key"
        assert orchestrator.gemini_api_key == "test_gemini_api_key"
        
        # Verify component models are configured correctly
        assert orchestrator.embeddings_generator.model_name == "text-embedding-004"
        assert orchestrator.embeddings_generator.embedding_dim == 768
        assert orchestrator.reranker.model_name == "gemini-2.0-flash"
    
    def test_resource_cleanup_integration(self, orchestrator):
        """Test that resources are properly cleaned up."""
        # Add some data
        orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", limit=1)
        orchestrator.process_articles_by_status(status="PENDING")
        orchestrator.embed_processed_articles(status="PENDING")
        
        # Verify data exists
        db_status = orchestrator.get_article_database_status()
        vector_status = orchestrator.get_vector_database_status()
        assert db_status["total_articles"] > 0
        assert vector_status["total_chunks"] > 0
        
        # Test cleanup
        orchestrator.close()
        
        # Verify orchestrator can be recreated after cleanup
        new_orchestrator = FinancialNewsRAG(
            db_path=self.test_db_path,
            chroma_persist_dir=self.test_chroma_dir,
            chroma_collection_name="test_integration_collection"
        )
        
        # Data should persist after cleanup
        new_db_status = new_orchestrator.get_article_database_status()
        new_vector_status = new_orchestrator.get_vector_database_status()
        assert new_db_status["total_articles"] == db_status["total_articles"]
        assert new_vector_status["total_chunks"] == vector_status["total_chunks"]
        
        new_orchestrator.close()
    
    def test_performance_and_scalability_integration(self, orchestrator):
        """Test performance characteristics with multiple articles."""
        # This test uses a limited number of articles to avoid long test times
        # but verifies the system can handle batch operations
        
        # Fetch multiple articles
        fetch_result = orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", limit=2)
        assert fetch_result["articles_stored"] == 2
        
        # Process all articles in batch
        process_result = orchestrator.process_articles_by_status(status="PENDING", limit=10)
        assert process_result["articles_processed"] == 2
        
        # Embed all articles in batch
        embedding_result = orchestrator.embed_processed_articles(status="PENDING", limit=10)
        assert embedding_result["articles_embedding_succeeded"] == 2
        
        # Verify final state
        final_db_status = orchestrator.get_article_database_status()
        final_vector_status = orchestrator.get_vector_database_status()
        
        assert final_db_status["total_articles"] == 2
        assert final_db_status["text_processing_status"]["SUCCESS"] == 2
        assert final_db_status["embedding_status"]["SUCCESS"] == 2
        assert final_vector_status["unique_articles"] == 2
        assert final_vector_status["total_chunks"] >= 2  # At least one chunk per article
        
        # Test that search performance is reasonable
        search_results = orchestrator.search_articles("technology", n_results=5)
        assert len(search_results) == 2  # Should return all available articles
