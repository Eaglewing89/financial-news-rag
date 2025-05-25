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
    
    def test_constructor_error_handling_integration(self):
        """Test constructor error handling for missing API keys."""
        # Test missing EODHD API key
        with patch.dict(os.environ, {
            "EODHD_API_KEY": "",
            "GEMINI_API_KEY": "test_gemini_api_key",
            "DATABASE_PATH_OVERRIDE": self.test_db_path,
            "CHROMA_DEFAULT_PERSIST_DIRECTORY_OVERRIDE": self.test_chroma_dir,
        }):
            with pytest.raises(ValueError, match="EODHD API key not provided"):
                FinancialNewsRAG(
                    db_path=self.test_db_path,
                    chroma_persist_dir=self.test_chroma_dir
                )
        
        # Test missing Gemini API key
        with patch.dict(os.environ, {
            "EODHD_API_KEY": "test_eodhd_api_key",
            "GEMINI_API_KEY": "",
            "DATABASE_PATH_OVERRIDE": self.test_db_path,
            "CHROMA_DEFAULT_PERSIST_DIRECTORY_OVERRIDE": self.test_chroma_dir,
        }):
            with pytest.raises(ValueError, match="Gemini API key not provided"):
                FinancialNewsRAG(
                    db_path=self.test_db_path,
                    chroma_persist_dir=self.test_chroma_dir
                )

    def test_fetch_articles_exception_handling_integration(self, orchestrator):
        """Test exception handling in fetch_and_store_articles method."""
        # Mock EODHD API to raise an exception
        with patch.object(orchestrator.eodhd_client, 'fetch_news') as mock_get_news:
            mock_get_news.side_effect = Exception("EODHD API connection failed")
            
            result = orchestrator.fetch_and_store_articles(tag="TECHNOLOGY")
            
            assert result["status"] == "FAILED"
            assert result["articles_fetched"] == 0
            assert result["articles_stored"] == 0
            assert len(result["errors"]) > 0
            assert "EODHD API connection failed" in str(result["errors"])

        # Test ArticleManager.store_articles exception
        with patch.object(orchestrator.article_manager, 'store_articles') as mock_store:
            mock_store.side_effect = Exception("Database write failed")
            
            result = orchestrator.fetch_and_store_articles(symbol="AAPL")
            
            assert result["status"] == "FAILED"
            assert len(result["errors"]) > 0
            assert "Database write failed" in str(result["errors"])

    def test_process_articles_exception_handling_integration(self, orchestrator):
        """Test exception handling in process_articles_by_status method."""
        # First add an article to process
        orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", limit=1)
        
        # Mock text processor to raise an exception during processing
        with patch.object(orchestrator.text_processor, 'process_and_validate_content') as mock_process:
            mock_process.side_effect = Exception("Text processing failed")
            
            result = orchestrator.process_articles_by_status(status="PENDING")
            
            assert result["status"] == "SUCCESS"  # Overall operation succeeds but individual articles fail
            assert result["articles_failed"] == 1
            assert result["articles_processed"] == 0
            assert len(result["errors"]) > 0
            assert "Text processing failed" in str(result["errors"])

        # Test main exception handling in process_articles_by_status
        with patch.object(orchestrator.article_manager, 'get_articles_by_processing_status') as mock_get:
            mock_get.side_effect = Exception("Database query failed")
            
            result = orchestrator.process_articles_by_status(status="PENDING")
            
            assert result["status"] == "FAILED"
            assert len(result["errors"]) > 0
            assert "Database query failed" in str(result["errors"])

    def test_embed_articles_exception_handling_integration(self, orchestrator):
        """Test exception handling in embed_processed_articles method."""
        # Add and process an article
        orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", limit=1)
        orchestrator.process_articles_by_status(status="PENDING")
        
        # Test missing processed content scenario
        with patch.object(orchestrator.article_manager, 'get_processed_articles_for_embedding') as mock_get:
            mock_get.return_value = [{
                'url_hash': 'test_hash_123',
                'processed_content': None,  # Missing content
                'title': 'Test Article',
                'published_at': '2024-01-01T00:00:00Z'
            }]
            
            result = orchestrator.embed_processed_articles(status="PENDING")
            
            assert result["status"] == "SUCCESS"
            assert result["articles_failed"] == 1
            assert result["articles_embedding_succeeded"] == 0

        # Test no chunks generated scenario
        with patch.object(orchestrator.text_processor, 'split_into_chunks') as mock_split:
            mock_split.return_value = []  # No chunks
            
            result = orchestrator.embed_processed_articles(status="PENDING")
            
            assert result["status"] == "SUCCESS"
            assert result["articles_failed"] >= 1

        # Test embedding generation failure
        with patch.object(orchestrator.embeddings_generator, 'generate_and_verify_embeddings') as mock_embed:
            # Make sure we return a properly structured failure with the expected data format
            mock_embed.return_value = {
                "embeddings": [],
                "all_valid": False  # Invalid embeddings
            }
            
            # Also ensure we have at least one article that will trigger the embedding process
            with patch.object(orchestrator.article_manager, 'get_processed_articles_for_embedding') as mock_get:
                mock_get.return_value = [{
                    'url_hash': 'test_hash_embed_failure',
                    'processed_content': 'Test content for embedding failure',
                    'title': 'Test Article',
                    'published_at': '2024-01-01T00:00:00Z'
                }]
                
                result = orchestrator.embed_processed_articles(status="PENDING")
                
                assert result["status"] == "SUCCESS"
                assert result["articles_failed"] >= 1

        # Test ChromaDB storage failure
        with patch.object(orchestrator.chroma_manager, 'add_article_chunks') as mock_add:
            mock_add.return_value = False  # Storage failed
            
            # Ensure we have an article that will trigger the storage code path
            with patch.object(orchestrator.article_manager, 'get_processed_articles_for_embedding') as mock_get:
                mock_get.return_value = [{
                    'url_hash': 'test_hash_storage_failure',
                    'processed_content': 'Test content for storage failure',
                    'title': 'Test Article',
                    'published_at': '2024-01-01T00:00:00Z'
                }]
                
                # Make sure embedding generation succeeds so we reach the storage step
                with patch.object(orchestrator.embeddings_generator, 'generate_and_verify_embeddings') as mock_embed:
                    mock_embed.return_value = {
                        "embeddings": ["mock_embedding"],
                        "all_valid": True  # Valid embeddings
                    }
                    
                    # Also mock the split_into_chunks to ensure we have chunks
                    with patch.object(orchestrator.text_processor, 'split_into_chunks') as mock_split:
                        mock_split.return_value = ["chunk1"]  # Return a valid chunk
                        
                        result = orchestrator.embed_processed_articles(status="PENDING")
                        
                        assert result["status"] == "SUCCESS"
                        assert result["articles_failed"] >= 1

        # Test main exception handling
        with patch.object(orchestrator.article_manager, 'get_processed_articles_for_embedding') as mock_get:
            mock_get.side_effect = Exception("Database connection lost")
            
            result = orchestrator.embed_processed_articles(status="PENDING")
            
            assert result["status"] == "FAILED"
            assert len(result["errors"]) > 0
            assert "Database connection lost" in str(result["errors"])

        # Test individual article exception handling
        with patch.object(orchestrator.embeddings_generator, 'generate_and_verify_embeddings') as mock_embed:
            mock_embed.side_effect = Exception("Embedding API error")
            
            # Ensure we have an article that will trigger the embedding process
            with patch.object(orchestrator.article_manager, 'get_processed_articles_for_embedding') as mock_get:
                mock_get.return_value = [{
                    'url_hash': 'test_hash_api_error',
                    'processed_content': 'Test content for API error',
                    'title': 'Test Article',
                    'published_at': '2024-01-01T00:00:00Z'
                }]
                
                result = orchestrator.embed_processed_articles(status="PENDING")
                
                assert result["status"] == "SUCCESS"
                assert result["articles_failed"] >= 1
                assert len(result["errors"]) > 0

    def test_database_status_exception_handling_integration(self, orchestrator):
        """Test exception handling in database status methods."""
        # Test get_article_database_status exception handling
        with patch.object(orchestrator.article_manager, 'get_database_statistics') as mock_status:
            mock_status.side_effect = Exception("SQLite connection failed")
            
            result = orchestrator.get_article_database_status()
            
            assert "error" in result
            assert result["status"] == "FAILED"
            assert "SQLite connection failed" in result["error"]

        # Test get_vector_database_status exception handling
        with patch.object(orchestrator.chroma_manager, 'get_collection_status') as mock_status:
            mock_status.side_effect = Exception("ChromaDB connection failed")
            
            result = orchestrator.get_vector_database_status()
            
            assert "error" in result
            assert result["status"] == "FAILED"
            assert "ChromaDB connection failed" in result["error"]

    def test_search_exception_handling_integration(self, orchestrator):
        """Test exception handling in search_articles method."""
        # Add some test data first
        orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", limit=1)
        orchestrator.process_articles_by_status(status="PENDING")
        orchestrator.embed_processed_articles(status="PENDING")
        
        # Test embedding generation failure during search
        with patch.object(orchestrator.embeddings_generator, 'generate_embeddings') as mock_embed:
            mock_embed.side_effect = Exception("Embedding generation failed")
            
            result = orchestrator.search_articles("test query")
            
            assert result == []  # Should return empty list on error

        # Test ChromaDB query failure
        with patch.object(orchestrator.chroma_manager, 'query_embeddings') as mock_query:
            mock_query.side_effect = Exception("Vector database query failed")
            
            result = orchestrator.search_articles("test query")
            
            assert result == []  # Should return empty list on error

        # Test article retrieval failure
        with patch.object(orchestrator.article_manager, 'get_article_by_hash') as mock_get:
            mock_get.side_effect = Exception("Article retrieval failed")
            
            result = orchestrator.search_articles("test query")
            
            assert result == []  # Should return empty list on error

    def test_delete_operations_comprehensive_integration(self, orchestrator):
        """Test comprehensive delete operations including error scenarios."""
        # Add some test articles with different ages
        from datetime import datetime, timezone, timedelta
        
        # Add current articles
        orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", limit=1)
        orchestrator.process_articles_by_status(status="PENDING")
        orchestrator.embed_processed_articles(status="PENDING")
        
        # Test deleting articles when none are old enough
        result = orchestrator.delete_articles_older_than(days=1)
        assert result["status"] in ["SUCCESS", "FAILED"]
        assert result["deleted_from_sqlite"] == 0
        assert result["deleted_from_chroma"] == 0
        
        # Mock old articles for deletion testing
        old_date = datetime.now(timezone.utc) - timedelta(days=200)
        with patch.object(orchestrator.chroma_manager, 'get_article_hashes_by_date_range') as mock_get_old:
            mock_get_old.return_value = ['old_hash_1', 'old_hash_2']
            
            # Test successful deletion
            with patch.object(orchestrator.chroma_manager, 'delete_embeddings_by_article') as mock_del_chroma, \
                 patch.object(orchestrator.article_manager, 'delete_article_by_hash') as mock_del_sqlite:
                mock_del_chroma.return_value = True
                mock_del_sqlite.return_value = True
                
                result = orchestrator.delete_articles_older_than(days=180)
                
                assert result["status"] == "SUCCESS"
                assert result["deleted_from_sqlite"] == 2
                assert result["deleted_from_chroma"] == 2
                assert len(result["errors"]) == 0

            # Test partial failure - ChromaDB deletion fails
            with patch.object(orchestrator.chroma_manager, 'delete_embeddings_by_article') as mock_del_chroma, \
                 patch.object(orchestrator.article_manager, 'delete_article_by_hash') as mock_del_sqlite:
                mock_del_chroma.return_value = False
                mock_del_sqlite.return_value = True
                
                result = orchestrator.delete_articles_older_than(days=180)
                
                assert result["status"] == "SUCCESS"  # Still success if SQLite deletion worked
                assert result["deleted_from_sqlite"] == 2
                assert result["deleted_from_chroma"] == 0

            # Test complete failure - both deletions fail
            with patch.object(orchestrator.chroma_manager, 'delete_embeddings_by_article') as mock_del_chroma, \
                 patch.object(orchestrator.article_manager, 'delete_article_by_hash') as mock_del_sqlite:
                mock_del_chroma.return_value = False
                mock_del_sqlite.return_value = False
                
                result = orchestrator.delete_articles_older_than(days=180)
                
                assert result["status"] == "FAILED"
                assert result["deleted_from_sqlite"] == 0
                assert result["deleted_from_chroma"] == 0
                assert len(result["errors"]) > 0

            # Test exception during deletion
            with patch.object(orchestrator.chroma_manager, 'delete_embeddings_by_article') as mock_del_chroma:
                mock_del_chroma.side_effect = Exception("ChromaDB deletion error")
                
                result = orchestrator.delete_articles_older_than(days=180)
                
                assert result["status"] in ["FAILED", "PARTIAL_FAILURE"]
                assert len(result["errors"]) > 0
                assert "ChromaDB deletion error" in str(result["errors"])

        # Test main exception handling
        with patch.object(orchestrator.chroma_manager, 'get_article_hashes_by_date_range') as mock_get_old:
            mock_get_old.side_effect = Exception("Database query failed")
            
            result = orchestrator.delete_articles_older_than(days=180)
            
            assert result["status"] == "FAILED"
            assert len(result["errors"]) > 0
            assert "Database query failed" in str(result["errors"])

    def test_resource_cleanup_error_handling_integration(self, orchestrator):
        """Test error handling in resource cleanup method."""
        # Test cleanup when component cleanup fails
        with patch.object(orchestrator.article_manager, 'close_connection') as mock_close_sqlite:
            mock_close_sqlite.side_effect = Exception("SQLite close failed")
            
            # Should not raise exception, just log error
            orchestrator.close()
            
            # Verify the method was called
            mock_close_sqlite.assert_called_once()

        # Test cleanup when ChromaDB cleanup fails
        with patch.object(orchestrator.chroma_manager, 'close_connection') as mock_close_chroma:
            mock_close_chroma.side_effect = Exception("ChromaDB close failed")
            
            # Should not raise exception, just log error
            orchestrator.close()
            
            # Verify the method was called
            mock_close_chroma.assert_called_once()

    def test_edge_cases_and_boundary_conditions_integration(self, orchestrator):
        """Test edge cases and boundary conditions."""
        # Test search with empty query
        result = orchestrator.search_articles("", n_results=5)
        assert isinstance(result, list)
        
        # Test search with very long query
        long_query = "technology " * 100
        result = orchestrator.search_articles(long_query, n_results=1)
        assert isinstance(result, list)
        
        # Test delete with edge case days values
        result = orchestrator.delete_articles_older_than(days=0)
        assert "status" in result
        
        result = orchestrator.delete_articles_older_than(days=99999)
        assert "status" in result
        
        # Test processing with different status values
        result = orchestrator.process_articles_by_status(status="NONEXISTENT")
        assert result["status"] == "SUCCESS"
        assert result["articles_processed"] == 0
        
        # Test embedding with FAILED status (re-embedding scenario)
        orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", limit=1)
        orchestrator.process_articles_by_status(status="PENDING")
        
        # Simulate failed embedding status
        with patch.object(orchestrator.article_manager, 'get_processed_articles_for_embedding') as mock_get:
            mock_get.return_value = [{
                'url_hash': 'test_hash_failed',
                'processed_content': 'Test content for re-embedding',
                'title': 'Test Article',
                'published_at': '2024-01-01T00:00:00Z'
            }]
            
            result = orchestrator.embed_processed_articles(status="FAILED")
            assert result["status"] == "SUCCESS"

    def test_search_metadata_filtering_integration(self, orchestrator):
        """Test search functionality with metadata filtering."""
        # Add test data
        orchestrator.fetch_and_store_articles(tag="TECHNOLOGY", limit=1)
        orchestrator.process_articles_by_status(status="PENDING")
        orchestrator.embed_processed_articles(status="PENDING")
        
        # Test search with sort_by_metadata
        result = orchestrator.search_articles(
            "technology",
            n_results=5,
            sort_by_metadata={"published_at_timestamp": "desc"}
        )
        assert isinstance(result, list)
        
        # Test search with date filters
        result = orchestrator.search_articles(
            "technology",
            n_results=5,
            from_date_str="2020-01-01",
            to_date_str="2030-12-31"
        )
        assert isinstance(result, list)
