"""
Unit tests for ChromaDB Manager.

Tests the ChromaDBManager class which handles vector database operations
including initialization, adding/retrieving embeddings, querying, and management.
"""

import tempfile
from typing import Dict, List, Any
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from financial_news_rag.chroma_manager import ChromaDBManager


class TestChromaDBManager:
    """Test suite for ChromaDBManager class."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, temp_directory):
        """Setup test environment before each test."""
        # Generate a unique collection name for each test to avoid state sharing
        self.test_collection_name = f"test_collection_{id(self)}"
        
        # Default embedding dimension for testing
        self.test_embedding_dimension = 768
        
        # Create ChromaDBManager with in-memory mode for faster testing
        self.chroma_manager = ChromaDBManager(
            persist_directory=temp_directory,
            collection_name=self.test_collection_name,
            embedding_dimension=self.test_embedding_dimension,
            in_memory=True  # Use in-memory mode for tests
        )
        
        # Sample article hash for testing
        self.sample_article_hash = "test_article_123"
    
    def test_initialization(self):
        """Test successful initialization and collection creation."""
        # Check if ChromaDBManager was initialized correctly
        assert self.chroma_manager.collection_name == self.test_collection_name
        assert self.chroma_manager.embedding_dimension == self.test_embedding_dimension
        
        # Check if collection was created
        collection_status = self.chroma_manager.get_collection_status()
        assert collection_status["collection_name"] == self.test_collection_name
        assert collection_status["is_empty"] is True
    
    def test_initialization_required_parameters(self, temp_directory):
        """Test that initialization requires the mandatory parameters."""
        # Test that initialization fails without persist_directory
        with pytest.raises(TypeError):
            ChromaDBManager(
                collection_name=self.test_collection_name,
                embedding_dimension=self.test_embedding_dimension
            )
        
        # Test that initialization fails without collection_name
        with pytest.raises(TypeError):
            ChromaDBManager(
                persist_directory=temp_directory,
                embedding_dimension=self.test_embedding_dimension
            )
        
        # Test that initialization fails without embedding_dimension
        with pytest.raises(TypeError):
            ChromaDBManager(
                persist_directory=temp_directory,
                collection_name=self.test_collection_name
            )
    
    def test_query_embeddings_with_similarity_score(self):
        """Test querying embeddings with similarity score calculation."""
        # Set up test data
        article_url_hash = self.sample_article_hash
        chunk_texts = ["This is test chunk 0 for the article."]
        # Create a normalized random vector for the embedding
        embedding = np.random.rand(self.test_embedding_dimension)
        embedding = embedding / np.linalg.norm(embedding)
        chunk_vector = embedding.tolist()
        article_data = {
            "published_at": "2023-05-15T14:30:00Z",
            "source_query_tag": "TECHNOLOGY"
        }
        
        # Add article chunk using add_article_chunks
        self.chroma_manager.add_article_chunks(
            article_url_hash,
            chunk_texts,
            [chunk_vector],
            article_data
        )
        
        # Create a query embedding (use the same embedding for testing)
        query_embedding = chunk_vector
        
        # Query with return_similarity_score=True
        results_with_score = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=2,
            return_similarity_score=True
        )
        
        # Check results
        assert len(results_with_score) == 1
        # The result should be the exact match with similarity_score close to 1.0
        assert results_with_score[0]["chunk_id"] == f"{self.sample_article_hash}_0"
        assert "similarity_score" in results_with_score[0]
        assert "distance" not in results_with_score[0]
        assert results_with_score[0]["similarity_score"] > 0.99  # Should be very close to 1
        
        # Query with return_similarity_score=False (default)
        results_without_score = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=2
        )
        
        # Check results
        assert len(results_without_score) == 1
        # The result should have distance but not similarity_score
        assert results_without_score[0]["chunk_id"] == f"{self.sample_article_hash}_0"
        assert "distance" in results_without_score[0]
        assert "similarity_score" not in results_without_score[0]
        assert results_without_score[0]["distance"] < 0.001  # Should be very close to 0
    
    def test_query_with_filter(self):
        """Test querying with metadata filters."""
        # Set up test data for first article
        first_article_hash = self.sample_article_hash
        first_chunk_texts = ["This is test chunk 0 for the article."]
        first_chunk_vector = np.random.rand(self.test_embedding_dimension).tolist()
        first_article_data = {
            "published_at": "2023-05-15T14:30:00Z",
            "source_query_tag": "TECHNOLOGY"
        }
        
        # Add first article chunk
        self.chroma_manager.add_article_chunks(
            first_article_hash,
            first_chunk_texts,
            [first_chunk_vector],
            first_article_data
        )
        
        # Set up test data for second article
        second_article_hash = "test_article_456"
        second_chunk_texts = ["This is test chunk 0 for another article."]
        second_chunk_vector = np.random.rand(self.test_embedding_dimension).tolist()
        second_article_data = {
            "published_at": "2023-06-20T10:45:00Z",
            "source_query_tag": "FINANCE"
        }
        
        # Add second article chunk
        self.chroma_manager.add_article_chunks(
            second_article_hash,
            second_chunk_texts,
            [second_chunk_vector],
            second_article_data
        )
        
        # Create a query embedding
        query_embedding = np.random.rand(self.test_embedding_dimension)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Query with filter for first article
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=5,
            filter_metadata={"article_url_hash": first_article_hash}
        )
        
        # Check results
        assert len(results) == 1  # Should return 1 result from first article
        for result in results:
            assert result["metadata"]["article_url_hash"] == first_article_hash
        
        # Query with filter for tag
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=5,
            filter_metadata={"source_query_tag": "FINANCE"}
        )
        
        # Check results
        assert len(results) == 1  # Should return 1 result with FINANCE tag
        for result in results:
            assert result["metadata"]["source_query_tag"] == "FINANCE"
    
    def test_get_collection_status(self):
        """Test getting collection status."""
        # Check empty collection
        status = self.chroma_manager.get_collection_status()
        assert status["total_chunks"] == 0
        assert status["unique_articles"] == 0
        assert status["is_empty"] is True
        
        # Add chunk using add_article_chunks
        article_url_hash = self.sample_article_hash
        chunk_texts = ["This is test chunk 0 for the article."]
        chunk_vector = np.random.rand(self.test_embedding_dimension).tolist()
        article_data = {
            "published_at": "2023-05-15T14:30:00Z",
            "source_query_tag": "TECHNOLOGY"
        }
        
        self.chroma_manager.add_article_chunks(
            article_url_hash,
            chunk_texts,
            [chunk_vector],
            article_data
        )
        
        # Check status after adding
        status = self.chroma_manager.get_collection_status()
        assert status["total_chunks"] == 1
        assert status["unique_articles"] == 1
        assert status["is_empty"] is False
    
    def test_delete_embeddings_by_article(self):
        """Test deleting embeddings for a specific article."""
        # Add first article's chunks
        first_article_hash = self.sample_article_hash
        first_chunk_texts = ["This is test chunk 0 for the article."]
        first_chunk_vector = np.random.rand(self.test_embedding_dimension).tolist()
        first_article_data = {
            "published_at": "2023-05-15T14:30:00Z",
            "source_query_tag": "TECHNOLOGY"
        }
        
        self.chroma_manager.add_article_chunks(
            first_article_hash,
            first_chunk_texts,
            [first_chunk_vector],
            first_article_data
        )
        
        # Add second article's chunks
        second_article_hash = "test_article_456"
        second_chunk_texts = ["This is test chunk 0 for another article."]
        second_chunk_vector = np.random.rand(self.test_embedding_dimension).tolist()
        second_article_data = {
            "published_at": "2023-06-20T10:45:00Z",
            "source_query_tag": "FINANCE"
        }
        
        self.chroma_manager.add_article_chunks(
            second_article_hash,
            second_chunk_texts,
            [second_chunk_vector],
            second_article_data
        )
        
        # Check total before deletion
        status_before = self.chroma_manager.get_collection_status()
        assert status_before["total_chunks"] == 2
        assert status_before["unique_articles"] == 2
        
        # Delete first article's embeddings
        success = self.chroma_manager.delete_embeddings_by_article(first_article_hash)
        assert success is True
        
        # Check status after deletion
        status_after = self.chroma_manager.get_collection_status()
        assert status_after["total_chunks"] == 1
        assert status_after["unique_articles"] == 1
        
        # Verify only the correct embeddings were deleted
        # Query for deleted article
        query_embedding = np.random.rand(self.test_embedding_dimension)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=10,
            filter_metadata={"article_url_hash": first_article_hash}
        )
        
        # Should be empty
        assert len(results) == 0
        
        # Query for remaining article
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=10,
            filter_metadata={"article_url_hash": second_article_hash}
        )
        
        # Should have 1 result
        assert len(results) == 1
    
    def test_handling_duplicate_chunk_ids_with_article_chunks(self):
        """Test handling of duplicate chunk IDs with add_article_chunks."""
        # Sample data for first addition
        article_url_hash = "test_article_duplicate"
        chunk_texts = ["This is the original chunk."]
        chunk_vectors = [
            [0.6] * self.chroma_manager.embedding_dimension
        ]
        article_data = {
            "published_at": "2023-05-15T14:30:00Z",
            "source_query_tag": "TECHNOLOGY"
        }
        
        # First addition
        self.chroma_manager.add_article_chunks(
            article_url_hash,
            chunk_texts,
            chunk_vectors,
            article_data
        )
        
        # Check initial status
        status_before = self.chroma_manager.get_collection_status()
        assert status_before["total_chunks"] == 1
        
        # Sample data for update with same chunk ID
        updated_chunk_texts = ["This is the updated chunk."]
        updated_chunk_vectors = [
            [0.7] * self.chroma_manager.embedding_dimension
        ]
        updated_article_data = {
            "published_at": "2023-05-16T14:30:00Z",
            "source_query_tag": "UPDATED_TAG"
        }
        
        # Update with new data (will use same chunk ID format article_url_hash_0)
        self.chroma_manager.add_article_chunks(
            article_url_hash,
            updated_chunk_texts,
            updated_chunk_vectors,
            updated_article_data
        )
        
        # Check that count hasn't changed (upsert should replace)
        status_after = self.chroma_manager.get_collection_status()
        assert status_after["total_chunks"] == 1
        
        # Query to verify update was applied
        query_embedding = updated_chunk_vectors[0]
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=1
        )
        
        # Check that updated content was stored
        assert len(results) == 1
        assert results[0]["chunk_id"] == f"{article_url_hash}_0"
        assert results[0]["text"] == updated_chunk_texts[0]
        assert results[0]["metadata"]["source_query_tag"] == "UPDATED_TAG"
    
    def test_add_article_chunks(self):
        """Test adding article chunks with the new method."""
        # Sample data
        article_url_hash = "test_article_add_chunks"
        chunk_texts = ["This is the first chunk.", "This is the second chunk."]
        chunk_vectors = [
            [0.1] * self.chroma_manager.embedding_dimension,
            [0.2] * self.chroma_manager.embedding_dimension
        ]
        article_data = {
            "published_at": "2023-05-15T14:30:00Z",
            "source_query_tag": "TECHNOLOGY",
            "source_query_symbol": "AAPL"
        }
        
        # Add article chunks
        success = self.chroma_manager.add_article_chunks(
            article_url_hash,
            chunk_texts,
            chunk_vectors,
            article_data
        )
        
        # Check if addition was successful
        assert success is True
        
        # Query to verify that chunks were added correctly with all metadata
        query_embedding = [0.1] * self.chroma_manager.embedding_dimension
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=2,
            filter_metadata={"article_url_hash": article_url_hash}
        )
        
        # Verify results
        assert len(results) == 2
        
        # Verify IDs
        assert results[0]["chunk_id"] == f"{article_url_hash}_0"
        assert results[1]["chunk_id"] == f"{article_url_hash}_1"
        
        # Verify texts
        assert results[0]["text"] == chunk_texts[0]
        assert results[1]["text"] == chunk_texts[1]
        
        # Verify metadata
        for result in results:
            metadata = result["metadata"]
            # Check required metadata fields
            assert metadata["article_url_hash"] == article_url_hash
            assert "chunk_index" in metadata
            
            # Check optional metadata fields
            assert "published_at_timestamp" in metadata  # Should be converted from ISO string
            assert metadata["source_query_tag"] == "TECHNOLOGY"
            assert metadata["source_query_symbol"] == "AAPL"
    
    def test_add_article_chunks_missing_optional_fields(self):
        """Test adding article chunks with missing optional metadata fields."""
        # Sample data with missing optional fields
        article_url_hash = "test_article_missing_fields"
        chunk_texts = ["This is a test chunk."]
        chunk_vectors = [
            [0.3] * self.chroma_manager.embedding_dimension
        ]
        article_data = {}  # Empty article data
        
        # Add article chunks
        success = self.chroma_manager.add_article_chunks(
            article_url_hash,
            chunk_texts,
            chunk_vectors,
            article_data
        )
        
        # Check if addition was successful
        assert success is True
        
        # Query to verify that chunks were added correctly
        query_embedding = [0.3] * self.chroma_manager.embedding_dimension
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=1,
            filter_metadata={"article_url_hash": article_url_hash}
        )
        
        # Verify results
        assert len(results) == 1
        metadata = results[0]["metadata"]
        
        # Check required metadata fields
        assert metadata["article_url_hash"] == article_url_hash
        assert "chunk_index" in metadata
        
        # Check that optional fields are not present
        assert "published_at_timestamp" not in metadata
        assert "source_query_tag" not in metadata
        assert "source_query_symbol" not in metadata
    
    def test_add_article_chunks_invalid_published_at(self):
        """Test adding article chunks with invalid published_at format."""
        # Sample data with invalid published_at
        article_url_hash = "test_article_invalid_date"
        chunk_texts = ["This is a test chunk."]
        chunk_vectors = [
            [0.4] * self.chroma_manager.embedding_dimension
        ]
        article_data = {
            "published_at": "invalid-date-format",
            "source_query_tag": "FINANCE"
        }
        
        # Add article chunks
        success = self.chroma_manager.add_article_chunks(
            article_url_hash,
            chunk_texts,
            chunk_vectors,
            article_data
        )
        
        # Check if addition was successful despite invalid date
        assert success is True
        
        # Query to verify that chunks were added correctly
        query_embedding = [0.4] * self.chroma_manager.embedding_dimension
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=1,
            filter_metadata={"article_url_hash": article_url_hash}
        )
        
        # Verify results
        assert len(results) == 1
        metadata = results[0]["metadata"]
        
        # Check required metadata fields
        assert metadata["article_url_hash"] == article_url_hash
        assert "chunk_index" in metadata
        
        # Check that invalid date wasn't converted to timestamp
        assert "published_at_timestamp" not in metadata
        assert metadata["source_query_tag"] == "FINANCE"
