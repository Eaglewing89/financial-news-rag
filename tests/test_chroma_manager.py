"""
Tests for ChromaDB Manager.

This module tests the ChromaDBManager class, which is responsible for:
- Initializing a ChromaDB client and collection
- Adding and retrieving embeddings
- Querying similar vectors
- Managing associations between ChromaDB and SQLite
"""

import os
import sys
import tempfile
import unittest
from typing import Dict, List, Any
from unittest.mock import MagicMock

import pytest
import numpy as np

# Add project root to path to ensure proper imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_news_rag.chroma_manager import ChromaDBManager


class TestChromaDBManager:
    """Test suite for ChromaDBManager class."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        # Create temporary directory for ChromaDB persistence
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate a unique collection name for each test to avoid state sharing
        self.test_collection_name = f"test_collection_{id(self)}"
        
        # Default embedding dimension for testing
        self.test_embedding_dimension = 768
        
        # Create ChromaDBManager with in-memory mode for faster testing
        self.chroma_manager = ChromaDBManager(
            persist_directory=self.temp_dir.name,
            collection_name=self.test_collection_name,
            embedding_dimension=self.test_embedding_dimension,
            in_memory=True  # Use in-memory mode for tests
        )
        
        # Sample article hash for testing
        self.sample_article_hash = "test_article_123"
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Delete temporary directory
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test successful initialization and collection creation."""
        # Check if ChromaDBManager was initialized correctly
        assert self.chroma_manager.collection_name == self.test_collection_name
        assert self.chroma_manager.embedding_dimension == self.test_embedding_dimension
        
        # Check if collection was created
        collection_status = self.chroma_manager.get_collection_status()
        assert collection_status["collection_name"] == self.test_collection_name
        assert collection_status["is_empty"] is True
    
    def test_initialization_required_parameters(self):
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
                persist_directory=self.temp_dir.name,
                embedding_dimension=self.test_embedding_dimension
            )
        
        # Test that initialization fails without embedding_dimension
        with pytest.raises(TypeError):
            ChromaDBManager(
                persist_directory=self.temp_dir.name,
                collection_name=self.test_collection_name
            )
    
    # Tests for add_embeddings method have been removed since the method was removed from the implementation
    
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
        
        # Check that published_at_timestamp is not present due to invalid format
        assert "published_at_timestamp" not in metadata
        # But other valid metadata should be present
        assert metadata["source_query_tag"] == "FINANCE"
    
    def test_add_article_chunks_empty_lists(self):
        """Test adding article chunks with empty chunk lists."""
        article_url_hash = "test_article_empty_chunks"
        chunk_texts = []
        chunk_vectors = []
        article_data = {"published_at": "2023-05-15T14:30:00Z"}
        
        # Add article chunks
        success = self.chroma_manager.add_article_chunks(
            article_url_hash,
            chunk_texts,
            chunk_vectors,
            article_data
        )
        
        # Should return False for empty lists
        assert success is False
    
    def test_add_article_chunks_exception_handling(self):
        """Test exception handling in add_article_chunks method."""
        article_url_hash = "test_article_exception"
        chunk_texts = ["This is a test chunk."]
        chunk_vectors = [
            [0.5] * self.chroma_manager.embedding_dimension
        ]
        article_data = {"published_at": "2023-05-15T14:30:00Z"}
        
        # Mock the collection.upsert method to raise an exception
        original_upsert = self.chroma_manager.collection.upsert
        self.chroma_manager.collection.upsert = MagicMock(side_effect=Exception("Test exception"))
        
        try:
            # Add article chunks (should handle the exception)
            success = self.chroma_manager.add_article_chunks(
                article_url_hash,
                chunk_texts,
                chunk_vectors,
                article_data
            )
            
            # Should return False when exception occurs
            assert success is False
        finally:
            # Restore the original upsert method
            self.chroma_manager.collection.upsert = original_upsert
    
    def test_get_article_hashes_by_date_range_both_timestamps(self):
        """Test get_article_hashes_by_date_range with both timestamp parameters."""
        # Add article chunks with specific published_at timestamps
        # First article: January 1, 2023
        first_article_hash = "article_jan_2023"
        first_timestamp = 1672531200  # 2023-01-01 00:00:00 UTC
        self.chroma_manager.add_article_chunks(
            first_article_hash,
            ["January 2023 article content"],
            [[0.1] * self.chroma_manager.embedding_dimension],
            {"published_at": "2023-01-01T00:00:00Z"}
        )
        
        # Second article: March 15, 2023
        second_article_hash = "article_mar_2023"
        second_timestamp = 1678838400  # 2023-03-15 00:00:00 UTC
        self.chroma_manager.add_article_chunks(
            second_article_hash,
            ["March 2023 article content"],
            [[0.2] * self.chroma_manager.embedding_dimension],
            {"published_at": "2023-03-15T00:00:00Z"}
        )
        
        # Third article: June 30, 2023
        third_article_hash = "article_jun_2023"
        third_timestamp = 1688083200  # 2023-06-30 00:00:00 UTC
        self.chroma_manager.add_article_chunks(
            third_article_hash,
            ["June 2023 article content"],
            [[0.3] * self.chroma_manager.embedding_dimension],
            {"published_at": "2023-06-30T00:00:00Z"}
        )
        
        # Query for articles between February 1 and June 1
        newer_than = 1675209600  # 2023-02-01 00:00:00 UTC
        older_than = 1685577600  # 2023-06-01 00:00:00 UTC
        
        results = self.chroma_manager.get_article_hashes_by_date_range(
            older_than_timestamp=older_than,
            newer_than_timestamp=newer_than
        )
        
        # Should return only the March article
        assert len(results) == 1
        assert second_article_hash in results
        assert first_article_hash not in results
        assert third_article_hash not in results
    
    def test_get_article_hashes_by_date_range_older_than_only(self):
        """Test get_article_hashes_by_date_range with only older_than_timestamp."""
        # We can reuse the setup from the previous test
        # Add article chunks with specific published_at timestamps if they don't exist
        if self.chroma_manager.get_collection_status()["total_chunks"] == 0:
            # First article: January 1, 2023
            first_article_hash = "article_jan_2023"
            self.chroma_manager.add_article_chunks(
                first_article_hash,
                ["January 2023 article content"],
                [[0.1] * self.chroma_manager.embedding_dimension],
                {"published_at": "2023-01-01T00:00:00Z"}
            )
            
            # Second article: March 15, 2023
            second_article_hash = "article_mar_2023"
            self.chroma_manager.add_article_chunks(
                second_article_hash,
                ["March 2023 article content"],
                [[0.2] * self.chroma_manager.embedding_dimension],
                {"published_at": "2023-03-15T00:00:00Z"}
            )
            
            # Third article: June 30, 2023
            third_article_hash = "article_jun_2023"
            self.chroma_manager.add_article_chunks(
                third_article_hash,
                ["June 2023 article content"],
                [[0.3] * self.chroma_manager.embedding_dimension],
                {"published_at": "2023-06-30T00:00:00Z"}
            )
        
        # Query for articles older than or equal to April 1
        older_than = 1680307200  # 2023-04-01 00:00:00 UTC
        
        results = self.chroma_manager.get_article_hashes_by_date_range(
            older_than_timestamp=older_than
        )
        
        # Should return January and March articles
        assert len(results) == 2
        assert "article_jan_2023" in results
        assert "article_mar_2023" in results
        assert "article_jun_2023" not in results
    
    def test_get_article_hashes_by_date_range_newer_than_only(self):
        """Test get_article_hashes_by_date_range with only newer_than_timestamp."""
        # We can reuse the setup from the previous test
        # Add article chunks with specific published_at timestamps if they don't exist
        if self.chroma_manager.get_collection_status()["total_chunks"] == 0:
            # First article: January 1, 2023
            first_article_hash = "article_jan_2023"
            self.chroma_manager.add_article_chunks(
                first_article_hash,
                ["January 2023 article content"],
                [[0.1] * self.chroma_manager.embedding_dimension],
                {"published_at": "2023-01-01T00:00:00Z"}
            )
            
            # Second article: March 15, 2023
            second_article_hash = "article_mar_2023"
            self.chroma_manager.add_article_chunks(
                second_article_hash,
                ["March 2023 article content"],
                [[0.2] * self.chroma_manager.embedding_dimension],
                {"published_at": "2023-03-15T00:00:00Z"}
            )
            
            # Third article: June 30, 2023
            third_article_hash = "article_jun_2023"
            self.chroma_manager.add_article_chunks(
                third_article_hash,
                ["June 2023 article content"],
                [[0.3] * self.chroma_manager.embedding_dimension],
                {"published_at": "2023-06-30T00:00:00Z"}
            )
        
        # Query for articles newer than or equal to March 1
        newer_than = 1677628800  # 2023-03-01 00:00:00 UTC
        
        results = self.chroma_manager.get_article_hashes_by_date_range(
            newer_than_timestamp=newer_than
        )
        
        # Should return March and June articles
        assert len(results) == 2
        assert "article_jan_2023" not in results
        assert "article_mar_2023" in results
        assert "article_jun_2023" in results
    
    def test_get_article_hashes_by_date_range_no_matching_results(self):
        """Test get_article_hashes_by_date_range with a range that yields no results."""
        # We can reuse the setup from the previous tests
        # Add article chunks with specific published_at timestamps if they don't exist
        if self.chroma_manager.get_collection_status()["total_chunks"] == 0:
            # First article: January 1, 2023
            first_article_hash = "article_jan_2023"
            self.chroma_manager.add_article_chunks(
                first_article_hash,
                ["January 2023 article content"],
                [[0.1] * self.chroma_manager.embedding_dimension],
                {"published_at": "2023-01-01T00:00:00Z"}
            )
            
            # Second article: March 15, 2023
            second_article_hash = "article_mar_2023"
            self.chroma_manager.add_article_chunks(
                second_article_hash,
                ["March 2023 article content"],
                [[0.2] * self.chroma_manager.embedding_dimension],
                {"published_at": "2023-03-15T00:00:00Z"}
            )
            
            # Third article: June 30, 2023
            third_article_hash = "article_jun_2023"
            self.chroma_manager.add_article_chunks(
                third_article_hash,
                ["June 2023 article content"],
                [[0.3] * self.chroma_manager.embedding_dimension],
                {"published_at": "2023-06-30T00:00:00Z"}
            )
        
        # Query for articles between July and August 2023
        newer_than = 1688169600  # 2023-07-01 00:00:00 UTC
        older_than = 1693526400  # 2023-09-01 00:00:00 UTC
        
        results = self.chroma_manager.get_article_hashes_by_date_range(
            older_than_timestamp=older_than,
            newer_than_timestamp=newer_than
        )
        
        # Should return empty list
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_get_article_hashes_by_date_range_multiple_chunks_per_article(self):
        """Test get_article_hashes_by_date_range with multiple chunks per article."""
        # Create an article with multiple chunks
        multi_chunk_hash = "article_multi_chunk"
        multi_chunk_texts = ["This is chunk 1", "This is chunk 2", "This is chunk 3"]
        multi_chunk_vectors = [
            [0.5] * self.chroma_manager.embedding_dimension,
            [0.6] * self.chroma_manager.embedding_dimension,
            [0.7] * self.chroma_manager.embedding_dimension
        ]
        
        self.chroma_manager.add_article_chunks(
            multi_chunk_hash,
            multi_chunk_texts,
            multi_chunk_vectors,
            {"published_at": "2023-05-01T00:00:00Z"}  # May 1, 2023
        )
        
        # Query for articles in May 2023
        newer_than = 1682899200  # 2023-05-01 00:00:00 UTC
        older_than = 1685577600  # 2023-06-01 00:00:00 UTC
        
        results = self.chroma_manager.get_article_hashes_by_date_range(
            older_than_timestamp=older_than,
            newer_than_timestamp=newer_than
        )
        
        # Should return exactly one unique article hash even though it has multiple chunks
        assert len(results) == 1
        assert multi_chunk_hash in results
    
    def test_get_article_hashes_by_date_range_no_params(self):
        """Test get_article_hashes_by_date_range with no timestamp parameters."""
        # Call method with no parameters
        results = self.chroma_manager.get_article_hashes_by_date_range()
        
        # Should return empty list and log a warning
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_get_article_hashes_by_date_range_exception_handling(self):
        """Test exception handling in get_article_hashes_by_date_range method."""
        # Mock the collection.get method to raise an exception
        original_get = self.chroma_manager.collection.get
        self.chroma_manager.collection.get = MagicMock(side_effect=Exception("Test exception"))
        
        try:
            # Call method with valid parameters that would otherwise succeed
            results = self.chroma_manager.get_article_hashes_by_date_range(
                older_than_timestamp=1685577600  # 2023-06-01 00:00:00 UTC
            )
            
            # Should return empty list when exception occurs
            assert isinstance(results, list)
            assert len(results) == 0
        finally:
            # Restore the original get method
            self.chroma_manager.collection.get = original_get
    
    def test_query_embeddings_with_date_filters(self):
        """Test querying embeddings with date string filters."""
        # Set up test data for different dates
        # Creating articles with different published dates
        
        # First article: 2023-01-15
        first_article_hash = "test_article_jan15_2023"
        first_chunk_texts = ["This is test chunk for January 15, 2023."]
        first_chunk_vector = np.random.rand(768).tolist()
        first_article_data = {
            "published_at": "2023-01-15T10:30:00Z",
            "source_query_tag": "FINANCE"
        }
        
        # Second article: 2023-05-20
        second_article_hash = "test_article_may20_2023"
        second_chunk_texts = ["This is test chunk for May 20, 2023."]
        second_chunk_vector = np.random.rand(768).tolist()
        second_article_data = {
            "published_at": "2023-05-20T14:45:00Z",
            "source_query_tag": "TECHNOLOGY"
        }
        
        # Third article: 2023-09-10
        third_article_hash = "test_article_sep10_2023"
        third_chunk_texts = ["This is test chunk for September 10, 2023."]
        third_chunk_vector = np.random.rand(768).tolist()
        third_article_data = {
            "published_at": "2023-09-10T09:15:00Z",
            "source_query_tag": "HEALTH"
        }
        
        # Add articles to ChromaDB
        self.chroma_manager.add_article_chunks(
            first_article_hash,
            first_chunk_texts,
            [first_chunk_vector],
            first_article_data
        )
        
        self.chroma_manager.add_article_chunks(
            second_article_hash,
            second_chunk_texts,
            [second_chunk_vector],
            second_article_data
        )
        
        self.chroma_manager.add_article_chunks(
            third_article_hash,
            third_chunk_texts,
            [third_chunk_vector],
            third_article_data
        )
        
        # Create a query embedding
        query_embedding = np.random.rand(768)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Test 1: Query with from_date_str only
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=10,
            from_date_str="2023-03-01"  # Should return May and September articles
        )
        
        # Verify results
        assert len(results) == 2
        article_hashes = {result["metadata"]["article_url_hash"] for result in results}
        assert second_article_hash in article_hashes
        assert third_article_hash in article_hashes
        assert first_article_hash not in article_hashes
        
        # Test 2: Query with to_date_str only
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=10,
            to_date_str="2023-06-01"  # Should return January and May articles
        )
        
        # Verify results
        assert len(results) == 2
        article_hashes = {result["metadata"]["article_url_hash"] for result in results}
        assert first_article_hash in article_hashes
        assert second_article_hash in article_hashes
        assert third_article_hash not in article_hashes
        
        # Test 3: Query with both from_date_str and to_date_str
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=10,
            from_date_str="2023-01-20",
            to_date_str="2023-08-01"  # Should return only May article
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0]["metadata"]["article_url_hash"] == second_article_hash
    
    def test_query_embeddings_with_invalid_date_strings(self):
        """Test querying embeddings with invalid date strings."""
        # Set up test data
        article_hash = self.sample_article_hash
        chunk_texts = ["This is test chunk for date string testing."]
        chunk_vector = np.random.rand(768).tolist()
        article_data = {
            "published_at": "2023-05-15T14:30:00Z",
            "source_query_tag": "TECHNOLOGY"
        }
        
        # Add article to ChromaDB
        self.chroma_manager.add_article_chunks(
            article_hash,
            chunk_texts,
            [chunk_vector],
            article_data
        )
        
        # Create a query embedding
        query_embedding = np.random.rand(768)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Test with invalid from_date_str
        # Test logs a warning instead of emitting a UserWarning, so we don't need to capture it
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=5,
            from_date_str="invalid-date",
            to_date_str="2023-12-31"
        )
            
        # Verify results (should use only the valid to_date_str)
        assert len(results) == 1
        assert results[0]["metadata"]["article_url_hash"] == article_hash
    
    def test_query_embeddings_with_filter_metadata_and_date_strings(self):
        """Test querying embeddings with both filter_metadata and date strings."""
        # Set up test data for different tags and dates
        
        # Finance article from January
        finance_article_hash = "test_finance_jan_2023"
        finance_chunk_texts = ["This is a finance article from January 2023."]
        finance_chunk_vector = np.random.rand(768).tolist()
        finance_article_data = {
            "published_at": "2023-01-15T10:30:00Z",
            "source_query_tag": "FINANCE"
        }
        
        # Technology article from May
        tech_article_hash = "test_tech_may_2023"
        tech_chunk_texts = ["This is a technology article from May 2023."]
        tech_chunk_vector = np.random.rand(768).tolist()
        tech_article_data = {
            "published_at": "2023-05-20T14:45:00Z",
            "source_query_tag": "TECHNOLOGY"
        }
        
        # Finance article from September
        finance2_article_hash = "test_finance_sep_2023"
        finance2_chunk_texts = ["This is another finance article from September 2023."]
        finance2_chunk_vector = np.random.rand(768).tolist()
        finance2_article_data = {
            "published_at": "2023-09-10T09:15:00Z",
            "source_query_tag": "FINANCE"
        }
        
        # Add articles to ChromaDB
        self.chroma_manager.add_article_chunks(
            finance_article_hash,
            finance_chunk_texts,
            [finance_chunk_vector],
            finance_article_data
        )
        
        self.chroma_manager.add_article_chunks(
            tech_article_hash,
            tech_chunk_texts,
            [tech_chunk_vector],
            tech_article_data
        )
        
        self.chroma_manager.add_article_chunks(
            finance2_article_hash,
            finance2_chunk_texts,
            [finance2_chunk_vector],
            finance2_article_data
        )
        
        # Create a query embedding
        query_embedding = np.random.rand(768)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Test: Query with both filter_metadata (by tag) and date filters
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=10,
            filter_metadata={"source_query_tag": "FINANCE"},
            from_date_str="2023-03-01"  # Should return only the September finance article
        )
        
        # Verify results
        assert len(results) == 1
        assert results[0]["metadata"]["article_url_hash"] == finance2_article_hash
        assert results[0]["metadata"]["source_query_tag"] == "FINANCE"
