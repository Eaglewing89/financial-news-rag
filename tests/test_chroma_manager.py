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
        
        # Create ChromaDBManager with in-memory mode for faster testing
        self.chroma_manager = ChromaDBManager(
            persist_directory=self.temp_dir.name,
            collection_name=self.test_collection_name,
            in_memory=True  # Use in-memory mode for tests
        )
        
        # Sample article and chunks for testing
        self.sample_article_hash = "test_article_123"
        
        # Create sample embeddings (768 dimensions to match text-embedding-004)
        self.sample_embeddings = []
        for i in range(3):
            # Create a normalized random vector for the embedding
            embedding = np.random.rand(768)
            embedding = embedding / np.linalg.norm(embedding)
            
            self.sample_embeddings.append({
                "chunk_id": f"{self.sample_article_hash}_{i}",
                "embedding": embedding.tolist(),
                "text": f"This is test chunk {i} for the article.",
                "metadata": {
                    "article_url_hash": self.sample_article_hash,
                    "chunk_index": i,
                    "published_at_timestamp": 1621456789,
                    "source_query_tag": "TECHNOLOGY"
                }
            })
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Delete temporary directory
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test successful initialization and collection creation."""
        # Check if ChromaDBManager was initialized correctly
        assert self.chroma_manager.collection_name == self.test_collection_name
        assert self.chroma_manager.embedding_dimension == 768
        
        # Check if collection was created
        collection_status = self.chroma_manager.get_collection_status()
        assert collection_status["collection_name"] == self.test_collection_name
        assert collection_status["is_empty"] is True
    
    def test_add_embeddings_single_article(self):
        """Test adding embeddings for a single article."""
        # Add sample embeddings
        success = self.chroma_manager.add_embeddings(
            self.sample_article_hash,
            self.sample_embeddings
        )
        
        # Check if addition was successful
        assert success is True
        
        # Verify embeddings were added
        collection_status = self.chroma_manager.get_collection_status()
        assert collection_status["total_chunks"] == 3
        assert collection_status["unique_articles"] == 1
    
    def test_add_embeddings_empty_list(self):
        """Test adding an empty list of embeddings."""
        # Try to add empty list
        success = self.chroma_manager.add_embeddings(
            self.sample_article_hash,
            []
        )
        
        # Should return False for empty list
        assert success is False
        
        # Collection should still be empty
        collection_status = self.chroma_manager.get_collection_status()
        assert collection_status["is_empty"] is True
    
    def test_query_embeddings(self):
        """Test querying embeddings."""
        # Add sample embeddings
        self.chroma_manager.add_embeddings(
            self.sample_article_hash,
            self.sample_embeddings
        )
        
        # Create a query embedding (use one of the sample embeddings for testing)
        query_embedding = self.sample_embeddings[0]["embedding"]
        
        # Query without filter
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=2
        )
        
        # Check results
        assert len(results) == 2
        # The first result should be the exact match with distance close to 0
        assert results[0]["chunk_id"] == self.sample_embeddings[0]["chunk_id"]
        assert results[0]["distance"] < 0.001  # Should be very close to 0
    
    def test_query_with_filter(self):
        """Test querying with metadata filters."""
        # Add sample embeddings
        self.chroma_manager.add_embeddings(
            self.sample_article_hash,
            self.sample_embeddings
        )
        
        # Add another article's embeddings with different metadata
        another_article_hash = "test_article_456"
        another_embeddings = []
        for i in range(2):
            embedding = np.random.rand(768)
            embedding = embedding / np.linalg.norm(embedding)
            
            another_embeddings.append({
                "chunk_id": f"{another_article_hash}_{i}",
                "embedding": embedding.tolist(),
                "text": f"This is test chunk {i} for another article.",
                "metadata": {
                    "article_url_hash": another_article_hash,
                    "chunk_index": i,
                    "published_at_timestamp": 1631456789,
                    "source_query_tag": "FINANCE"
                }
            })
        
        self.chroma_manager.add_embeddings(
            another_article_hash,
            another_embeddings
        )
        
        # Create a query embedding
        query_embedding = np.random.rand(768)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Query with filter for first article
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=5,
            filter_metadata={"article_url_hash": self.sample_article_hash}
        )
        
        # Check results
        assert len(results) <= 3  # Should return at most 3 results (all from first article)
        for result in results:
            assert result["metadata"]["article_url_hash"] == self.sample_article_hash
        
        # Query with filter for tag
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=5,
            filter_metadata={"source_query_tag": "FINANCE"}
        )
        
        # Check results
        assert len(results) <= 2  # Should return at most 2 results (all with FINANCE tag)
        for result in results:
            assert result["metadata"]["source_query_tag"] == "FINANCE"
    
    def test_get_collection_status(self):
        """Test getting collection status."""
        # Check empty collection
        status = self.chroma_manager.get_collection_status()
        assert status["total_chunks"] == 0
        assert status["unique_articles"] == 0
        assert status["is_empty"] is True
        
        # Add sample embeddings
        self.chroma_manager.add_embeddings(
            self.sample_article_hash,
            self.sample_embeddings
        )
        
        # Check status after adding
        status = self.chroma_manager.get_collection_status()
        assert status["total_chunks"] == 3
        assert status["unique_articles"] == 1
        assert status["is_empty"] is False
    
    def test_delete_embeddings_by_article(self):
        """Test deleting embeddings for a specific article."""
        # Add sample embeddings
        self.chroma_manager.add_embeddings(
            self.sample_article_hash,
            self.sample_embeddings
        )
        
        # Add another article's embeddings
        another_article_hash = "test_article_456"
        another_embeddings = []
        for i in range(2):
            embedding = np.random.rand(768)
            embedding = embedding / np.linalg.norm(embedding)
            
            another_embeddings.append({
                "chunk_id": f"{another_article_hash}_{i}",
                "embedding": embedding.tolist(),
                "text": f"This is test chunk {i} for another article.",
                "metadata": {
                    "article_url_hash": another_article_hash,
                    "chunk_index": i,
                    "published_at_timestamp": 1631456789,
                }
            })
        
        self.chroma_manager.add_embeddings(
            another_article_hash,
            another_embeddings
        )
        
        # Check total before deletion
        status_before = self.chroma_manager.get_collection_status()
        assert status_before["total_chunks"] == 5
        assert status_before["unique_articles"] == 2
        
        # Delete first article's embeddings
        success = self.chroma_manager.delete_embeddings_by_article(self.sample_article_hash)
        assert success is True
        
        # Check status after deletion
        status_after = self.chroma_manager.get_collection_status()
        assert status_after["total_chunks"] == 2  # 5 - 3 = 2
        assert status_after["unique_articles"] == 1
        
        # Verify only the correct embeddings were deleted
        # Query for deleted article
        query_embedding = np.random.rand(768)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=10,
            filter_metadata={"article_url_hash": self.sample_article_hash}
        )
        
        # Should be empty
        assert len(results) == 0
        
        # Query for remaining article
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding.tolist(),
            n_results=10,
            filter_metadata={"article_url_hash": another_article_hash}
        )
        
        # Should have 2 results
        assert len(results) == 2
    
    def test_handling_duplicate_chunk_ids(self):
        """Test handling of duplicate chunk IDs."""
        # Add initial embeddings
        self.chroma_manager.add_embeddings(
            self.sample_article_hash,
            self.sample_embeddings
        )
        
        # Create a duplicate embedding with same chunk_id but different content
        duplicate_embedding = {
            "chunk_id": self.sample_embeddings[0]["chunk_id"],  # Same ID as existing
            "embedding": np.random.rand(768).tolist(),  # Different embedding
            "text": "This is an updated chunk text.",  # Different text
            "metadata": {
                "article_url_hash": self.sample_article_hash,
                "chunk_index": 0,
                "published_at_timestamp": 1621456789,
                "is_updated": True  # Additional metadata
            }
        }
        
        # Add the duplicate embedding
        self.chroma_manager.add_embeddings(
            self.sample_article_hash,
            [duplicate_embedding]
        )
        
        # Check that the collection count hasn't changed (upsert replaces duplicate)
        status = self.chroma_manager.get_collection_status()
        assert status["total_chunks"] == 3  # Still 3, not 4
        
        # Query to check if the duplicate was replaced
        query_embedding = duplicate_embedding["embedding"]
        
        results = self.chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=1
        )
        
        # Check that the updated content was stored
        assert results[0]["chunk_id"] == duplicate_embedding["chunk_id"]
        assert results[0]["text"] == duplicate_embedding["text"]
        assert results[0]["metadata"].get("is_updated") is True
