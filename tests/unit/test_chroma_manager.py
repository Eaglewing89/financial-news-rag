"""
Unit tests for ChromaDB Manager.

Tests the ChromaDBManager class which handles vector database operations
including initialization, adding/retrieving embeddings, querying, and management.
All tests use in-memory ChromaDB instances for fast, isolated testing.
"""

import pytest

from financial_news_rag.chroma_manager import ChromaDBManager
from tests.fixtures.sample_data import (
    create_test_article_with_hash,
    create_test_normalized_embedding
)


class TestChromaDBManagerInitialization:
    """Test suite for ChromaDBManager initialization and configuration."""
    
    def test_initialization(self, chroma_manager):
        """Test successful initialization and collection creation."""
        assert chroma_manager.collection_name.startswith("test_collection_")
        assert chroma_manager.embedding_dimension == 768
        
        # Check if collection was created and is empty
        collection_status = chroma_manager.get_collection_status()
        assert collection_status["collection_name"] == chroma_manager.collection_name
        assert collection_status["is_empty"] is True
        assert collection_status["total_chunks"] == 0
        assert collection_status["unique_articles"] == 0
    
    def test_initialization_required_parameters(self, temp_directory):
        """Test that initialization requires the mandatory parameters."""
        # Test that initialization fails without persist_directory
        with pytest.raises(TypeError):
            ChromaDBManager(
                collection_name="test_collection",
                embedding_dimension=768
            )
        
        # Test that initialization fails without collection_name
        with pytest.raises(TypeError):
            ChromaDBManager(
                persist_directory=temp_directory,
                embedding_dimension=768
            )
        
        # Test that initialization fails without embedding_dimension
        with pytest.raises(TypeError):
            ChromaDBManager(
                persist_directory=temp_directory,
                collection_name="test_collection"
            )


class TestChromaDBManagerQueryOperations:
    """Test suite for ChromaDB query and retrieval operations."""
    
    @pytest.fixture
    def sample_article_data(self):
        """Create sample article data for testing."""
        return create_test_article_with_hash(
            title="Test Financial Article",
            symbols=["AAPL.US"],
            tags=["TECHNOLOGY"]
        )
    
    def test_query_embeddings_with_similarity_score(self, chroma_manager, sample_article_data):
        """Test querying embeddings with similarity score calculation."""
        # Set up test data
        article_url_hash = sample_article_data["url_hash"]
        chunk_texts = ["This is test chunk 0 for the article."]
        
        # Create a normalized random vector for the embedding
        embedding = create_test_normalized_embedding(768)
        chunk_vector = embedding.tolist()
        
        article_data = {
            "published_at": sample_article_data["published_at"],
            "source_query_tag": "TECHNOLOGY"
        }
        
        # Add article chunk using add_article_chunks
        chroma_manager.add_article_chunks(
            article_url_hash,
            chunk_texts,
            [chunk_vector],
            article_data
        )
        
        # Query with return_similarity_score=True
        results_with_score = chroma_manager.query_embeddings(
            query_embedding=chunk_vector,
            n_results=2,
            return_similarity_score=True
        )
        
        # Check results
        assert len(results_with_score) == 1
        result = results_with_score[0]
        assert result["chunk_id"] == f"{article_url_hash}_0"
        assert "similarity_score" in result
        assert "distance" not in result
        assert result["similarity_score"] > 0.99  # Should be very close to 1
        
        # Query with return_similarity_score=False (default)
        results_without_score = chroma_manager.query_embeddings(
            query_embedding=chunk_vector,
            n_results=2
        )
        
        # Check results
        assert len(results_without_score) == 1
        result = results_without_score[0]
        assert result["chunk_id"] == f"{article_url_hash}_0"
        assert "distance" in result
        assert "similarity_score" not in result
        assert result["distance"] < 0.001  # Should be very close to 0
    
    def test_query_with_filter(self, chroma_manager):
        """Test querying with metadata filters."""
        # Set up test data for first article
        first_article = create_test_article_with_hash(symbol="AAPL.US", tag="TECHNOLOGY")
        first_chunk_texts = ["This is test chunk 0 for the article."]
        first_chunk_vector = create_test_normalized_embedding(768).tolist()
        first_article_data = {
            "published_at": first_article["published_at"],
            "source_query_tag": "TECHNOLOGY"
        }
        
        # Add first article chunk
        chroma_manager.add_article_chunks(
            first_article["url_hash"],
            first_chunk_texts,
            [first_chunk_vector],
            first_article_data
        )
        
        # Set up test data for second article
        second_article = create_test_article_with_hash(symbol="MSFT.US", tag="FINANCE")
        second_chunk_texts = ["This is test chunk 0 for another article."]
        second_chunk_vector = create_test_normalized_embedding(768).tolist()
        second_article_data = {
            "published_at": second_article["published_at"],
            "source_query_tag": "FINANCE"
        }
        
        # Add second article chunk
        chroma_manager.add_article_chunks(
            second_article["url_hash"],
            second_chunk_texts,
            [second_chunk_vector],
            second_article_data
        )
        
        # Create a query embedding
        query_embedding = create_test_normalized_embedding(768).tolist()
        
        # Query with filter for first article
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=5,
            filter_metadata={"article_url_hash": first_article["url_hash"]}
        )
        
        # Check results
        assert len(results) == 1  # Should return 1 result from first article
        assert results[0]["metadata"]["article_url_hash"] == first_article["url_hash"]
        
        # Query with filter for tag
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=5,
            filter_metadata={"source_query_tag": "FINANCE"}
        )
        
        # Check results
        assert len(results) == 1  # Should return 1 result with FINANCE tag
        assert results[0]["metadata"]["source_query_tag"] == "FINANCE"


class TestChromaDBManagerCollectionManagement:
    """Test suite for ChromaDB collection management operations."""
    
    def test_get_collection_status_empty(self, chroma_manager):
        """Test getting collection status for empty collection."""
        status = chroma_manager.get_collection_status()
        assert status["total_chunks"] == 0
        assert status["unique_articles"] == 0
        assert status["is_empty"] is True
    
    def test_get_collection_status_with_data(self, chroma_manager):
        """Test getting collection status after adding data."""
        # Add chunk using add_article_chunks
        article = create_test_article_with_hash()
        chunk_texts = ["This is test chunk 0 for the article."]
        chunk_vector = create_test_normalized_embedding(768).tolist()
        article_data = {
            "published_at": article["published_at"],
            "source_query_tag": "TECHNOLOGY"
        }
        
        chroma_manager.add_article_chunks(
            article["url_hash"],
            chunk_texts,
            [chunk_vector],
            article_data
        )
        
        # Check status after adding
        status = chroma_manager.get_collection_status()
        assert status["total_chunks"] == 1
        assert status["unique_articles"] == 1
        assert status["is_empty"] is False
    
    def test_delete_embeddings_by_article(self, chroma_manager):
        """Test deleting embeddings for a specific article."""
        # Add first article's chunks
        first_article = create_test_article_with_hash(symbol="AAPL.US")
        first_chunk_texts = ["This is test chunk 0 for the article."]
        first_chunk_vector = create_test_normalized_embedding(768).tolist()
        first_article_data = {
            "published_at": first_article["published_at"],
            "source_query_tag": "TECHNOLOGY"
        }
        
        chroma_manager.add_article_chunks(
            first_article["url_hash"],
            first_chunk_texts,
            [first_chunk_vector],
            first_article_data
        )
        
        # Add second article's chunks
        second_article = create_test_article_with_hash(symbol="MSFT.US")
        second_chunk_texts = ["This is test chunk 0 for another article."]
        second_chunk_vector = create_test_normalized_embedding(768).tolist()
        second_article_data = {
            "published_at": second_article["published_at"],
            "source_query_tag": "FINANCE"
        }
        
        chroma_manager.add_article_chunks(
            second_article["url_hash"],
            second_chunk_texts,
            [second_chunk_vector],
            second_article_data
        )
        
        # Check total before deletion
        status_before = chroma_manager.get_collection_status()
        assert status_before["total_chunks"] == 2
        assert status_before["unique_articles"] == 2
        
        # Delete first article's embeddings
        success = chroma_manager.delete_embeddings_by_article(first_article["url_hash"])
        assert success is True
        
        # Check status after deletion
        status_after = chroma_manager.get_collection_status()
        assert status_after["total_chunks"] == 1
        assert status_after["unique_articles"] == 1
        
        # Verify only the correct embeddings were deleted
        query_embedding = create_test_normalized_embedding(768).tolist()
        
        # Query for deleted article
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=10,
            filter_metadata={"article_url_hash": first_article["url_hash"]}
        )
        assert len(results) == 0
        
        # Query for remaining article
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=10,
            filter_metadata={"article_url_hash": second_article["url_hash"]}
        )
        assert len(results) == 1


class TestChromaDBManagerArticleChunks:
    """Test suite for ChromaDB article chunk management operations."""
    
    def test_add_article_chunks_success(self, chroma_manager):
        """Test adding article chunks with the new method."""
        # Sample data
        article = create_test_article_with_hash(symbol="AAPL.US", tag="TECHNOLOGY")
        chunk_texts = ["This is the first chunk.", "This is the second chunk."]
        chunk_vectors = [
            create_test_normalized_embedding(768).tolist(),
            create_test_normalized_embedding(768).tolist()
        ]
        article_data = {
            "published_at": article["published_at"],
            "source_query_tag": "TECHNOLOGY",
            "source_query_symbol": "AAPL"
        }
        
        # Add article chunks
        success = chroma_manager.add_article_chunks(
            article["url_hash"],
            chunk_texts,
            chunk_vectors,
            article_data
        )
        
        # Check if addition was successful
        assert success is True
        
        # Query to verify that chunks were added correctly with all metadata
        query_embedding = chunk_vectors[0]
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=2,
            filter_metadata={"article_url_hash": article["url_hash"]}
        )
        
        # Verify results
        assert len(results) == 2
        
        # Verify IDs
        chunk_ids = [result["chunk_id"] for result in results]
        assert f"{article['url_hash']}_0" in chunk_ids
        assert f"{article['url_hash']}_1" in chunk_ids
        
        # Verify texts
        texts = [result["text"] for result in results]
        assert chunk_texts[0] in texts
        assert chunk_texts[1] in texts
        
        # Verify metadata
        for result in results:
            metadata = result["metadata"]
            # Check required metadata fields
            assert metadata["article_url_hash"] == article["url_hash"]
            assert "chunk_index" in metadata
            
            # Check optional metadata fields
            assert "published_at_timestamp" in metadata  # Should be converted from ISO string
            assert metadata["source_query_tag"] == "TECHNOLOGY"
            assert metadata["source_query_symbol"] == "AAPL"
    
    def test_add_article_chunks_missing_optional_fields(self, chroma_manager):
        """Test adding article chunks with missing optional metadata fields."""
        # Sample data with missing optional fields
        article = create_test_article_with_hash()
        chunk_texts = ["This is a test chunk."]
        chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        article_data = {}  # Empty article data
        
        # Add article chunks
        success = chroma_manager.add_article_chunks(
            article["url_hash"],
            chunk_texts,
            chunk_vectors,
            article_data
        )
        
        # Check if addition was successful
        assert success is True
        
        # Query to verify that chunks were added correctly
        query_embedding = chunk_vectors[0]
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=1,
            filter_metadata={"article_url_hash": article["url_hash"]}
        )
        
        # Verify results
        assert len(results) == 1
        metadata = results[0]["metadata"]
        
        # Check required metadata fields
        assert metadata["article_url_hash"] == article["url_hash"]
        assert "chunk_index" in metadata
        
        # Check that optional fields are not present
        assert "published_at_timestamp" not in metadata
        assert "source_query_tag" not in metadata
        assert "source_query_symbol" not in metadata
    
    def test_add_article_chunks_invalid_published_at(self, chroma_manager):
        """Test adding article chunks with invalid published_at format."""
        # Sample data with invalid published_at
        article = create_test_article_with_hash()
        chunk_texts = ["This is a test chunk."]
        chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        article_data = {
            "published_at": "invalid-date-format",
            "source_query_tag": "FINANCE"
        }
        
        # Add article chunks
        success = chroma_manager.add_article_chunks(
            article["url_hash"],
            chunk_texts,
            chunk_vectors,
            article_data
        )
        
        # Check if addition was successful despite invalid date
        assert success is True
        
        # Query to verify that chunks were added correctly
        query_embedding = chunk_vectors[0]
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=1,
            filter_metadata={"article_url_hash": article["url_hash"]}
        )
        
        # Verify results
        assert len(results) == 1
        metadata = results[0]["metadata"]
        
        # Check required metadata fields
        assert metadata["article_url_hash"] == article["url_hash"]
        assert "chunk_index" in metadata
        
        # Check that invalid date wasn't converted to timestamp
        assert "published_at_timestamp" not in metadata
        assert metadata["source_query_tag"] == "FINANCE"
    
    def test_handling_duplicate_chunk_ids_with_article_chunks(self, chroma_manager):
        """Test handling of duplicate chunk IDs with add_article_chunks."""
        # Sample data for first addition
        article = create_test_article_with_hash()
        chunk_texts = ["This is the original chunk."]
        chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        article_data = {
            "published_at": article["published_at"],
            "source_query_tag": "TECHNOLOGY"
        }
        
        # First addition
        chroma_manager.add_article_chunks(
            article["url_hash"],
            chunk_texts,
            chunk_vectors,
            article_data
        )
        
        # Check initial status
        status_before = chroma_manager.get_collection_status()
        assert status_before["total_chunks"] == 1
        
        # Sample data for update with same chunk ID
        updated_chunk_texts = ["This is the updated chunk."]
        updated_chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        updated_article_data = {
            "published_at": article["published_at"],
            "source_query_tag": "UPDATED_TAG"
        }
        
        # Update with new data (will use same chunk ID format article_url_hash_0)
        chroma_manager.add_article_chunks(
            article["url_hash"],
            updated_chunk_texts,
            updated_chunk_vectors,
            updated_article_data
        )
        
        # Check that count hasn't changed (upsert should replace)
        status_after = chroma_manager.get_collection_status()
        assert status_after["total_chunks"] == 1
        
        # Query to verify update was applied
        query_embedding = updated_chunk_vectors[0]
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=1
        )
        
        # Check that updated content was stored
        assert len(results) == 1
        assert results[0]["chunk_id"] == f"{article['url_hash']}_0"
        assert results[0]["text"] == updated_chunk_texts[0]
        assert results[0]["metadata"]["source_query_tag"] == "UPDATED_TAG"
