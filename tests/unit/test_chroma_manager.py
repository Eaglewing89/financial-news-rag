"""
Unit tests for ChromaDB Manager.

Tests the ChromaDBManager class which handles vector database operations
including initialization, adding/retrieving embeddings, querying, and management.
All tests use in-memory ChromaDB instances for fast, isolated testing.

Each test class focuses on a specific area of functionality:
- Initialization and configuration
- Query and retrieval operations
- Collection management
- Date range filtering
- Error handling
- Article chunks operations
"""


# Third-party imports
import pytest

# Project imports
from financial_news_rag.chroma_manager import ChromaDBManager
from tests.fixtures.sample_data import (
    create_test_article_with_hash,
    create_test_normalized_embedding,
)


class TestChromaDBManagerInitialization:
    """Test suite for ChromaDBManager initialization and configuration."""

    def test_initialization(self, chroma_manager):
        """Test successful initialization and collection creation."""
        # Verify manager properties
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
        # Test missing persist_directory
        with pytest.raises(TypeError):
            ChromaDBManager(collection_name="test_collection", embedding_dimension=768)

        # Test missing collection_name
        with pytest.raises(TypeError):
            ChromaDBManager(persist_directory=temp_directory, embedding_dimension=768)

        # Test missing embedding_dimension
        with pytest.raises(TypeError):
            ChromaDBManager(
                persist_directory=temp_directory, collection_name="test_collection"
            )


class TestChromaDBManagerQueryOperations:
    """Test suite for ChromaDB query and retrieval operations."""

    @pytest.fixture
    def sample_article_data(self):
        """Create sample article data for testing."""
        return create_test_article_with_hash(
            title="Test Financial Article", symbols=["AAPL.US"], tags=["TECHNOLOGY"]
        )

    def test_query_embeddings_with_similarity_score(
        self, chroma_manager, sample_article_data
    ):
        """Test querying embeddings with similarity score calculation."""
        # Set up test data
        article_url_hash = sample_article_data["url_hash"]
        chunk_texts = ["This is test chunk 0 for the article."]

        # Create a normalized random vector for the embedding
        embedding = create_test_normalized_embedding(768)
        chunk_vector = embedding.tolist()

        article_data = {
            "published_at": sample_article_data["published_at"],
            "source_query_tag": "TECHNOLOGY",
        }

        # Add article chunk using add_article_chunks
        chroma_manager.add_article_chunks(
            article_url_hash, chunk_texts, [chunk_vector], article_data
        )

        # Query with return_similarity_score=True
        results_with_score = chroma_manager.query_embeddings(
            query_embedding=chunk_vector, n_results=2, return_similarity_score=True
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
            query_embedding=chunk_vector, n_results=2
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
        first_article = create_test_article_with_hash(
            symbol="AAPL.US", tag="TECHNOLOGY"
        )
        first_chunk_texts = ["This is test chunk 0 for the article."]
        first_chunk_vector = create_test_normalized_embedding(768).tolist()
        first_article_data = {
            "published_at": first_article["published_at"],
            "source_query_tag": "TECHNOLOGY",
        }

        # Add first article chunk
        chroma_manager.add_article_chunks(
            first_article["url_hash"],
            first_chunk_texts,
            [first_chunk_vector],
            first_article_data,
        )

        # Set up test data for second article
        second_article = create_test_article_with_hash(symbol="MSFT.US", tag="FINANCE")
        second_chunk_texts = ["This is test chunk 0 for another article."]
        second_chunk_vector = create_test_normalized_embedding(768).tolist()
        second_article_data = {
            "published_at": second_article["published_at"],
            "source_query_tag": "FINANCE",
        }

        # Add second article chunk
        chroma_manager.add_article_chunks(
            second_article["url_hash"],
            second_chunk_texts,
            [second_chunk_vector],
            second_article_data,
        )

        # Create a query embedding
        query_embedding = create_test_normalized_embedding(768).tolist()

        # Query with filter for first article
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=5,
            filter_metadata={"article_url_hash": first_article["url_hash"]},
        )

        # Check results
        assert len(results) == 1  # Should return 1 result from first article
        assert results[0]["metadata"]["article_url_hash"] == first_article["url_hash"]

        # Query with filter for tag
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=5,
            filter_metadata={"source_query_tag": "FINANCE"},
        )

        # Check results
        assert len(results) == 1  # Should return 1 result with FINANCE tag
        assert results[0]["metadata"]["source_query_tag"] == "FINANCE"


class TestChromaDBManagerCollectionManagement:
    """Test suite for ChromaDB collection management operations."""

    @pytest.fixture
    def isolated_chroma_manager(self, temp_directory):
        """Create an isolated ChromaDBManager instance for tests that need complete isolation."""
        isolated_manager = ChromaDBManager(
            persist_directory=temp_directory,
            collection_name=f"isolated_collection_mgmt_{id(temp_directory)}",
            embedding_dimension=768,
            in_memory=True,
        )
        return isolated_manager

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
            "source_query_tag": "TECHNOLOGY",
        }

        chroma_manager.add_article_chunks(
            article["url_hash"], chunk_texts, [chunk_vector], article_data
        )

        # Check status after adding
        status = chroma_manager.get_collection_status()
        assert status["total_chunks"] == 1
        assert status["unique_articles"] == 1
        assert status["is_empty"] is False

    def test_delete_embeddings_by_article(self, isolated_chroma_manager):
        """Test deleting embeddings for a specific article."""
        # Add first article's chunks
        first_article = create_test_article_with_hash(symbol="AAPL.US")
        first_chunk_texts = ["This is test chunk 0 for the article."]
        first_chunk_vector = create_test_normalized_embedding(768).tolist()
        first_article_data = {
            "published_at": first_article["published_at"],
            "source_query_tag": "TECHNOLOGY",
        }

        isolated_chroma_manager.add_article_chunks(
            first_article["url_hash"],
            first_chunk_texts,
            [first_chunk_vector],
            first_article_data,
        )

        # Add second article's chunks
        second_article = create_test_article_with_hash(symbol="MSFT.US")
        second_chunk_texts = ["This is test chunk 0 for another article."]
        second_chunk_vector = create_test_normalized_embedding(768).tolist()
        second_article_data = {
            "published_at": second_article["published_at"],
            "source_query_tag": "FINANCE",
        }

        isolated_chroma_manager.add_article_chunks(
            second_article["url_hash"],
            second_chunk_texts,
            [second_chunk_vector],
            second_article_data,
        )

        # Check total before deletion
        status_before = isolated_chroma_manager.get_collection_status()
        assert status_before["total_chunks"] == 2
        assert status_before["unique_articles"] == 2

        # Delete first article's embeddings
        success = isolated_chroma_manager.delete_embeddings_by_article(
            first_article["url_hash"]
        )
        assert success is True

        # Check status after deletion
        status_after = isolated_chroma_manager.get_collection_status()
        assert status_after["total_chunks"] == 1
        assert status_after["unique_articles"] == 1

        # Verify only the correct embeddings were deleted
        query_embedding = create_test_normalized_embedding(768).tolist()

        # Query for deleted article
        results = isolated_chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=10,
            filter_metadata={"article_url_hash": first_article["url_hash"]},
        )
        assert len(results) == 0

        # Query for remaining article
        results = isolated_chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=10,
            filter_metadata={"article_url_hash": second_article["url_hash"]},
        )
        assert len(results) == 1

    def test_get_collection_status_comprehensive(self, chroma_manager):
        """Test comprehensive collection status information."""
        # Test with multiple articles
        articles = [
            create_test_article_with_hash(symbol="AAPL.US"),
            create_test_article_with_hash(symbol="MSFT.US"),
            create_test_article_with_hash(symbol="GOOGL.US"),
        ]

        # Add multiple chunks per article
        for i, article in enumerate(articles):
            chunk_texts = [f"Chunk 0 for article {i}", f"Chunk 1 for article {i}"]
            chunk_vectors = [
                create_test_normalized_embedding(768).tolist(),
                create_test_normalized_embedding(768).tolist(),
            ]
            article_data = {
                "published_at": article["published_at"],
                "source_query_tag": "TECHNOLOGY",
            }

            chroma_manager.add_article_chunks(
                article["url_hash"], chunk_texts, chunk_vectors, article_data
            )

        # Check comprehensive status
        status = chroma_manager.get_collection_status()

        assert status["total_chunks"] == 6  # 3 articles × 2 chunks each
        assert status["unique_articles"] == 3
        assert status["is_empty"] is False
        assert status["embedding_dimension"] == 768
        assert status["collection_name"] == chroma_manager.collection_name
        assert status["persist_directory"] == chroma_manager.persist_directory


class TestChromaDBManagerDateRangeOperations:
    """Test suite for ChromaDB date range filtering operations."""

    @pytest.fixture
    def sample_articles_with_dates(self):
        """Create sample articles with different dates."""
        return [
            create_test_article_with_hash(symbol="AAPL.US"),  # article1 - older
            create_test_article_with_hash(symbol="MSFT.US"),  # article2 - middle
            create_test_article_with_hash(symbol="GOOGL.US"),  # article3 - newer
        ]

    @pytest.fixture
    def sample_dates(self):
        """Create sample dates in ISO format."""
        return [
            "2024-01-15T10:00:00Z",  # article1 - older
            "2024-02-15T10:00:00Z",  # article2 - middle
            "2024-03-15T10:00:00Z",  # article3 - newer
        ]

    @pytest.fixture
    def populated_chroma_manager(
        self, chroma_manager, sample_articles_with_dates, sample_dates
    ):
        """Populate chroma_manager with sample articles with different dates."""
        for i, (article, date) in enumerate(
            zip(sample_articles_with_dates, sample_dates)
        ):
            chunk_texts = [f"Test chunk for article {i}"]
            chunk_vectors = [create_test_normalized_embedding(768).tolist()]
            article_data = {"published_at": date, "source_query_tag": "TECHNOLOGY"}

            chroma_manager.add_article_chunks(
                article["url_hash"], chunk_texts, chunk_vectors, article_data
            )

        return chroma_manager, sample_articles_with_dates

    def test_query_embeddings_with_date_filters(self, populated_chroma_manager):
        """Test querying embeddings with from_date_str and to_date_str filters."""
        chroma_manager, articles = populated_chroma_manager
        query_embedding = create_test_normalized_embedding(768).tolist()

        # Test from_date_str filter (should get articles from Feb 15 onwards)
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding, n_results=10, from_date_str="2024-02-15"
        )
        assert len(results) == 2  # Should get article2 and article3

        # Test to_date_str filter (should get articles up to Feb 15)
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=10,
            to_date_str="2024-02-15T23:59:59Z",
        )
        assert len(results) == 2  # Should get article1 and article2

        # Test both from_date_str and to_date_str (should get only article2)
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=10,
            from_date_str="2024-02-10",
            to_date_str="2024-02-20",
        )
        assert len(results) == 1  # Should get only article2
        assert results[0]["metadata"]["article_url_hash"] == articles[1]["url_hash"]

    def test_query_embeddings_invalid_date_filters(self, chroma_manager):
        """Test querying with invalid date filters."""
        # Close any existing connections to ensure a clean slate
        chroma_manager.close_connection()

        # Reinitialize the client and collection
        chroma_manager._initialize_client_and_collection()

        # Add test data
        article = create_test_article_with_hash()
        chunk_texts = ["Test chunk"]
        chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        article_data = {
            "published_at": "2024-01-15T10:00:00Z",
            "source_query_tag": "TECHNOLOGY",
        }

        chroma_manager.add_article_chunks(
            article["url_hash"], chunk_texts, chunk_vectors, article_data
        )

        query_embedding = create_test_normalized_embedding(768).tolist()

        # Test with invalid from_date_str
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=10,
            from_date_str="invalid-date-format",
        )
        # Should still return results, ignoring invalid filter
        assert len(results) == 1

        # Test with invalid to_date_str
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding, n_results=10, to_date_str="not-a-date"
        )
        # Should still return results, ignoring invalid filter
        assert len(results) == 1

    def test_get_article_hashes_by_date_range_both_timestamps(
        self, populated_chroma_manager
    ):
        """Test getting article hashes with both older_than and newer_than timestamps."""
        chroma_manager, articles = populated_chroma_manager

        # Define timestamps that should include all articles (Jan 15 to Mar 15)
        newer_than = 1705309200  # Jan 15, 2024 timestamp
        older_than = 1710496800  # Mar 15, 2024 timestamp

        result_hashes = chroma_manager.get_article_hashes_by_date_range(
            older_than_timestamp=older_than, newer_than_timestamp=newer_than
        )

        # Should return all three articles (all within the range)
        assert len(result_hashes) == 3
        assert articles[0]["url_hash"] in result_hashes
        assert articles[1]["url_hash"] in result_hashes
        assert articles[2]["url_hash"] in result_hashes

    @pytest.fixture
    def isolated_chroma_manager(self, temp_directory):
        """Create an isolated ChromaDBManager instance for tests that need complete isolation."""
        isolated_manager = ChromaDBManager(
            persist_directory=temp_directory,
            collection_name="isolated_test_collection",
            embedding_dimension=768,
            in_memory=True,
        )
        return isolated_manager

    def test_get_article_hashes_by_date_range_single_timestamp(
        self, isolated_chroma_manager
    ):
        """Test getting article hashes with only one timestamp boundary."""
        # Set up test data
        article1 = create_test_article_with_hash(symbol="AAPL.US")
        article2 = create_test_article_with_hash(symbol="MSFT.US")

        dates = ["2024-01-15T10:00:00Z", "2024-03-15T10:00:00Z"]

        for i, (article, date) in enumerate(zip([article1, article2], dates)):
            chunk_texts = [f"Test chunk for article {i}"]
            chunk_vectors = [create_test_normalized_embedding(768).tolist()]
            article_data = {"published_at": date, "source_query_tag": "TECHNOLOGY"}

            isolated_chroma_manager.add_article_chunks(
                article["url_hash"], chunk_texts, chunk_vectors, article_data
            )

        # Test with only older_than timestamp (Feb 15)
        older_than = 1708077600  # Feb 15, 2024 timestamp
        result_hashes = isolated_chroma_manager.get_article_hashes_by_date_range(
            older_than_timestamp=older_than
        )

        # Should return only article1 (published before Feb 15)
        assert len(result_hashes) == 1
        assert article1["url_hash"] in result_hashes

        # Test with only newer_than timestamp (Feb 15)
        newer_than = 1708077600  # Feb 15, 2024 timestamp
        result_hashes = isolated_chroma_manager.get_article_hashes_by_date_range(
            newer_than_timestamp=newer_than
        )

        # Should return only article2 (published after Feb 15)
        assert len(result_hashes) == 1
        assert article2["url_hash"] in result_hashes

    def test_get_article_hashes_by_date_range_no_criteria(self, chroma_manager):
        """Test getting article hashes with no timestamp criteria."""
        result_hashes = chroma_manager.get_article_hashes_by_date_range()

        # Should return empty list when no criteria provided
        assert result_hashes == []

    def test_get_article_hashes_by_date_range_no_matches(self, chroma_manager):
        """Test getting article hashes when no articles match the date range."""
        # Add test data
        article = create_test_article_with_hash()
        chunk_texts = ["Test chunk"]
        chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        article_data = {
            "published_at": "2024-01-15T10:00:00Z",
            "source_query_tag": "TECHNOLOGY",
        }

        chroma_manager.add_article_chunks(
            article["url_hash"], chunk_texts, chunk_vectors, article_data
        )

        # Query for articles much older than what we have
        older_than = 1640995200  # Jan 1, 2022 timestamp (much older)
        result_hashes = chroma_manager.get_article_hashes_by_date_range(
            older_than_timestamp=older_than
        )

        # Should return empty list
        assert result_hashes == []

    def test_get_article_hashes_by_date_range_articles_without_timestamps(
        self, chroma_manager
    ):
        """Test getting article hashes when articles don't have published_at_timestamp."""
        # First, close any existing connections to ensure a clean slate
        chroma_manager.close_connection()

        # Reinitialize the client and collection
        chroma_manager._initialize_client_and_collection()

        # Add article without published_at (no timestamp in metadata)
        article = create_test_article_with_hash()
        chunk_texts = ["Test chunk"]
        chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        article_data = {}  # No published_at

        chroma_manager.add_article_chunks(
            article["url_hash"], chunk_texts, chunk_vectors, article_data
        )

        # Query for articles with timestamp filter
        newer_than = 1640995200  # Jan 1, 2022 timestamp
        result_hashes = chroma_manager.get_article_hashes_by_date_range(
            newer_than_timestamp=newer_than
        )

        # Should return empty list (article has no timestamp to match)
        assert result_hashes == []


class TestChromaDBManagerErrorHandling:
    """Test suite for ChromaDB error handling scenarios."""

    def test_query_embeddings_error_handling(self, chroma_manager):
        """Test query_embeddings error handling."""
        # Test with invalid query embedding (wrong dimension)
        invalid_embedding = [0.1, 0.2]  # Wrong dimension (should be 768)

        results = chroma_manager.query_embeddings(
            query_embedding=invalid_embedding, n_results=5
        )

        # Should return empty list on error
        assert results == []

    def test_delete_embeddings_by_article_nonexistent(self, chroma_manager):
        """Test deleting embeddings for non-existent article."""
        # Try to delete embeddings for an article that doesn't exist
        success = chroma_manager.delete_embeddings_by_article("nonexistent_hash")

        # Should still return True (ChromaDB doesn't error on deleting non-existent items)
        assert success is True

    def test_add_article_chunks_empty_data(self, chroma_manager):
        """Test adding article chunks with empty data."""
        # Test with empty chunk lists
        success = chroma_manager.add_article_chunks(
            "test_hash",
            [],  # Empty chunk texts
            [],  # Empty chunk vectors
            {},  # Empty article data
        )

        # Should return False for empty data
        assert success is False

    def test_add_article_chunks_mismatched_lengths(self, chroma_manager):
        """Test adding article chunks with mismatched text and vector lengths."""
        # This should be handled gracefully by zip() function
        success = chroma_manager.add_article_chunks(
            "test_hash",
            ["chunk1", "chunk2"],  # 2 texts
            [create_test_normalized_embedding(768).tolist()],  # 1 vector
            {},
        )

        # Should succeed but only process the first chunk (zip stops at shortest)
        assert success is True

        # Verify only one chunk was added
        status = chroma_manager.get_collection_status()
        assert status["total_chunks"] == 1


class TestChromaDBManagerInitializationErrors:
    """Test suite for ChromaDB initialization error scenarios."""

    def test_initialization_with_invalid_directory(self, temp_directory):
        """Test initialization with invalid persist directory."""
        import os

        # Create a file where we want a directory (this should cause issues)
        invalid_path = os.path.join(temp_directory, "invalid_file")
        with open(invalid_path, "w") as f:
            f.write("This is a file, not a directory")

        # Try to initialize with the file path as persist_directory
        # This should succeed because ChromaDB can handle this case
        manager = ChromaDBManager(
            persist_directory=invalid_path,
            collection_name="test_collection",
            embedding_dimension=768,
            in_memory=True,  # Use in_memory to avoid actual file operations
        )

        assert manager is not None

    def test_initialization_persistent_mode(self, temp_directory):
        """Test initialization in persistent mode."""
        # Test creating a manager in persistent mode
        manager = ChromaDBManager(
            persist_directory=temp_directory,
            collection_name="test_persistent_collection",
            embedding_dimension=768,
            in_memory=False,
        )

        assert manager is not None
        assert manager.in_memory is False
        assert manager.persist_directory == temp_directory
        assert manager.collection_name == "test_persistent_collection"
        assert manager.embedding_dimension == 768

    def test_initialization_existing_collection(self, temp_directory):
        """Test initialization with existing collection."""
        collection_name = "test_existing_collection"

        # Create first manager to establish collection
        manager1 = ChromaDBManager(
            persist_directory=temp_directory,
            collection_name=collection_name,
            embedding_dimension=768,
            in_memory=False,
        )

        # Add some data
        article = create_test_article_with_hash()
        chunk_texts = ["Test chunk"]
        chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        article_data = {"source_query_tag": "TECHNOLOGY"}

        manager1.add_article_chunks(
            article["url_hash"], chunk_texts, chunk_vectors, article_data
        )

        # Create second manager with same collection name - should connect to existing
        manager2 = ChromaDBManager(
            persist_directory=temp_directory,
            collection_name=collection_name,
            embedding_dimension=768,
            in_memory=False,
        )

        # Should see the existing data
        status = manager2.get_collection_status()
        assert status["total_chunks"] == 1
        assert status["is_empty"] is False


class TestChromaDBManagerArticleChunks:
    """Test suite for ChromaDB article chunk management operations."""

    @pytest.fixture
    def isolated_chroma_manager(self, temp_directory):
        """Create an isolated ChromaDBManager instance for tests that need complete isolation."""
        isolated_manager = ChromaDBManager(
            persist_directory=temp_directory,
            collection_name=f"isolated_article_chunks_{id(temp_directory)}",
            embedding_dimension=768,
            in_memory=True,
        )
        return isolated_manager

    def test_add_article_chunks_success(self, chroma_manager):
        """Test adding article chunks with the new method."""
        # Sample data
        article = create_test_article_with_hash(symbol="AAPL.US", tag="TECHNOLOGY")
        chunk_texts = ["This is the first chunk.", "This is the second chunk."]
        chunk_vectors = [
            create_test_normalized_embedding(768).tolist(),
            create_test_normalized_embedding(768).tolist(),
        ]
        article_data = {
            "published_at": article["published_at"],
            "source_query_tag": "TECHNOLOGY",
            "source_query_symbol": "AAPL",
        }

        # Add article chunks
        success = chroma_manager.add_article_chunks(
            article["url_hash"], chunk_texts, chunk_vectors, article_data
        )

        # Check if addition was successful
        assert success is True

        # Query to verify that chunks were added correctly with all metadata
        query_embedding = chunk_vectors[0]
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=2,
            filter_metadata={"article_url_hash": article["url_hash"]},
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
            assert (
                "published_at_timestamp" in metadata
            )  # Should be converted from ISO string
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
            article["url_hash"], chunk_texts, chunk_vectors, article_data
        )

        # Check if addition was successful
        assert success is True

        # Query to verify that chunks were added correctly
        query_embedding = chunk_vectors[0]
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=1,
            filter_metadata={"article_url_hash": article["url_hash"]},
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
            "source_query_tag": "FINANCE",
        }

        # Add article chunks
        success = chroma_manager.add_article_chunks(
            article["url_hash"], chunk_texts, chunk_vectors, article_data
        )

        # Check if addition was successful despite invalid date
        assert success is True

        # Query to verify that chunks were added correctly
        query_embedding = chunk_vectors[0]
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=1,
            filter_metadata={"article_url_hash": article["url_hash"]},
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

    def test_handling_duplicate_chunk_ids_with_article_chunks(
        self, isolated_chroma_manager
    ):
        """Test handling of duplicate chunk IDs with add_article_chunks."""
        # Sample data for first addition
        article = create_test_article_with_hash()
        chunk_texts = ["This is the original chunk."]
        chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        article_data = {
            "published_at": article["published_at"],
            "source_query_tag": "TECHNOLOGY",
        }

        # First addition
        isolated_chroma_manager.add_article_chunks(
            article["url_hash"], chunk_texts, chunk_vectors, article_data
        )

        # Check initial status
        status_before = isolated_chroma_manager.get_collection_status()
        assert status_before["total_chunks"] == 1

        # Sample data for update with same chunk ID
        updated_chunk_texts = ["This is the updated chunk."]
        updated_chunk_vectors = [create_test_normalized_embedding(768).tolist()]
        updated_article_data = {
            "published_at": article["published_at"],
            "source_query_tag": "UPDATED_TAG",
        }

        # Update with new data (will use same chunk ID format article_url_hash_0)
        isolated_chroma_manager.add_article_chunks(
            article["url_hash"],
            updated_chunk_texts,
            updated_chunk_vectors,
            updated_article_data,
        )

        # Check that count hasn't changed (upsert should replace)
        status_after = isolated_chroma_manager.get_collection_status()
        assert status_after["total_chunks"] == 1

        # Query to verify update was applied
        query_embedding = updated_chunk_vectors[0]
        results = isolated_chroma_manager.query_embeddings(
            query_embedding=query_embedding, n_results=1
        )

        # Check that updated content was stored
        assert len(results) == 1
        assert results[0]["chunk_id"] == f"{article['url_hash']}_0"
        assert results[0]["text"] == updated_chunk_texts[0]
        assert results[0]["metadata"]["source_query_tag"] == "UPDATED_TAG"
