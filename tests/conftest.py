"""
Shared pytest fixtures and configuration for the Financial News RAG test suite.

This module provides reusable fixtures for database setup, mocking components,
and creating test data that can be used across all test modules.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import the components we need to create fixtures for
from financial_news_rag.article_manager import ArticleManager
from financial_news_rag.chroma_manager import ChromaDBManager
from financial_news_rag.config import Config
from financial_news_rag.embeddings import EmbeddingsGenerator
from financial_news_rag.eodhd import EODHDClient
from financial_news_rag.orchestrator import FinancialNewsRAG
from financial_news_rag.reranker import ReRanker
from financial_news_rag.text_processor import TextProcessor

# =============================================================================
# Test Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration with environment variables set."""
    with patch.dict(
        os.environ,
        {
            "EODHD_API_KEY": "test_eodhd_api_key",
            "GEMINI_API_KEY": "test_gemini_api_key",
            "DATABASE_PATH_OVERRIDE": ":memory:",  # Use in-memory SQLite for tests
            "CHROMA_DEFAULT_PERSIST_DIRECTORY_OVERRIDE": "/tmp/test_chroma",
            "CHROMA_DEFAULT_COLLECTION_NAME_OVERRIDE": "test_financial_news_embeddings",
            "EMBEDDINGS_DEFAULT_MODEL": "text-embedding-004",
            "EMBEDDINGS_MODEL_DIMENSIONS": '{"text-embedding-004": 768}',
        },
    ):
        yield Config()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(temp_directory):
    """Create a temporary database path."""
    return os.path.join(temp_directory, "test_financial_news.db")


@pytest.fixture
def article_manager(temp_db_path):
    """Create an ArticleManager instance with a temporary database."""
    manager = ArticleManager(db_path=temp_db_path)
    yield manager
    # Cleanup
    manager.close_connection()


@pytest.fixture
def chroma_manager(temp_directory):
    """Create a ChromaDBManager instance with temporary storage."""
    collection_name = f"test_collection_{id(temp_directory)}"
    manager = ChromaDBManager(
        persist_directory=temp_directory,
        collection_name=collection_name,
        embedding_dimension=768,
        in_memory=True,  # Use in-memory for faster tests
    )
    yield manager
    # Cleanup happens automatically with in-memory mode


# =============================================================================
# Mock Fixtures for Unit Tests
# =============================================================================


@pytest.fixture
def mock_eodhd_client():
    """Create a mock EODHDClient for unit tests."""
    mock = MagicMock(spec=EODHDClient)

    # Default successful response
    mock.fetch_news.return_value = {
        "articles": [
            {
                "title": "Test Article 1",
                "published_at": "2023-01-01T12:00:00Z",
                "raw_content": "Test content 1",
                "url": "https://example.com/article1",
                "source_query_tag": "TECHNOLOGY",
            },
            {
                "title": "Test Article 2",
                "published_at": "2023-01-02T12:00:00Z",
                "raw_content": "Test content 2",
                "url": "https://example.com/article2",
                "source_query_tag": "TECHNOLOGY",
            },
        ],
        "status_code": 200,
        "success": True,
    }

    return mock


@pytest.fixture
def mock_text_processor():
    """Create a mock TextProcessor for unit tests."""
    mock = MagicMock(spec=TextProcessor)

    # Default successful processing
    mock.clean_content.return_value = "Cleaned content"
    mock.chunk_text.return_value = ["Chunk 1", "Chunk 2"]

    return mock


@pytest.fixture
def mock_embeddings_generator():
    """Create a mock EmbeddingsGenerator for unit tests."""
    mock = MagicMock(spec=EmbeddingsGenerator)

    # Default properties
    mock.embedding_dim = 768
    mock.model_name = "text-embedding-004"

    # Default embedding generation
    mock.generate_embeddings.return_value = [
        np.random.rand(768).tolist() for _ in range(2)
    ]

    return mock


@pytest.fixture
def mock_chroma_manager():
    """Create a mock ChromaDBManager for unit tests."""
    mock = MagicMock(spec=ChromaDBManager)

    # Default successful operations
    mock.add_article_chunks.return_value = True
    mock.query_embeddings.return_value = [
        {
            "chunk_id": "test_article_0",
            "text": "Test chunk content",
            "metadata": {"article_url_hash": "test_hash", "chunk_index": 0},
            "distance": 0.5,
        }
    ]
    mock.get_collection_status.return_value = {
        "total_chunks": 10,
        "unique_articles": 5,
        "is_empty": False,
        "collection_name": "test_collection",
    }

    return mock


@pytest.fixture
def mock_reranker():
    """Create a mock ReRanker for unit tests."""
    mock = MagicMock(spec=ReRanker)

    # Default reranking behavior (returns input unchanged)
    mock.rerank.side_effect = lambda query, results, **kwargs: results

    return mock


@pytest.fixture
def mock_article_manager():
    """Create a mock ArticleManager for unit tests."""
    mock = MagicMock(spec=ArticleManager)

    # Default successful operations
    mock.store_articles.return_value = 2
    mock.get_articles_by_processing_status.return_value = []
    mock.get_processed_articles_for_embedding.return_value = []
    mock.update_article_processing_status.return_value = True
    mock.update_article_embedding_status.return_value = True
    mock.get_database_statistics.return_value = {
        "total_articles": 100,
        "text_processing_status": {"PENDING": 50, "SUCCESS": 40, "FAILED": 10},
        "embedding_status": {"PENDING": 60, "SUCCESS": 30, "FAILED": 10},
        "articles_by_tag": {"TECHNOLOGY": 30, "FINANCE": 25},
        "articles_by_symbol": {"AAPL.US": 15, "MSFT.US": 10},
        "date_range": {
            "earliest_article": "2023-01-01T00:00:00Z",
            "latest_article": "2023-12-31T23:59:59Z",
        },
        "api_calls": {"total_articles_retrieved": 60},
    }

    return mock


# =============================================================================
# Orchestrator Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator_components():
    """Create all mocked components for the orchestrator."""
    with patch("financial_news_rag.orchestrator.EODHDClient") as mock_eodhd, patch(
        "financial_news_rag.orchestrator.ArticleManager"
    ) as mock_article_mgr, patch(
        "financial_news_rag.orchestrator.TextProcessor"
    ) as mock_text_proc, patch(
        "financial_news_rag.orchestrator.EmbeddingsGenerator"
    ) as mock_embeddings, patch(
        "financial_news_rag.orchestrator.ChromaDBManager"
    ) as mock_chroma, patch(
        "financial_news_rag.orchestrator.ReRanker"
    ) as mock_reranker:

        # Configure mocks
        mock_embeddings_instance = mock_embeddings.return_value
        mock_embeddings_instance.embedding_dim = 768
        mock_embeddings_instance.model_name = "text-embedding-004"

        yield {
            "eodhd": mock_eodhd,
            "article_manager": mock_article_mgr,
            "text_processor": mock_text_proc,
            "embeddings": mock_embeddings,
            "chroma": mock_chroma,
            "reranker": mock_reranker,
            "embeddings_instance": mock_embeddings_instance,
        }


@pytest.fixture
def orchestrator_with_mocks(mock_orchestrator_components, test_config):
    """Create a FinancialNewsRAG instance with all dependencies mocked."""
    # Set environment variables for the config
    with patch.dict(
        os.environ,
        {
            "EODHD_API_KEY": "test_eodhd_api_key",
            "GEMINI_API_KEY": "test_gemini_api_key",
        },
    ):
        orchestrator = FinancialNewsRAG()

        # Attach mocks for easy access in tests
        orchestrator._test_mocks = mock_orchestrator_components

        return orchestrator


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_article():
    """Create a sample article for testing."""
    return {
        "title": "Sample Test Article",
        "raw_content": "<p>This is a sample article with <b>HTML</b> tags.</p>",
        "url": "https://example.com/sample-article",
        "published_at": "2023-05-18T12:00:00+00:00",
        "source_api": "EODHD",
        "symbols": ["AAPL.US", "MSFT.US"],
        "tags": ["TECHNOLOGY", "EARNINGS"],
        "sentiment": {"polarity": 0.5, "neg": 0.1, "neu": 0.5, "pos": 0.4},
    }


@pytest.fixture
def sample_articles_list():
    """Create a list of sample articles for testing."""
    return [
        {
            "title": "Tech Article 1",
            "raw_content": "<p>Technology news content</p>",
            "url": "https://example.com/tech-1",
            "published_at": "2023-01-01T12:00:00Z",
            "source_api": "EODHD",
            "symbols": ["AAPL.US"],
            "tags": ["TECHNOLOGY"],
            "source_query_tag": "TECHNOLOGY",
        },
        {
            "title": "Finance Article 1",
            "raw_content": "<p>Financial news content</p>",
            "url": "https://example.com/finance-1",
            "published_at": "2023-01-02T12:00:00Z",
            "source_api": "EODHD",
            "symbols": ["MSFT.US"],
            "tags": ["FINANCE"],
            "source_query_symbol": "MSFT.US",
        },
    ]


@pytest.fixture
def sample_processed_chunks():
    """Create sample processed text chunks for testing."""
    return [
        "This is the first chunk of processed content.",
        "This is the second chunk of processed content.",
        "This is the third chunk with different content.",
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return [
        np.random.rand(768).tolist(),
        np.random.rand(768).tolist(),
        np.random.rand(768).tolist(),
    ]


@pytest.fixture
def sample_chroma_results():
    """Create sample ChromaDB query results for testing."""
    return [
        {
            "chunk_id": "article_hash_1_0",
            "text": "First chunk content",
            "metadata": {
                "article_url_hash": "article_hash_1",
                "chunk_index": 0,
                "published_at_timestamp": 1672574400,
                "source_query_tag": "TECHNOLOGY",
            },
            "distance": 0.3,
        },
        {
            "chunk_id": "article_hash_2_0",
            "text": "Second chunk content",
            "metadata": {
                "article_url_hash": "article_hash_2",
                "chunk_index": 0,
                "published_at_timestamp": 1672660800,
                "source_query_symbol": "AAPL.US",
            },
            "distance": 0.5,
        },
    ]


# =============================================================================
# Test Data Fixtures for Text Processing
# =============================================================================


@pytest.fixture
def sample_html_content():
    """Provide sample HTML content for testing text cleaning."""
    return {
        "with_basic_tags": "<p>This is a <b>test</b> article</p>",
        "with_complex_tags": '<div class="content"><p>Content with <a href="link">link</a></p></div>',
        "with_boilerplate": "This is an article. Click here to read more.",
        "with_financial_data": "<p>AAPL stock price is $150.50, up 2.5% from yesterday.</p>",
        "realistic_article": """
        <div class="article">
            <p>This is the first paragraph with <b>bold text</b>.</p>
            <p>This is the second paragraph with <a href="link">a link</a>.</p>
            <p>This contains financial data: AAPL $150.50 (+2.5%).</p>
            <p>Click here to read more about this topic.</p>
        </div>
        """,
    }


@pytest.fixture
def long_test_sentences():
    """Provide a list of test sentences for chunking tests."""
    return [f"This is test sentence number {i}." for i in range(100)]


@pytest.fixture
def financial_test_sentences():
    """Provide financial-specific test sentences."""
    return [
        "First sentence about financial markets.",
        "Second sentence discusses market volatility.",
        "Third sentence covers investment strategies.",
        "Fourth sentence analyzes economic trends.",
    ]


@pytest.fixture
def sample_financial_article():
    """Provide a realistic financial article with HTML content."""
    return """
    <p>Apple Inc. (NASDAQ:AAPL) shares gained 3.2% in pre-market trading following the company's Q4 2023 earnings report.</p>
    
    <p>The tech giant reported revenue of $89.5 billion, slightly below the consensus estimate of $89.9 billion but representing steady performance in a challenging economic environment.</p>
    
    <p>iPhone revenue came in at $43.8 billion, down 3% year-over-year but better than feared amid concerns about consumer spending on premium devices.</p>
    
    <p>The Services segment continued its strong growth trajectory, posting revenue of $22.3 billion, up 16% from the prior year period. This includes revenue from the App Store, Apple Music, and iCloud services.</p>
    
    <p>CEO Tim Cook noted during the earnings call that the company sees "continued strength in emerging markets" and expects Services growth to remain robust.</p>
    
    <p>Click here to read the full earnings report. Source: Apple Inc. Investor Relations</p>
    """


# =============================================================================
# Parameterized Test Data
# =============================================================================


@pytest.fixture(params=["TECHNOLOGY", "FINANCE", "EARNINGS"])
def tag_parameter(request):
    """Parameterized fixture for testing different tags."""
    return request.param


@pytest.fixture(params=["AAPL.US", "MSFT.US", "GOOGL.US"])
def symbol_parameter(request):
    """Parameterized fixture for testing different symbols."""
    return request.param


@pytest.fixture(params=["SUCCESS", "FAILED", "PENDING"])
def status_parameter(request):
    """Parameterized fixture for testing different processing statuses."""
    return request.param


# =============================================================================
# Utility Functions for Tests
# =============================================================================


@pytest.fixture
def assert_article_stored():
    """Helper function to assert an article was stored correctly."""

    def _assert_article_stored(article_manager, article, expected_count=1):
        """Assert that an article was stored correctly in the database."""
        from financial_news_rag.utils import generate_url_hash

        url_hash = generate_url_hash(article["url"])
        stored_article = article_manager.get_article_by_hash(url_hash)

        assert stored_article is not None
        assert stored_article["title"] == article["title"]
        assert stored_article["url"] == article["url"]
        assert "url_hash" in stored_article

        return stored_article

    return _assert_article_stored


@pytest.fixture
def assert_chunks_in_chroma():
    """Helper function to assert chunks were added to ChromaDB correctly."""

    def _assert_chunks_in_chroma(chroma_manager, article_hash, expected_chunks):
        """Assert that chunks were added to ChromaDB correctly."""
        # Query for all chunks for this article
        query_embedding = np.random.rand(768).tolist()
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=100,
            filter_metadata={"article_url_hash": article_hash},
        )

        assert len(results) == len(expected_chunks)

        # Check that all chunk texts are present
        result_texts = {result["text"] for result in results}
        expected_texts = set(expected_chunks)
        assert result_texts == expected_texts

        return results

    return _assert_chunks_in_chroma


# =============================================================================
# Session-level Cleanup
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_environment():
    """Clean up test environment after all tests."""
    yield
    # Any session-level cleanup can go here
    pass
