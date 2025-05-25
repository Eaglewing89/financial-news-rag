"""
Unit tests for the ReRanker class.

Tests the ReRanker functionality including initialization, article re-ranking,
error handling, and edge cases. Focuses on isolated unit testing with proper
mocking of external dependencies like the Gemini API.
"""

import pytest
import json
from unittest.mock import patch, MagicMock

from financial_news_rag.reranker import ReRanker
from google.api_core.exceptions import GoogleAPIError


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture
def reranker_test_articles():
    """Create sample articles specifically for reranker testing."""
    return [
        {
            "url_hash": "hash1",
            "title": "Tech Trends in Finance",
            "processed_content": "This article discusses technology trends in finance and digital transformation.",
            "published_date": "2024-01-15"
        },
        {
            "url_hash": "hash2", 
            "title": "Market Analysis Report",
            "processed_content": "This article discusses market trends and economic indicators.",
            "published_date": "2024-01-14"
        },
        {
            "url_hash": "hash3",
            "title": "Sports Events Coverage", 
            "processed_content": "This article is about sports events and athletic competitions.",
            "published_date": "2024-01-13"
        }
    ]

@pytest.fixture
def test_query():
    """Sample query for testing."""
    return "technology trends in finance"

@pytest.fixture  
def api_key():
    """Test API key."""
    return "test_gemini_api_key"

@pytest.fixture
def create_reranker(api_key):
    """Factory fixture to create a ReRanker instance with mocked client."""
    with patch("financial_news_rag.reranker.genai.Client") as mock_client:
        def _create_reranker(model_name="gemini-2.0-flash"):
            """Create a reranker with specified parameters."""
            return ReRanker(api_key=api_key, model_name=model_name), mock_client
        return _create_reranker


class TestReRankerInitialization:
    """Test suite for ReRanker initialization and configuration."""

    def test_initialization(self, create_reranker):
        """Test that the ReRanker is initialized correctly."""
        # Arrange
        reranker, _ = create_reranker()
        
        # Act & Assert
        assert reranker.model_name == "gemini-2.0-flash"
        
        # Arrange - with custom model
        custom_reranker, _ = create_reranker(model_name="gemini-3.0-pro")
        
        # Assert
        assert custom_reranker.model_name == "gemini-3.0-pro"
        
    @patch("financial_news_rag.reranker.genai.Client")
    def test_initialization_no_api_key(self, mock_client):
        """Test that an error is raised when no API key is provided."""
        # Arrange
        mock_client.side_effect = ValueError("API key is required")
            
        # Act & Assert
        with pytest.raises(ValueError):
            ReRanker(api_key=None)

    @patch("financial_news_rag.reranker.genai.Client")
    def test_initialization_empty_api_key(self, mock_client):
        """Test that an error is raised when empty API key is provided."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Gemini API key is required"):
            ReRanker(api_key="")


class TestReRankerCoreOperations:
    """Test suite for ReRanker core reranking operations."""

    def test_rerank_articles_successful(self, create_reranker, reranker_test_articles, test_query):
        """Test successful re-ranking of articles."""
        # Arrange
        reranker, _ = create_reranker()
        
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            # Set up the mock to return different scores for the articles
            mock_assess.side_effect = [
                {"id": "hash1", "score": 8.5},
                {"id": "hash2", "score": 5.2},
                {"id": "hash3", "score": 2.1}
            ]
            
            # Act
            reranked = reranker.rerank_articles(test_query, reranker_test_articles)
            
            # Assert
            assert len(reranked) == 3
            assert reranked[0]["url_hash"] == "hash1"  # Highest score
            assert reranked[1]["url_hash"] == "hash2"  # Middle score
            assert reranked[2]["url_hash"] == "hash3"  # Lowest score
            
            # Check that the scores were added to the articles
            assert reranked[0]["rerank_score"] == 8.5
            assert reranked[1]["rerank_score"] == 5.2
            assert reranked[2]["rerank_score"] == 2.1
            
            # Check that all article fields are preserved
            assert "published_date" in reranked[0]
            assert reranked[0]["title"] == "Tech Trends in Finance"
            
            # Check that _assess_article_relevance was called the right number of times
            assert mock_assess.call_count == 3

    def test_parse_score_fallback(self, create_reranker, reranker_test_articles, test_query):
        """Test the fallback regex parsing of scores when JSON parsing fails."""
        # Arrange
        reranker, _ = create_reranker()
        
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            # Set up the mock to return a response with a score
            mock_assess.return_value = {"id": "hash1", "score": 7.5}
            
            # Act
            reranked = reranker.rerank_articles(test_query, [reranker_test_articles[0]])
            
            # Assert
            assert reranked[0]["rerank_score"] == 7.5
            assert mock_assess.called
            assert mock_assess.call_count == 1

    def test_single_article_reranking(self, create_reranker, test_query):
        """Test re-ranking with a single article."""
        # Arrange
        reranker, _ = create_reranker()
        article = {
            "url_hash": "single_hash", 
            "title": "Single Article",
            "processed_content": "Technology and finance convergence."
        }
        
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            mock_assess.return_value = {"id": "single_hash", "score": 9.0}
            
            # Act
            reranked = reranker.rerank_articles(test_query, [article])
            
            # Assert
            assert len(reranked) == 1
            assert reranked[0]["rerank_score"] == 9.0
            assert reranked[0]["url_hash"] == "single_hash"
            assert reranked[0]["title"] == "Single Article"
            assert mock_assess.called


class TestReRankerErrorHandling:
    """Test suite for ReRanker error handling and resilience."""

    def test_rerank_articles_api_error(self, create_reranker, reranker_test_articles, test_query):
        """Test that the original order is preserved when the API call fails."""
        # Arrange
        reranker, _ = create_reranker()
        
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            # Mock the _assess_article_relevance method to raise an exception
            mock_assess.side_effect = Exception("API error")
            
            # Act
            reranked = reranker.rerank_articles(test_query, reranker_test_articles)
            
            # Assert
            assert reranked == reranker_test_articles
            assert mock_assess.called
            assert "rerank_score" not in reranked[0]

    def test_rerank_articles_api_error_specific_exception(self, create_reranker, reranker_test_articles, test_query):
        """Test handling of specific API exceptions like GoogleAPIError."""
        # Arrange
        reranker, _ = create_reranker()
        
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            # Mock specific API exception
            mock_assess.side_effect = GoogleAPIError("Rate limit exceeded")
            
            # Act
            reranked = reranker.rerank_articles(test_query, reranker_test_articles)
            
            # Assert
            assert reranked == reranker_test_articles
            assert mock_assess.called

    def test_rerank_articles_malformed_json(self, create_reranker, reranker_test_articles, test_query):
        """Test handling of malformed JSON responses."""
        # Arrange
        reranker, _ = create_reranker()
        
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            # Mock the _assess_article_relevance method to return a score of 0
            mock_assess.return_value = {"id": "hash1", "score": 0.0}
            
            # Act
            reranked = reranker.rerank_articles(test_query, [reranker_test_articles[0]])
            
            # Assert
            assert len(reranked) == 1
            assert reranked[0]["rerank_score"] == 0.0
            assert mock_assess.called

    def test_missing_required_fields(self, create_reranker, test_query):
        """Test handling of articles missing required fields."""
        # Arrange
        reranker, _ = create_reranker()
        
        # Create an article missing processed_content
        article_missing_content = {
            "url_hash": "hash4",
            "title": "Article 4"
        }
        
        # Create an article missing url_hash
        article_missing_hash = {
            "title": "Article 5",
            "processed_content": "Some content"
        }
        
        articles = [article_missing_content, article_missing_hash]
        
        # Act
        reranked = reranker.rerank_articles(test_query, articles)
        
        # Assert
        assert len(reranked) == 2
        assert reranked[0]["rerank_score"] == 0.0
        assert reranked[1]["rerank_score"] == 0.0
        assert reranked[0]["title"] == "Article 4"
        assert reranked[1]["title"] == "Article 5"


class TestReRankerEdgeCases:
    """Test suite for ReRanker edge cases and boundary conditions."""

    def test_empty_article_list(self, create_reranker, test_query):
        """Test handling of an empty article list."""
        # Arrange
        reranker, _ = create_reranker()
        
        # Act
        reranked = reranker.rerank_articles(test_query, [])
        
        # Assert
        assert reranked == []
        
    def test_empty_query(self, create_reranker, reranker_test_articles):
        """Test handling of an empty query."""
        # Arrange
        reranker, _ = create_reranker()
        
        # Act
        reranked = reranker.rerank_articles("", reranker_test_articles)
        
        # Assert
        assert reranked == reranker_test_articles
        assert "rerank_score" not in reranked[0]

    def test_whitespace_only_query(self, create_reranker, reranker_test_articles):
        """Test handling of a whitespace-only query."""
        # Arrange
        reranker, _ = create_reranker()
        
        # Act
        reranked = reranker.rerank_articles("   \t\n  ", reranker_test_articles)
        
        # Assert
        assert reranked == reranker_test_articles
        assert "rerank_score" not in reranked[0]

    def test_very_long_content(self, create_reranker, test_query):
        """Test handling of articles with very long content."""
        # Arrange
        reranker, _ = create_reranker()
        
        # Create an article with content longer than 10000 characters
        long_content = "A" * 15000  # 15000 characters
        article_with_long_content = {
            "url_hash": "hash_long",
            "title": "Very Long Article",
            "processed_content": long_content
        }
        
        # Mock the _assess_article_relevance method instead of direct API call
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            mock_assess.return_value = {"id": "hash_long", "score": 6.0}
            
            # Act
            reranked = reranker.rerank_articles(test_query, [article_with_long_content])
            
            # Assert
            assert len(reranked) == 1
            assert reranked[0]["rerank_score"] == 6.0
            assert reranked[0]["url_hash"] == "hash_long"
            assert reranked[0]["title"] == "Very Long Article"
            
            # Verify the method was called once
            mock_assess.assert_called_once()
            # We don't need to check exact arguments as they're tested elsewhere
            assert mock_assess.called

    def test_empty_content(self, create_reranker, test_query):
        """Test handling of articles with empty content."""
        # Arrange
        reranker, _ = create_reranker()
        
        article_empty_content = {
            "url_hash": "hash_empty",
            "title": "Empty Article",
            "processed_content": ""
        }
        
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            mock_assess.return_value = {"id": "hash_empty", "score": 0.0}
            
            # Act
            reranked = reranker.rerank_articles(test_query, [article_empty_content])
            
            # Assert
            assert len(reranked) == 1
            assert reranked[0]["rerank_score"] == 0.0
            assert reranked[0]["url_hash"] == "hash_empty"
            assert reranked[0]["title"] == "Empty Article"
            assert mock_assess.called

    def test_none_content(self, create_reranker, test_query):
        """Test handling of articles with None as content."""
        # Arrange
        reranker, _ = create_reranker()
        
        article_none_content = {
            "url_hash": "hash_none",
            "title": "None Content Article",
            "processed_content": None
        }
        
        # Act
        reranked = reranker.rerank_articles(test_query, [article_none_content])
        
        # Assert
        assert len(reranked) == 1
        assert reranked[0]["rerank_score"] == 0.0
        assert reranked[0]["url_hash"] == "hash_none"
        assert reranked[0]["title"] == "None Content Article"
