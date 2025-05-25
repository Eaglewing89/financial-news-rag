"""
Unit tests for the ReRanker class.

Tests the ReRanker functionality including initialization, article re-ranking,
error handling, and edge cases. Focuses on isolated unit testing with proper
mocking of external dependencies like the Gemini API.
"""

import pytest
from unittest.mock import patch, MagicMock

from financial_news_rag.reranker import ReRanker


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


class TestReRankerInitialization:
    """Test suite for ReRanker initialization and configuration."""

    @patch("financial_news_rag.reranker.genai.Client")
    def test_initialization(self, mock_client, api_key):
        """Test that the ReRanker is initialized correctly."""
        reranker = ReRanker(api_key=api_key)
        assert reranker.model_name == "gemini-2.0-flash"
        
        # Test with custom model name
        custom_reranker = ReRanker(api_key=api_key, model_name="gemini-3.0-pro")
        assert custom_reranker.model_name == "gemini-3.0-pro"
        
    @patch("financial_news_rag.reranker.genai.Client")
    def test_initialization_no_api_key(self, mock_client):
        """Test that an error is raised when no API key is provided."""
        # Mock the client's __init__ to raise ValueError when no API key is provided
        mock_client.side_effect = ValueError("API key is required")
            
        with pytest.raises(ValueError):
            ReRanker(api_key=None)

    @patch("financial_news_rag.reranker.genai.Client")
    def test_initialization_empty_api_key(self, mock_client):
        """Test that an error is raised when empty API key is provided."""
        with pytest.raises(ValueError, match="Gemini API key is required"):
            ReRanker(api_key="")


class TestReRankerCoreOperations:
    """Test suite for ReRanker core reranking operations."""

    @patch("financial_news_rag.reranker.genai.Client")
    @patch("financial_news_rag.reranker.ReRanker._assess_article_relevance")
    def test_rerank_articles_successful(self, mock_assess, mock_client, api_key, reranker_test_articles, test_query):
        """Test successful re-ranking of articles."""
        # Set up the mock to return different scores for the articles
        mock_assess.side_effect = [
            {"id": "hash1", "score": 8.5},
            {"id": "hash2", "score": 5.2},
            {"id": "hash3", "score": 2.1}
        ]
        
        reranker = ReRanker(api_key=api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(test_query, reranker_test_articles)
        
        # Check that the articles were re-ranked correctly
        assert len(reranked) == 3
        assert reranked[0]["url_hash"] == "hash1"  # Highest score
        assert reranked[1]["url_hash"] == "hash2"  # Middle score
        assert reranked[2]["url_hash"] == "hash3"  # Lowest score
        
        # Check that the scores were added to the articles
        assert reranked[0]["rerank_score"] == 8.5
        assert reranked[1]["rerank_score"] == 5.2
        assert reranked[2]["rerank_score"] == 2.1
        
        # Check that _assess_article_relevance was called the right number of times
        assert mock_assess.call_count == 3

    @patch("financial_news_rag.reranker.genai.Client")
    @patch("financial_news_rag.reranker.ReRanker._assess_article_relevance")
    def test_parse_score_fallback(self, mock_assess, mock_client, api_key, reranker_test_articles, test_query):
        """Test the fallback regex parsing of scores when JSON parsing fails."""
        # Set up the mock to return a response with a score
        mock_assess.return_value = {"id": "hash1", "score": 7.5}
        
        reranker = ReRanker(api_key=api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(test_query, [reranker_test_articles[0]])
        
        # Check that the score was extracted correctly
        assert reranked[0]["rerank_score"] == 7.5

    @patch("financial_news_rag.reranker.genai.Client")
    def test_single_article_reranking(self, mock_client, api_key, test_query):
        """Test re-ranking with a single article."""
        article = {
            "url_hash": "single_hash", 
            "title": "Single Article",
            "processed_content": "Technology and finance convergence."
        }
        
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            mock_assess.return_value = {"id": "single_hash", "score": 9.0}
            
            reranker = ReRanker(api_key=api_key)
            reranked = reranker.rerank_articles(test_query, [article])
            
            assert len(reranked) == 1
            assert reranked[0]["rerank_score"] == 9.0
            assert reranked[0]["url_hash"] == "single_hash"


class TestReRankerErrorHandling:
    """Test suite for ReRanker error handling and resilience."""

    @patch("financial_news_rag.reranker.genai.Client")
    @patch("financial_news_rag.reranker.ReRanker._assess_article_relevance")
    def test_rerank_articles_api_error(self, mock_assess, mock_client, api_key, reranker_test_articles, test_query):
        """Test that the original order is preserved when the API call fails."""
        # Mock the _assess_article_relevance method to raise an exception
        mock_assess.side_effect = Exception("API error")
        
        reranker = ReRanker(api_key=api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(test_query, reranker_test_articles)
        
        # Check that the original articles are returned
        assert reranked == reranker_test_articles

    @patch("financial_news_rag.reranker.genai.Client")
    @patch("financial_news_rag.reranker.ReRanker._assess_article_relevance")
    def test_rerank_articles_malformed_json(self, mock_assess, mock_client, api_key, reranker_test_articles, test_query):
        """Test handling of malformed JSON responses."""
        # Mock the _assess_article_relevance method to return a score of 0
        mock_assess.return_value = {"id": "hash1", "score": 0.0}
        
        reranker = ReRanker(api_key=api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(test_query, [reranker_test_articles[0]])
        
        # Check that the article was processed and given a score of 0.0
        assert len(reranked) == 1
        assert reranked[0]["rerank_score"] == 0.0

    @patch("financial_news_rag.reranker.genai.Client")
    def test_missing_required_fields(self, mock_client, api_key, test_query):
        """Test handling of articles missing required fields."""
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
        
        reranker = ReRanker(api_key=api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(test_query, articles)
        
        # Check that the articles have a score of 0.0
        assert reranked[0]["rerank_score"] == 0.0
        assert reranked[1]["rerank_score"] == 0.0


class TestReRankerEdgeCases:
    """Test suite for ReRanker edge cases and boundary conditions."""

    @patch("financial_news_rag.reranker.genai.Client")
    def test_empty_article_list(self, mock_client, api_key, test_query):
        """Test handling of an empty article list."""
        reranker = ReRanker(api_key=api_key)
        reranked = reranker.rerank_articles(test_query, [])
        assert reranked == []
        
    @patch("financial_news_rag.reranker.genai.Client")
    def test_empty_query(self, mock_client, api_key, reranker_test_articles):
        """Test handling of an empty query."""
        reranker = ReRanker(api_key=api_key)
        reranked = reranker.rerank_articles("", reranker_test_articles)
        assert reranked == reranker_test_articles

    @patch("financial_news_rag.reranker.genai.Client")
    def test_whitespace_only_query(self, mock_client, api_key, reranker_test_articles):
        """Test handling of a whitespace-only query."""
        reranker = ReRanker(api_key=api_key)
        reranked = reranker.rerank_articles("   \t\n  ", reranker_test_articles)
        assert reranked == reranker_test_articles

    @patch("financial_news_rag.reranker.genai.Client")
    def test_very_long_content(self, mock_client, api_key, test_query):
        """Test handling of articles with very long content."""
        # Create an article with content longer than 10000 characters
        long_content = "A" * 15000  # 15000 characters
        article_with_long_content = {
            "url_hash": "hash_long",
            "title": "Very Long Article",
            "processed_content": long_content
        }
        
        # Mock the Gemini API response
        mock_response = MagicMock()
        mock_response.text = '{"id": "hash_long", "score": 6.0}'
        
        reranker = ReRanker(api_key=api_key)
        reranker.client.models.generate_content.return_value = mock_response
        
        reranked = reranker.rerank_articles(test_query, [article_with_long_content])
        
        # Check that the article was processed successfully
        assert len(reranked) == 1
        assert reranked[0]["rerank_score"] == 6.0
        
        # Verify that generate_content was called (content truncation happens internally)
        assert reranker.client.models.generate_content.called

    @patch("financial_news_rag.reranker.genai.Client") 
    def test_empty_content(self, mock_client, api_key, test_query):
        """Test handling of articles with empty content."""
        article_empty_content = {
            "url_hash": "hash_empty",
            "title": "Empty Article",
            "processed_content": ""
        }
        
        with patch("financial_news_rag.reranker.ReRanker._assess_article_relevance") as mock_assess:
            mock_assess.return_value = {"id": "hash_empty", "score": 0.0}
            
            reranker = ReRanker(api_key=api_key)
            reranked = reranker.rerank_articles(test_query, [article_empty_content])
            
            # Check that the article received a score of 0.0
            assert len(reranked) == 1
            assert reranked[0]["rerank_score"] == 0.0
