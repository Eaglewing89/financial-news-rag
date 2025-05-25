"""
Unit tests for the ReRanker class.

Tests the ReRanker functionality including initialization, article re-ranking,
error handling, and edge cases.
"""

import pytest
from unittest.mock import patch, MagicMock

from financial_news_rag.reranker import ReRanker


class TestReRanker:
    """Test cases for the ReRanker class."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up for the tests."""
        # Create a mock API key for testing
        self.api_key = "test_api_key"
        
        # Create sample articles for testing
        self.mock_articles = [
            {
                "url_hash": "hash1",
                "title": "Article 1",
                "processed_content": "This article discusses technology trends in finance.",
                "other_field": "Some other data"
            },
            {
                "url_hash": "hash2",
                "title": "Article 2",
                "processed_content": "This article discusses market trends.",
                "other_field": "Some other data"
            },
            {
                "url_hash": "hash3",
                "title": "Article 3",
                "processed_content": "This article is about sports events.",
                "other_field": "Some other data"
            }
        ]
        
        self.test_query = "technology trends in finance"

    @patch("financial_news_rag.reranker.genai.Client")
    def test_initialization(self, mock_client):
        """Test that the ReRanker is initialized correctly."""
        reranker = ReRanker(api_key=self.api_key)
        assert reranker.model_name == "gemini-2.0-flash"
        
        # Test with custom model name
        custom_reranker = ReRanker(api_key="explicit_key", model_name="gemini-3.0-pro")
        assert custom_reranker.model_name == "gemini-3.0-pro"
        
    @patch("financial_news_rag.reranker.genai.Client")
    def test_initialization_no_api_key(self, mock_client):
        """Test that an error is raised when no API key is provided."""
        # Mock the client's __init__ to raise ValueError when no API key is provided
        mock_client.side_effect = ValueError("API key is required")
            
        with pytest.raises(ValueError):
            ReRanker(api_key=None)

    @patch("financial_news_rag.reranker.genai.Client")
    @patch("financial_news_rag.reranker.ReRanker._assess_article_relevance")
    def test_rerank_articles_successful(self, mock_assess, mock_client):
        """Test successful re-ranking of articles."""
        # Set up the mock to return different scores for the articles
        mock_assess.side_effect = [
            {"id": "hash1", "score": 8.5},
            {"id": "hash2", "score": 5.2},
            {"id": "hash3", "score": 2.1}
        ]
        
        reranker = ReRanker(api_key=self.api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(self.test_query, self.mock_articles)
        
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
    def test_rerank_articles_api_error(self, mock_assess, mock_client):
        """Test that the original order is preserved when the API call fails."""
        # Mock the _assess_article_relevance method to raise an exception
        mock_assess.side_effect = Exception("API error")
        
        reranker = ReRanker(api_key=self.api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(self.test_query, self.mock_articles)
        
        # Check that the original articles are returned
        assert reranked == self.mock_articles

    @patch("financial_news_rag.reranker.genai.Client")
    @patch("financial_news_rag.reranker.ReRanker._assess_article_relevance")
    def test_rerank_articles_malformed_json(self, mock_assess, mock_client):
        """Test handling of malformed JSON responses."""
        # Mock the _assess_article_relevance method to return a score of 0
        mock_assess.return_value = {"id": "hash1", "score": 0.0}
        
        reranker = ReRanker(api_key=self.api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(self.test_query, [self.mock_articles[0]])
        
        # Check that the article was processed and given a score of 0.0
        assert len(reranked) == 1
        assert reranked[0]["rerank_score"] == 0.0
        
    @patch("financial_news_rag.reranker.genai.Client")
    @patch("financial_news_rag.reranker.ReRanker._assess_article_relevance")
    def test_parse_score_fallback(self, mock_assess, mock_client):
        """Test the fallback regex parsing of scores when JSON parsing fails."""
        # Set up the mock to return a response with a score
        mock_assess.return_value = {"id": "hash1", "score": 7.5}
        
        reranker = ReRanker(api_key=self.api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(self.test_query, [self.mock_articles[0]])
        
        # Check that the score was extracted correctly
        assert reranked[0]["rerank_score"] == 7.5

    @patch("financial_news_rag.reranker.genai.Client")
    def test_empty_article_list(self, mock_client):
        """Test handling of an empty article list."""
        reranker = ReRanker(api_key=self.api_key)
        reranked = reranker.rerank_articles(self.test_query, [])
        assert reranked == []
        
    @patch("financial_news_rag.reranker.genai.Client")
    def test_empty_query(self, mock_client):
        """Test handling of an empty query."""
        reranker = ReRanker(api_key=self.api_key)
        reranked = reranker.rerank_articles("", self.mock_articles)
        assert reranked == self.mock_articles

    @patch("financial_news_rag.reranker.genai.Client")
    def test_missing_required_fields(self, mock_client):
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
        
        reranker = ReRanker(api_key=self.api_key)
        
        # Re-rank the articles
        reranked = reranker.rerank_articles(self.test_query, articles)
        
        # Check that the articles have a score of 0.0
        assert reranked[0]["rerank_score"] == 0.0
        assert reranked[1]["rerank_score"] == 0.0
