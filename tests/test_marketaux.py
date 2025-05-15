"""
Tests for the Marketaux news fetcher module.
"""

import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from datetime import datetime

from financial_news_rag.marketaux import (
    MarketauxNewsFetcher, 
    RateLimiter,
    fetch_with_retry
)


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    def test_initialization(self):
        """Test that rate limiter initializes with correct values."""
        limiter = RateLimiter(calls_per_minute=30)
        assert limiter.calls_per_minute == 30
        assert limiter.call_times == []
    
    @patch('time.time')
    @patch('time.sleep')
    def test_wait_if_needed(self, mock_sleep, mock_time):
        """Test wait_if_needed respects rate limits."""
        # Mock the current time to be fixed
        mock_time.return_value = 1000
        
        # Initialize with 2 calls per minute
        limiter = RateLimiter(calls_per_minute=2)
        
        # Add two calls just within the window (less than 60s ago)
        limiter.call_times = [950, 970]
        
        # Call wait_if_needed (should trigger sleep)
        limiter.wait_if_needed()
        
        # Sleep should be called with appropriate duration (60 - (1000 - 950))
        mock_sleep.assert_called_once_with(10)
        
        # After wait, a new call should be recorded
        assert len(limiter.call_times) == 3
        assert limiter.call_times[-1] == 1000


class TestFetchWithRetry:
    """Tests for fetch_with_retry function."""
    
    @patch('requests.get')
    def test_successful_request(self, mock_get):
        """Test successful API request with no retries needed."""
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test_data"}
        mock_get.return_value = mock_response
        
        result = fetch_with_retry("https://api.test.com", params={"key": "value"})
        
        # Should return the JSON response
        assert result == {"data": "test_data"}
        # Should only call requests.get once
        mock_get.assert_called_once()
    
    @patch('requests.get')
    @patch('time.sleep')
    def test_retry_logic(self, mock_sleep, mock_get):
        """Test retry logic with failed requests."""
        # Mock first two calls to fail, third to succeed
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test_data"}
        
        mock_error = requests.exceptions.RequestException("Test error")
        mock_get.side_effect = [mock_error, mock_error, mock_response]
        
        result = fetch_with_retry(
            "https://api.test.com", 
            params={"key": "value"},
            max_retries=3,
            backoff_factor=1.0
        )
        
        # Should return the JSON response after retries
        assert result == {"data": "test_data"}
        # Should have called requests.get three times
        assert mock_get.call_count == 3
        # Should have called sleep twice (after first two failures)
        assert mock_sleep.call_count == 2
    
    @patch('requests.get')
    @patch('time.sleep')
    def test_max_retries_exceeded(self, mock_sleep, mock_get):
        """Test exception raised when max retries exceeded."""
        # Mock all calls to fail
        mock_error = requests.exceptions.RequestException("Test error")
        mock_get.side_effect = [mock_error, mock_error, mock_error]
        
        with pytest.raises(Exception) as excinfo:
            fetch_with_retry(
                "https://api.test.com", 
                params={"key": "value"},
                max_retries=3,
                backoff_factor=1.0
            )
        
        # Exception message should mention max retries
        assert "Failed after 3 attempts" in str(excinfo.value)
        # Should have called requests.get three times
        assert mock_get.call_count == 3
        # Should have called sleep twice
        assert mock_sleep.call_count == 2


class TestMarketauxNewsFetcher:
    """Tests for the MarketauxNewsFetcher class."""
    
    def test_initialization_with_token(self):
        """Test initialization with explicit token."""
        fetcher = MarketauxNewsFetcher(api_token="test_token")
        assert fetcher.api_token == "test_token"
        assert isinstance(fetcher.rate_limiter, RateLimiter)
    
    @patch.dict(os.environ, {"MARKETAUX_API_KEY": "env_test_token"})
    def test_initialization_from_env(self):
        """Test initialization with token from environment."""
        fetcher = MarketauxNewsFetcher()
        assert fetcher.api_token == "env_test_token"
    
    @patch('os.getenv')
    def test_initialization_missing_token(self, mock_getenv):
        """Test error when token is missing."""
        # Mock os.getenv to return None for MARKETAUX_API_KEY
        mock_getenv.return_value = None
        
        with pytest.raises(ValueError) as excinfo:
            MarketauxNewsFetcher()
        assert "Marketaux API token is required" in str(excinfo.value)
    
    @patch('financial_news_rag.marketaux.fetch_with_retry')
    @patch.object(RateLimiter, 'wait_if_needed')
    def test_fetch_news_basic(self, mock_wait, mock_fetch):
        """Test basic news fetching with minimal parameters."""
        # Mock fetch_with_retry response
        mock_fetch.return_value = {
            "data": [{"title": "Test Article"}],
            "meta": {"found": 1, "returned": 1}
        }
        
        fetcher = MarketauxNewsFetcher(api_token="test_token")
        result = fetcher.fetch_news(symbols=["TSLA", "AAPL"])
        
        # Rate limiter should be called
        mock_wait.assert_called_once()
        
        # fetch_with_retry should be called with correct params
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args[0]
        assert call_args[0] == MarketauxNewsFetcher.NEWS_API_URL
        
        # Check parameters
        params = call_args[1]
        assert params["api_token"] == "test_token"
        assert params["symbols"] == "TSLA,AAPL"
        assert params["page"] == 1
        
        # Should return the mock response
        assert result["data"][0]["title"] == "Test Article"
    
    @patch('financial_news_rag.marketaux.fetch_with_retry')
    def test_fetch_news_all_parameters(self, mock_fetch):
        """Test news fetching with all parameters."""
        mock_fetch.return_value = {"data": []}
        
        fetcher = MarketauxNewsFetcher(api_token="test_token")
        result = fetcher.fetch_news(
            symbols=["TSLA"],
            entity_types=["equity"],
            industries=["Technology"],
            countries=["us"],
            sentiment_gte=0.2,
            sentiment_lte=0.9,
            min_match_score=0.5,
            filter_entities=True,
            must_have_entities=True,
            group_similar=False,
            search="earnings",
            domains=["finance.example.com"],
            exclude_domains=["spam.example.com"],
            source_ids=["source1"],
            exclude_source_ids=["source2"],
            language=["en"],
            published_before=datetime(2025, 5, 1),
            published_after=datetime(2025, 4, 1),
            published_on="2025-04-15",
            sort="published_on",
            sort_order="asc",
            limit=10,
            page=2
        )
        
        # Check parameters passed to fetch_with_retry
        call_args = mock_fetch.call_args[0]
        params = call_args[1]
        
        assert params["symbols"] == "TSLA"
        assert params["entity_types"] == "equity"
        assert params["industries"] == "Technology"
        assert params["countries"] == "us"
        assert params["sentiment_gte"] == 0.2
        assert params["sentiment_lte"] == 0.9
        assert params["min_match_score"] == 0.5
        assert params["filter_entities"] == "true"
        assert params["must_have_entities"] == "true"
        assert params["group_similar"] == "false"
        assert params["search"] == "earnings"
        assert params["domains"] == "finance.example.com"
        assert params["exclude_domains"] == "spam.example.com"
        assert params["source_ids"] == "source1"
        assert params["exclude_source_ids"] == "source2"
        assert params["language"] == "en"
        assert params["published_before"] == "2025-05-01T00:00:00"
        assert params["published_after"] == "2025-04-01T00:00:00"
        assert params["published_on"] == "2025-04-15"
        assert params["sort"] == "published_on"
        assert params["sort_order"] == "asc"
        assert params["limit"] == 10
        assert params["page"] == 2
    
    @patch('financial_news_rag.marketaux.fetch_with_retry')
    def test_invalid_sort_option(self, mock_fetch):
        """Test error when invalid sort option is provided."""
        fetcher = MarketauxNewsFetcher(api_token="test_token")
        
        with pytest.raises(ValueError) as excinfo:
            fetcher.fetch_news(sort="invalid_option")
        
        assert "Invalid sort option" in str(excinfo.value)
        # fetch_with_retry should not be called
        mock_fetch.assert_not_called()
    
    @patch('financial_news_rag.marketaux.fetch_with_retry')
    def test_search_entities(self, mock_fetch):
        """Test entity search functionality."""
        mock_fetch.return_value = {"data": [{"symbol": "TSLA"}]}
        
        fetcher = MarketauxNewsFetcher(api_token="test_token")
        result = fetcher.search_entities(
            search="Tesla",
            symbols=["TSLA"],
            types=["equity"]
        )
        
        # Check parameters passed to fetch_with_retry
        call_args = mock_fetch.call_args[0]
        assert call_args[0] == MarketauxNewsFetcher.ENTITY_SEARCH_URL
        
        params = call_args[1]
        assert params["search"] == "Tesla"
        assert params["symbols"] == "TSLA"
        assert params["types"] == "equity"
    
    def test_normalize_article_data(self):
        """Test normalization of article data."""
        fetcher = MarketauxNewsFetcher(api_token="test_token")
        
        # Test with minimal article data
        articles = [
            {
                "uuid": "test-uuid",
                "title": "Test Article",
                "url": "https://example.com/article"
            }
        ]
        
        normalized = fetcher.normalize_article_data(articles)
        
        # Check that required fields are present
        assert len(normalized) == 1
        assert normalized[0]["uuid"] == "test-uuid"
        assert normalized[0]["title"] == "Test Article"
        assert normalized[0]["url"] == "https://example.com/article"
        assert "entities" in normalized[0]
        assert "published_at" in normalized[0]
        
        # Test with entities
        articles = [
            {
                "uuid": "test-uuid",
                "title": "Test Article",
                "entities": [
                    {"symbol": "TSLA", "sentiment_score": 0.8}
                ]
            }
        ]
        
        normalized = fetcher.normalize_article_data(articles)
        
        # Check that entity normalization works
        assert len(normalized[0]["entities"]) == 1
        assert normalized[0]["entities"][0]["symbol"] == "TSLA"
        assert normalized[0]["entities"][0]["sentiment_score"] == 0.8
        assert "name" in normalized[0]["entities"][0]  # Should add missing fields
