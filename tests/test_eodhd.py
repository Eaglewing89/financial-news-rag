"""
Tests for the EODHD API client module.
"""

import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from datetime import datetime

from financial_news_rag.eodhd import EODHDClient, EODHDApiError

# Sample response data for mocking
SAMPLE_ARTICLE = {
    "date": "2025-05-08T13:33:00+00:00",
    "title": "Visiting Media Appoints Chad Kimner as SVP of Growth & Operations",
    "content": "NEW YORK, May 08, 2025 (GLOBE NEWSWIRE) -- Visiting Media, the leader in immersive sales and marketing technology for the hospitality industry, today announced the appointment of Chad Kimner as Senior Vice President of Growth & Operations.",
    "link": "https://www.globenewswire.com/news-release/2025/05/08/3077370/0/en/Visiting-Media-Appoints-Chad-Kimner-as-SVP-of-Growth-Operations.html",
    "symbols": [],
    "tags": ["DIRECTORS AND OFFICERS", "EXECUTIVE", "HOSPITALITY"],
    "sentiment": {"polarity": 0.996, "neg": 0, "neu": 0.847, "pos": 0.153}
}

SAMPLE_RESPONSE = [SAMPLE_ARTICLE]


class TestEODHDClient:
    """Tests for the EODHDClient class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Mock API key for testing
        self.api_key = "test_api_key"
        
        # Create client with the mock API key
        self.client = EODHDClient(api_key=self.api_key)
    
    def test_init_with_explicit_api_key(self):
        """Test initialization with an explicit API key."""
        client = EODHDClient(api_key="explicit_key")
        assert client.api_key == "explicit_key"
    
    @patch.dict(os.environ, {"EODHD_API_KEY": "env_key"})
    def test_init_with_env_api_key(self):
        """Test initialization with an API key from environment variables."""
        client = EODHDClient()
        assert client.api_key == "env_key"
    
    def test_init_without_api_key(self):
        """Test initialization without an API key raises ValueError."""
        with patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError):
                EODHDClient()
    
    @patch("requests.get")
    def test_fetch_news_with_tag(self, mock_get):
        """Test fetching news with a tag parameter."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_response
        
        # Call the method
        articles = self.client.fetch_news(tag="HOSPITALITY")
        
        # Check the call parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["t"] == "HOSPITALITY"
        assert kwargs["params"]["api_token"] == self.api_key
        
        # Check the response processing
        assert len(articles) == 1
        assert articles[0]["title"] == SAMPLE_ARTICLE["title"]
        assert articles[0]["source_api"] == "EODHD"
        assert articles[0]["tags"] == SAMPLE_ARTICLE["tags"]
    
    @patch("requests.get")
    def test_fetch_news_with_symbol(self, mock_get):
        """Test fetching news with symbol parameter."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_response
        
        # Call the method with a single valid symbol
        articles = self.client.fetch_news(symbol="AAPL.US")
        
        # Check the call parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["s"] == "AAPL.US"
    
    @patch("requests.get")
    def test_fetch_news_with_none_symbol(self, mock_get):
        """Test fetching news with symbol=None."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_response
        
        # Call the method with tag since symbol is None
        articles = self.client.fetch_news(symbol=None, tag="HOSPITALITY")
        
        # Check the call parameters - should use tag and not symbol
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "s" not in kwargs["params"]
        assert kwargs["params"]["t"] == "HOSPITALITY"
    
    def test_fetch_news_without_tag_or_symbol(self):
        """Test fetching news without tag or symbol raises ValueError."""
        with pytest.raises(ValueError):
            self.client.fetch_news()
    
    @patch("requests.get")
    def test_fetch_news_with_date_range(self, mock_get):
        """Test fetching news with date range parameters."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_response
        
        # Call the method
        articles = self.client.fetch_news(
            tag="HOSPITALITY",
            from_date="2025-05-01",
            to_date="2025-05-10"
        )
        
        # Check the call parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["from"] == "2025-05-01"
        assert kwargs["params"]["to"] == "2025-05-10"
    
    @patch("requests.get")
    def test_fetch_news_with_pagination(self, mock_get):
        """Test fetching news with pagination parameters."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_response
        
        # Call the method
        articles = self.client.fetch_news(
            tag="HOSPITALITY",
            limit=20,
            offset=40
        )
        
        # Check the call parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["limit"] == 20
        assert kwargs["params"]["offset"] == 40
    
    def test_fetch_news_invalid_limit(self):
        """Test that using an invalid limit raises ValueError."""
        # Test limit below the minimum (1)
        with pytest.raises(ValueError) as exc_info:
            self.client.fetch_news(tag="HOSPITALITY", limit=0)
        assert "'limit' must be between 1 and 1000" in str(exc_info.value)
        
        # Test limit above the maximum (1000)
        with pytest.raises(ValueError) as exc_info:
            self.client.fetch_news(tag="HOSPITALITY", limit=1001)
        assert "'limit' must be between 1 and 1000" in str(exc_info.value)
    
    def test_fetch_news_invalid_date_format(self):
        """Test that using invalid date formats raises ValueError."""
        # Test invalid from_date format
        with pytest.raises(ValueError) as exc_info:
            self.client.fetch_news(tag="HOSPITALITY", from_date="05-01-2025")
        assert "'from_date' must be in YYYY-MM-DD format" in str(exc_info.value)
        
        # Test invalid to_date format
        with pytest.raises(ValueError) as exc_info:
            self.client.fetch_news(tag="HOSPITALITY", to_date="2025/05/01")
        assert "'to_date' must be in YYYY-MM-DD format" in str(exc_info.value)
        
        # Test both invalid date formats
        with pytest.raises(ValueError) as exc_info:
            self.client.fetch_news(
                tag="HOSPITALITY", 
                from_date="May 1, 2025",
                to_date="May 10, 2025"
            )
        assert "'from_date' must be in YYYY-MM-DD format" in str(exc_info.value)
    
    def test_fetch_news_valid_date_format(self):
        """Test that using valid date formats works correctly."""
        # Set up the mock
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_RESPONSE
            mock_get.return_value = mock_response
            
            # Test valid from_date and to_date formats
            articles = self.client.fetch_news(
                tag="HOSPITALITY",
                from_date="2025-05-01",
                to_date="2025-05-31"
            )
            
            # Check the call parameters
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert kwargs["params"]["from"] == "2025-05-01"
            assert kwargs["params"]["to"] == "2025-05-31"
    
    @patch("requests.get")
    def test_normalize_article(self, mock_get):
        """Test the normalization of API response articles."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_response
        
        # Call the method
        articles = self.client.fetch_news(tag="HOSPITALITY")
        normalized = articles[0]
        
        # Check the normalization
        assert "url_hash" not in normalized
        assert "fetched_at" not in normalized
        assert normalized["title"] == SAMPLE_ARTICLE["title"]
        assert normalized["raw_content"] == SAMPLE_ARTICLE["content"]
        assert normalized["url"] == SAMPLE_ARTICLE["link"]
        assert normalized["published_at"] == "2025-05-08T13:33:00+00:00"
        assert normalized["source_api"] == "EODHD"
        assert normalized["symbols"] == SAMPLE_ARTICLE["symbols"]
        assert normalized["tags"] == SAMPLE_ARTICLE["tags"]
        assert normalized["sentiment"] == SAMPLE_ARTICLE["sentiment"]
        assert "source_query_tag" in normalized
        assert normalized["source_query_tag"] == "HOSPITALITY"
        
    @patch("requests.get")
    def test_normalize_article_with_symbol(self, mock_get):
        """Test the normalization of API response articles with symbol query."""
        # Set up the mock
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_RESPONSE
        mock_get.return_value = mock_response
        
        # Call the method
        articles = self.client.fetch_news(symbol="AAPL")
        normalized = articles[0]
        
        # Check source query symbol
        assert "source_query_symbol" in normalized
        assert normalized["source_query_symbol"] == "AAPL"
        assert "source_query_tag" not in normalized
    
    @patch("requests.get")
    def test_error_handling(self, mock_get):
        """Test error handling and retry logic."""
        # Create a successful mock response for the third attempt
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_RESPONSE
        
        # Set up the mock to raise exceptions for the first two attempts
        # and return the successful response on the third attempt
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection error"),
            requests.exceptions.Timeout("Timeout error"),
            mock_response  # Successful response on third attempt
        ]
        
        # Call the method with minimal retry settings for faster tests
        with patch("time.sleep"):  # Mock sleep to speed up test
            articles = self.client.fetch_news(
                tag="HOSPITALITY",
                max_retries=3,
                backoff_factor=0.1
            )
        
        # Check that we got a response after retries
        assert len(articles) == 1
        assert articles[0]["title"] == SAMPLE_ARTICLE["title"]
        
        # Reset the mock and test complete failure
        mock_get.reset_mock()
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection error")
        
        # Should raise EODHDApiError after all retries fail
        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(EODHDApiError):
                self.client.fetch_news(
                    tag="HOSPITALITY",
                    max_retries=3,
                    backoff_factor=0.1
                )
    
    @patch("requests.get")
    def test_api_error_propagation(self, mock_get):
        """Test that API errors are properly propagated to the caller."""
        # Create a mock response with an API error
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": "401",
            "message": "Invalid API key or access denied"
        }
        mock_get.return_value = mock_response
        
        # Should raise EODHDApiError with the error message from the API
        with pytest.raises(EODHDApiError) as exc_info:
            self.client.fetch_news(symbol="AAPL")
        
        assert "Invalid API key or access denied" in str(exc_info.value)
        
        # Test HTTP error propagation
        mock_get.reset_mock()
        mock_get.side_effect = requests.exceptions.HTTPError("404 Client Error")
        
        with pytest.raises(EODHDApiError) as exc_info:
            self.client.fetch_news(symbol="AAPL")
        
        assert "404 Client Error" in str(exc_info.value)
