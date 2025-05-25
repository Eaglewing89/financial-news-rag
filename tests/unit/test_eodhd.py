"""
Unit tests for the EODHDClient class.

These tests validate the EODHD API client functionality including
initialization, API calls, error handling, and response processing.
All external API calls are mocked for isolated testing.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from financial_news_rag.eodhd import EODHDClient
from tests.fixtures.sample_data import EODHDResponseFactory


class TestEODHDClientInitialization:
    """
    Test suite for EODHDClient initialization and configuration.
    """

    def test_init_with_explicit_api_key(self):
        """Test initialization with an explicit API key."""
        client = EODHDClient(api_key="explicit_test_key")
        assert client.api_key == "explicit_test_key"
        assert client.api_url == "https://eodhd.com/api/news"  # Default URL
        assert client.default_timeout == 30  # Default timeout

    def test_init_with_custom_parameters(self):
        """Test initialization with custom configuration parameters."""
        client = EODHDClient(
            api_key="test_key",
            api_url="https://custom.api.url",
            default_timeout=60,
            default_max_retries=5,
            default_backoff_factor=2.0,
            default_limit=500,
        )

        assert client.api_key == "test_key"
        assert client.api_url == "https://custom.api.url"
        assert client.default_timeout == 60
        assert client.default_max_retries == 5
        assert client.default_backoff_factor == 2.0
        assert client.default_limit == 500

    @patch.dict("os.environ", {"EODHD_API_KEY": "env_test_key"})
    def test_init_with_environment_variables(self):
        """Test initialization with configuration values from environment."""
        # Test that EODHDClient can be initialized with explicit values
        # that might come from a configuration system
        client = EODHDClient(
            api_key="env_test_key",
            api_url="https://eodhd.com/api/news",
            default_timeout=30,
            default_max_retries=3,
            default_backoff_factor=1.0,
            default_limit=1000,
        )

        assert client.api_key == "env_test_key"
        assert client.api_url == "https://eodhd.com/api/news"
        assert client.default_timeout == 30
        assert client.default_max_retries == 3
        assert client.default_backoff_factor == 1.0
        assert client.default_limit == 1000

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="EODHD API key is required"):
            EODHDClient(api_key="")

        with pytest.raises(ValueError, match="EODHD API key is required"):
            EODHDClient(api_key=None)

    def test_init_preserves_none_values_for_optional_params(self):
        """Test that None values for optional parameters are preserved."""
        client = EODHDClient(
            api_key="test_key", default_timeout=None, default_max_retries=None
        )

        assert client.api_key == "test_key"
        assert client.default_timeout is None
        assert client.default_max_retries is None


class TestEODHDClientFetchNews:
    """Test suite for EODHDClient news fetching functionality."""

    @pytest.fixture
    def client(self):
        """Create a test EODHDClient instance."""
        return EODHDClient(api_key="test_api_key")

    @patch("requests.get")
    def test_fetch_news_with_tag_success(self, mock_get, client):
        """Test successful news fetching with a tag parameter."""
        # Setup mock response
        sample_articles = EODHDResponseFactory.create_response(count=2)
        mock_response = MagicMock()
        mock_response.json.return_value = sample_articles
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Make the API call
        result = client.fetch_news(tag="TECHNOLOGY")

        # Verify request parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["t"] == "TECHNOLOGY"
        assert kwargs["params"]["api_token"] == "test_api_key"
        assert kwargs["params"]["fmt"] == "json"

        # Verify response processing
        assert isinstance(result, dict)
        assert "articles" in result
        assert "status_code" in result
        assert "success" in result
        assert "error_message" in result

        assert len(result["articles"]) == 2
        assert result["status_code"] == 200
        assert result["success"] is True
        assert result["error_message"] is None

        # Verify articles have source_api field added
        for article in result["articles"]:
            assert article["source_api"] == "EODHD"

    @patch("requests.get")
    def test_fetch_news_with_symbol_success(self, mock_get, client):
        """Test successful news fetching with a symbol parameter."""
        sample_articles = EODHDResponseFactory.create_response(
            count=1, symbols=["AAPL.US"]
        )
        mock_response = MagicMock()
        mock_response.json.return_value = sample_articles
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Make API call with symbol
        result = client.fetch_news(symbol="AAPL.US")

        # Verify request parameters
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["s"] == "AAPL.US"
        assert "t" not in kwargs["params"]  # Tag should not be present

        # Verify response
        assert result["success"] is True
        assert len(result["articles"]) == 1
        assert result["articles"][0]["symbols"] == ["AAPL.US"]

    @patch("requests.get")
    def test_fetch_news_with_date_range(self, mock_get, client):
        """Test news fetching with date range parameters."""
        sample_articles = EODHDResponseFactory.create_response(count=1)
        mock_response = MagicMock()
        mock_response.json.return_value = sample_articles
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Make API call with date range
        result = client.fetch_news(
            tag="TECHNOLOGY", from_date="2023-01-01", to_date="2023-01-31"
        )

        # Verify date parameters
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["from"] == "2023-01-01"
        assert kwargs["params"]["to"] == "2023-01-31"

        assert result["success"] is True

    @patch("requests.get")
    def test_fetch_news_with_limit_and_offset(self, mock_get, client):
        """Test news fetching with pagination parameters."""
        sample_articles = EODHDResponseFactory.create_response(count=5)
        mock_response = MagicMock()
        mock_response.json.return_value = sample_articles
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Make API call with pagination
        result = client.fetch_news(tag="FINANCE", limit=5, offset=10)

        # Verify pagination parameters
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["limit"] == 5
        assert kwargs["params"]["offset"] == 10

        assert result["success"] is True
        assert len(result["articles"]) == 5

    @patch("time.sleep")  # Mock sleep to speed up tests
    @patch("requests.get")
    def test_fetch_news_request_timeout_error(self, mock_get, mock_sleep, client):
        """Test handling of request timeout errors."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = client.fetch_news(tag="TECHNOLOGY")

        # Verify error handling
        assert result["success"] is False
        assert result["articles"] == []
        assert result["status_code"] is None
        assert "request timed out" in result["error_message"].lower()

    @patch("time.sleep")  # Mock sleep to speed up tests
    @patch("requests.get")
    def test_fetch_news_connection_error(self, mock_get, mock_sleep, client):
        """Test handling of connection errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        result = client.fetch_news(tag="TECHNOLOGY")

        # Verify error handling
        assert result["success"] is False
        assert result["articles"] == []
        assert result["status_code"] is None
        assert "connection" in result["error_message"].lower()

    @patch("time.sleep")  # Mock sleep to speed up tests
    @patch("requests.get")
    def test_fetch_news_http_error_response(self, mock_get, mock_sleep, client):
        """Test handling of HTTP error responses."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "401 Unauthorized"
        )
        mock_response.text = "Invalid API key"
        mock_get.return_value = mock_response

        result = client.fetch_news(tag="TECHNOLOGY")

        # Verify error handling
        assert result["success"] is False
        assert result["articles"] == []
        assert result["status_code"] == 401
        assert "401" in result["error_message"]

    @patch("time.sleep")  # Mock sleep to speed up tests
    @patch("requests.get")
    def test_fetch_news_invalid_json_response(self, mock_get, mock_sleep, client):
        """Test handling of invalid JSON responses."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "Invalid JSON response"
        mock_get.return_value = mock_response

        result = client.fetch_news(tag="TECHNOLOGY")

        # Verify error handling
        assert result["success"] is False
        assert result["articles"] == []
        assert result["status_code"] == 200
        assert "json" in result["error_message"].lower()

    @patch("requests.get")
    def test_fetch_news_empty_response(self, mock_get, client):
        """Test handling of empty API responses."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.fetch_news(tag="OBSCURE_TAG")

        # Verify successful handling of empty response
        assert result["success"] is True
        assert result["articles"] == []
        assert result["status_code"] == 200
        assert result["error_message"] is None

    def test_fetch_news_no_parameters_raises_error(self, client):
        """Test that calling fetch_news without tag or symbol raises error."""
        with pytest.raises(
            ValueError, match="Either 'tag' or 'symbol' parameter is required"
        ):
            client.fetch_news()

    def test_fetch_news_both_tag_and_symbol_raises_error(self, client):
        """Test that providing both tag and symbol raises error."""
        with pytest.raises(ValueError, match="Cannot specify both 'tag' and 'symbol'"):
            client.fetch_news(tag="TECHNOLOGY", symbol="AAPL.US")


class TestEODHDClientResponseProcessing:
    """Test suite for EODHDClient response processing functionality."""

    @pytest.fixture
    def client(self):
        """Create a test EODHDClient instance."""
        return EODHDClient(api_key="test_api_key")

    @patch("requests.get")
    def test_response_article_processing(self, mock_get, client):
        """Test that articles are properly processed and enhanced."""
        # Create sample response with various article types
        sample_articles = [
            {
                "date": "2023-01-01T12:00:00+00:00",
                "title": "Test Article 1",
                "content": "Test content 1",
                "link": "https://example.com/article1",
                "symbols": ["AAPL.US"],
                "tags": ["TECHNOLOGY"],
                "sentiment": {"polarity": 0.5, "neg": 0.1, "neu": 0.5, "pos": 0.4},
            },
            {
                "date": "2023-01-02T15:30:00+00:00",
                "title": "Test Article 2",
                "content": "Test content 2",
                "link": "https://example.com/article2",
                "symbols": ["MSFT.US", "GOOGL.US"],
                "tags": ["TECHNOLOGY", "EARNINGS"],
                "sentiment": {"polarity": -0.2, "neg": 0.6, "neu": 0.3, "pos": 0.1},
            },
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = sample_articles
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.fetch_news(tag="TECHNOLOGY")

        # Verify article processing
        assert len(result["articles"]) == 2

        # Check first article
        article1 = result["articles"][0]
        assert article1["source_api"] == "EODHD"
        assert article1["published_at"] == "2023-01-01T12:00:00+00:00"
        assert article1["raw_content"] == "Test content 1"
        assert article1["url"] == "https://example.com/article1"
        assert article1["symbols"] == ["AAPL.US"]
        assert article1["tags"] == ["TECHNOLOGY"]

        # Check second article
        article2 = result["articles"][1]
        assert article2["source_api"] == "EODHD"
        assert article2["symbols"] == ["MSFT.US", "GOOGL.US"]
        assert article2["tags"] == ["TECHNOLOGY", "EARNINGS"]

    @patch("requests.get")
    def test_source_query_metadata_addition(self, mock_get, client):
        """Test that source query metadata is added to articles."""
        sample_articles = EODHDResponseFactory.create_response(count=1)
        mock_response = MagicMock()
        mock_response.json.return_value = sample_articles
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test with tag query
        result = client.fetch_news(tag="TECHNOLOGY")
        article = result["articles"][0]
        assert article.get("source_query_tag") == "TECHNOLOGY"
        assert "source_query_symbol" not in article

        # Test with symbol query
        result = client.fetch_news(symbol="AAPL.US")
        article = result["articles"][0]
        assert article.get("source_query_symbol") == "AAPL.US"
        assert "source_query_tag" not in article

    @patch("requests.get")
    def test_field_mapping_compatibility(self, mock_get, client):
        """Test that API response fields are properly mapped to internal format."""
        # Test article with EODHD API field names
        eodhd_article = {
            "date": "2023-01-01T12:00:00+00:00",  # Maps to published_at
            "title": "Test Article",
            "content": "Test content",  # Maps to raw_content
            "link": "https://example.com/article",  # Maps to url
            "symbols": ["AAPL.US"],
            "tags": ["TECHNOLOGY"],
            "sentiment": {"polarity": 0.5},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = [eodhd_article]
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.fetch_news(tag="TECHNOLOGY")
        article = result["articles"][0]

        # Verify field mapping
        assert article["published_at"] == "2023-01-01T12:00:00+00:00"
        assert article["raw_content"] == "Test content"
        assert article["url"] == "https://example.com/article"
        assert article["title"] == "Test Article"
        assert article["symbols"] == ["AAPL.US"]
        assert article["tags"] == ["TECHNOLOGY"]
        assert article["sentiment"] == {"polarity": 0.5}


class TestEODHDClientRetryLogic:
    """Test suite for EODHDClient retry and backoff functionality."""

    @pytest.fixture
    def client_with_retries(self):
        """Create a client configured with retry parameters."""
        return EODHDClient(
            api_key="test_api_key",
            default_max_retries=3,
            default_backoff_factor=0.1,  # Fast retries for testing
        )

    @patch("requests.get")
    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_retry_on_temporary_failure(
        self, mock_sleep, mock_get, client_with_retries
    ):
        """Test retry logic on temporary failures."""
        # First two calls fail, third succeeds
        sample_articles = EODHDResponseFactory.create_response(count=1)
        success_response = MagicMock()
        success_response.json.return_value = sample_articles
        success_response.status_code = 200
        success_response.raise_for_status.return_value = None

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Temporary failure"),
            requests.exceptions.Timeout("Temporary timeout"),
            success_response,
        ]

        result = client_with_retries.fetch_news(tag="TECHNOLOGY")

        # Verify success after retries
        assert result["success"] is True
        assert len(result["articles"]) == 1
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries

    @patch("requests.get")
    def test_max_retries_exceeded(self, mock_get, client_with_retries):
        """Test behavior when maximum retries are exceeded."""
        # All calls fail
        mock_get.side_effect = requests.exceptions.ConnectionError("Persistent failure")

        result = client_with_retries.fetch_news(tag="TECHNOLOGY")

        # Verify failure after max retries
        assert result["success"] is False
        assert "persistent failure" in result["error_message"].lower()
        # Should try max_retries times (not 1 + max_retries)
        assert mock_get.call_count == client_with_retries.default_max_retries
