"""
Tests for the data module.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from financial_news_rag.data import (
    fetch_marketaux_news_snippets,
    normalize_news_data,
    search_entities,
    get_entity_types,
    get_industry_list,
    get_news_sources
)


@patch('financial_news_rag.data.MarketauxNewsFetcher')
def test_fetch_marketaux_news_snippets_basic(mock_marketaux_class):
    """Test basic news fetching functionality."""
    # Set up mock
    mock_marketaux = MagicMock()
    mock_marketaux.fetch_news.return_value = {
        "meta": {"found": 10, "returned": 3},
        "data": [{"title": "Test Article"}]
    }
    mock_marketaux_class.return_value = mock_marketaux
    
    # Call function
    result = fetch_marketaux_news_snippets(
        symbols=["TSLA", "AAPL"],
        language=["en"],
        limit=3
    )
    
    # Check results
    assert result["meta"]["found"] == 10
    assert result["meta"]["returned"] == 3
    assert result["data"][0]["title"] == "Test Article"
    
    # Verify mock was called with correct parameters
    mock_marketaux.fetch_news.assert_called_once()
    call_kwargs = mock_marketaux.fetch_news.call_args[1]
    assert call_kwargs["symbols"] == ["TSLA", "AAPL"]
    assert call_kwargs["language"] == ["en"]
    assert call_kwargs["limit"] == 3


@patch('financial_news_rag.data.MarketauxNewsFetcher')
def test_fetch_marketaux_news_snippets_with_dates(mock_marketaux_class):
    """Test fetching news with date parameters."""
    # Set up mock
    mock_marketaux = MagicMock()
    mock_marketaux.fetch_news.return_value = {"data": []}
    mock_marketaux_class.return_value = mock_marketaux
    
    # Call with days_back
    fetch_marketaux_news_snippets(days_back=5)
    
    # Verify date handling
    call_kwargs = mock_marketaux.fetch_news.call_args[1]
    assert isinstance(call_kwargs["published_after"], datetime)
    
    # Today minus 5 days (with some allowance for test execution time)
    expected_date = datetime.now() - timedelta(days=5)
    date_diff = call_kwargs["published_after"] - expected_date
    assert abs(date_diff.total_seconds()) < 10  # Should be within 10 seconds
    
    # Call with date_range
    start_date = datetime(2025, 5, 1)
    end_date = datetime(2025, 5, 10)
    fetch_marketaux_news_snippets(date_range=(start_date, end_date))
    
    # Verify date range handling
    call_kwargs = mock_marketaux.fetch_news.call_args[1]
    assert call_kwargs["published_after"] == start_date
    assert call_kwargs["published_before"] == end_date


@patch('financial_news_rag.data.MarketauxNewsFetcher')
def test_fetch_marketaux_news_snippets_error_handling(mock_marketaux_class):
    """Test error handling in fetch_marketaux_news_snippets."""
    # Set up mock to raise exception
    mock_marketaux = MagicMock()
    mock_marketaux.fetch_news.side_effect = Exception("API error")
    mock_marketaux_class.return_value = mock_marketaux
    
    # Call should raise exception
    with pytest.raises(Exception) as excinfo:
        fetch_marketaux_news_snippets(symbols=["TSLA"])
    
    assert "API error" in str(excinfo.value)


def test_normalize_news_data():
    """Test normalization of news data."""
    # Create sample articles
    articles = [
        {
            "uuid": "test-uuid",
            "title": "Test Article",
            "url": "https://example.com",
            "entities": [{"symbol": "TSLA"}]
        }
    ]
    
    # Mock MarketauxNewsFetcher.normalize_article_data
    with patch('financial_news_rag.marketaux.MarketauxNewsFetcher.normalize_article_data') as mock_normalize:
        mock_normalize.return_value = [
            {
                "uuid": "test-uuid",
                "title": "Test Article",
                "url": "https://example.com",
                "entities": [{"symbol": "TSLA", "name": "Tesla Inc."}]
            }
        ]
        
        result = normalize_news_data(articles)
        
        # Check result
        assert result[0]["uuid"] == "test-uuid"
        assert result[0]["entities"][0]["name"] == "Tesla Inc."
        
        # Verify mock was called
        mock_normalize.assert_called_once_with(articles)


@patch('financial_news_rag.data.MarketauxNewsFetcher')
def test_search_entities(mock_marketaux_class):
    """Test entity search functionality."""
    # Set up mock
    mock_marketaux = MagicMock()
    mock_marketaux.search_entities.return_value = {
        "data": [{"symbol": "TSLA", "name": "Tesla Inc."}]
    }
    mock_marketaux_class.return_value = mock_marketaux
    
    # Call function
    result = search_entities(search="Tesla", entity_types=["equity"])
    
    # Check results
    assert result["data"][0]["symbol"] == "TSLA"
    
    # Verify mock was called correctly
    mock_marketaux.search_entities.assert_called_once()
    call_kwargs = mock_marketaux.search_entities.call_args[1]
    assert call_kwargs["search"] == "Tesla"
    assert call_kwargs["types"] == ["equity"]


@patch('financial_news_rag.data.MarketauxNewsFetcher')
def test_get_entity_types(mock_marketaux_class):
    """Test fetching entity types."""
    # Set up mock
    mock_marketaux = MagicMock()
    mock_marketaux.get_entity_types.return_value = {
        "data": ["equity", "index", "etf"]
    }
    mock_marketaux_class.return_value = mock_marketaux
    
    # Call function
    result = get_entity_types()
    
    # Check results
    assert len(result) == 3
    assert "equity" in result
    assert "index" in result
    
    # Verify mock was called
    mock_marketaux.get_entity_types.assert_called_once()


@patch('financial_news_rag.data.MarketauxNewsFetcher')
def test_get_industry_list(mock_marketaux_class):
    """Test fetching industry list."""
    # Set up mock
    mock_marketaux = MagicMock()
    mock_marketaux.get_industry_list.return_value = {
        "data": ["Technology", "Healthcare"]
    }
    mock_marketaux_class.return_value = mock_marketaux
    
    # Call function
    result = get_industry_list()
    
    # Check results
    assert len(result) == 2
    assert "Technology" in result
    assert "Healthcare" in result
    
    # Verify mock was called
    mock_marketaux.get_industry_list.assert_called_once()


@patch('financial_news_rag.data.MarketauxNewsFetcher')
def test_get_news_sources(mock_marketaux_class):
    """Test fetching news sources."""
    # Set up mock
    mock_marketaux = MagicMock()
    mock_marketaux.get_news_sources.return_value = {
        "data": [
            {"domain": "example.com", "language": "en"},
            {"domain": "news.com", "language": "en"}
        ]
    }
    mock_marketaux_class.return_value = mock_marketaux
    
    # Call function
    result = get_news_sources(language=["en"], distinct_domain=True)
    
    # Check results
    assert len(result["data"]) == 2
    assert result["data"][0]["domain"] == "example.com"
    
    # Verify mock was called correctly
    mock_marketaux.get_news_sources.assert_called_once()
    call_kwargs = mock_marketaux.get_news_sources.call_args[1]
    assert call_kwargs["language"] == ["en"]
    assert call_kwargs["distinct_domain"] == True
