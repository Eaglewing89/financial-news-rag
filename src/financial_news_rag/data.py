"""
Data management module for the Financial News RAG system.

This module handles fetching, storing, and managing news article data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from financial_news_rag.config import get_config
from financial_news_rag.marketaux import MarketauxNewsFetcher
from financial_news_rag.utils import safe_get

# Configure module-level logger
logger = logging.getLogger("financial_news_rag")


def fetch_marketaux_news_snippets(
    symbols: Optional[List[str]] = None,
    sentiment_gte: Optional[float] = None,
    sentiment_lte: Optional[float] = None,
    entity_types: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    countries: Optional[List[str]] = None,
    search: Optional[str] = None,
    must_have_entities: bool = False,
    filter_entities: bool = True,
    language: Optional[List[str]] = None,
    days_back: Optional[int] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    limit: Optional[int] = None,
    page: int = 1,
    api_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch financial news articles from the Marketaux API.
    
    Args:
        symbols: List of stock symbols to filter by.
        sentiment_gte: Minimum sentiment score (-1 to 1).
        sentiment_lte: Maximum sentiment score (-1 to 1).
        entity_types: List of entity types (e.g., ["equity", "index"]).
        industries: List of industries (e.g., ["Technology", "Healthcare"]).
        countries: List of country codes (e.g., ["us", "ca"]).
        search: Search query string.
        must_have_entities: If True, return only articles with entities.
        filter_entities: If True, filter entities to only those matching query.
        language: List of language codes (e.g., ["en", "fr"]).
        days_back: Number of days to look back for news.
        date_range: Tuple of (start_date, end_date) for more specific date filtering.
        limit: Maximum number of articles to return.
        page: Page number for pagination.
        api_token: Optional override for API token.
        
    Returns:
        Dictionary containing the API response with articles and metadata.
    """
    # Get configuration
    config = get_config()
    
    # Initialize API client
    api_token = api_token or config.get("MARKETAUX_API_KEY")
    marketaux = MarketauxNewsFetcher(api_token=api_token)
    
    # Handle date parameters
    published_after = None
    published_before = None
    
    if days_back:
        published_after = datetime.now() - timedelta(days=days_back)
    
    if date_range:
        published_after = date_range[0]
        published_before = date_range[1]
    
    # Default to English if not specified
    if not language:
        language = ["en"]
    
    # Make API request
    try:
        response = marketaux.fetch_news(
            symbols=symbols,
            sentiment_gte=sentiment_gte,
            sentiment_lte=sentiment_lte,
            entity_types=entity_types,
            industries=industries,
            countries=countries,
            search=search,
            must_have_entities=must_have_entities,
            filter_entities=filter_entities,
            language=language,
            published_after=published_after,
            published_before=published_before,
            limit=limit,
            page=page
        )
        
        # Extract key metrics for logging
        articles_found = safe_get(response, "meta.found", 0)
        articles_returned = safe_get(response, "meta.returned", 0)
        
        # Log success
        logger.info(
            f"Fetched {articles_returned} articles (found {articles_found} total) "
            f"matching query. Symbols: {symbols}, Search: {search}"
        )
        
        return response
    
    except Exception as e:
        # Log error
        logger.error(f"Error fetching news: {str(e)}")
        # Re-raise for caller to handle
        raise


def normalize_news_data(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize news article data for storage in the database.
    
    Args:
        articles: List of raw article data from Marketaux API.
        
    Returns:
        List of normalized article dictionaries.
    """
    # Initialize Marketaux fetcher to use its normalization method
    marketaux = MarketauxNewsFetcher(api_token="dummy")  # Token not used for normalization
    
    # Use the MarketauxNewsFetcher's normalize_article_data method
    return marketaux.normalize_article_data(articles)


def search_entities(
    search: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    entity_types: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    countries: Optional[List[str]] = None,
    page: int = 1,
    api_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search for financial entities (companies, indices, etc.) via the Marketaux API.
    
    Args:
        search: Search string to find entities.
        symbols: Specific symbols to return.
        entity_types: Filter by entity types.
        industries: Filter by industries.
        countries: Filter by country codes.
        page: Page number for pagination.
        api_token: Optional override for API token.
        
    Returns:
        Dictionary containing entity search results.
    """
    # Get configuration
    config = get_config()
    
    # Initialize API client
    api_token = api_token or config.get("MARKETAUX_API_KEY")
    marketaux = MarketauxNewsFetcher(api_token=api_token)
    
    try:
        return marketaux.search_entities(
            search=search,
            symbols=symbols,
            types=entity_types,
            industries=industries,
            countries=countries,
            page=page
        )
    except Exception as e:
        logger.error(f"Error searching entities: {str(e)}")
        raise


def get_entity_types(api_token: Optional[str] = None) -> List[str]:
    """
    Get all supported entity types from the Marketaux API.
    
    Args:
        api_token: Optional override for API token.
        
    Returns:
        List of entity type strings.
    """
    # Get configuration
    config = get_config()
    
    # Initialize API client
    api_token = api_token or config.get("MARKETAUX_API_KEY")
    marketaux = MarketauxNewsFetcher(api_token=api_token)
    
    try:
        result = marketaux.get_entity_types()
        return result.get("data", [])
    except Exception as e:
        logger.error(f"Error getting entity types: {str(e)}")
        raise


def get_industry_list(api_token: Optional[str] = None) -> List[str]:
    """
    Get all supported industries from the Marketaux API.
    
    Args:
        api_token: Optional override for API token.
        
    Returns:
        List of industry strings.
    """
    # Get configuration
    config = get_config()
    
    # Initialize API client
    api_token = api_token or config.get("MARKETAUX_API_KEY")
    marketaux = MarketauxNewsFetcher(api_token=api_token)
    
    try:
        result = marketaux.get_industry_list()
        return result.get("data", [])
    except Exception as e:
        logger.error(f"Error getting industry list: {str(e)}")
        raise


def get_news_sources(
    distinct_domain: bool = False,
    language: Optional[List[str]] = None,
    page: int = 1,
    api_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get available news sources from the Marketaux API.
    
    Args:
        distinct_domain: If True, group distinct domains.
        language: Filter by language codes.
        page: Page number for pagination.
        api_token: Optional override for API token.
        
    Returns:
        Dictionary containing news sources.
    """
    # Get configuration
    config = get_config()
    
    # Initialize API client
    api_token = api_token or config.get("MARKETAUX_API_KEY")
    marketaux = MarketauxNewsFetcher(api_token=api_token)
    
    try:
        return marketaux.get_news_sources(
            distinct_domain=distinct_domain,
            language=language,
            page=page
        )
    except Exception as e:
        logger.error(f"Error getting news sources: {str(e)}")
        raise
