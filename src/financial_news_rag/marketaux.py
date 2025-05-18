"""
Marketaux API Integration Module for Financial News RAG.

This module provides functionality to fetch financial news articles from the Marketaux API,
handling authentication, request parameters, error handling, and rate limiting.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import requests
from dotenv import load_dotenv


class MarketauxRateLimiter:
    """
    Implements a simple in-memory rate limiter to respect Marketaux API limits.
    
    Attributes:
        calls_per_minute (int): Maximum allowed API calls per minute.
        call_times (List[float]): Timestamps of recent API calls for tracking.
    """
    
    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize rate limiter with calls per minute limit.
        
        Args:
            calls_per_minute (int): Maximum allowed API calls per minute.
        """
        self.calls_per_minute = calls_per_minute
        self.call_times = []
    
    def wait_if_needed(self) -> None:
        """
        Check if we need to wait before making a new API call.
        Implements a sliding window approach to stay within rate limits.
        """
        now = time.time()
        # Remove calls older than 60 seconds
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        if len(self.call_times) >= self.calls_per_minute:
            # Find the oldest call within the current 60s window
            oldest_call_in_window = self.call_times[0]
            sleep_time = 60 - (now - oldest_call_in_window)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Record this call
        self.call_times.append(time.time())


def fetch_marketaux_with_retry(url: str, params: Dict[str, Any], max_retries: int = 3, 
                    backoff_factor: float = 1.5, timeout: int = 10) -> Dict[str, Any]:
    """
    Fetch data from an API with exponential backoff retry logic.
    
    Args:
        url: API endpoint URL.
        params: Query parameters for the request.
        max_retries: Maximum number of retry attempts.
        backoff_factor: Multiplier for exponential backoff.
        timeout: Request timeout in seconds.
        
    Returns:
        Dict containing the JSON response data.
        
    Raises:
        Exception: If all retry attempts fail.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            # Check if we've used all retry attempts
            if attempt == max_retries - 1:
                error_msg = f"Failed after {max_retries} attempts: {str(e)}"
                # Include response details if available
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    error_msg += f" (Status code: {status_code})"
                    if e.response.text:
                        try:
                            error_data = e.response.json()
                            error_msg += f", API response: {error_data}"
                        except ValueError:
                            error_msg += f", Response text: {e.response.text[:200]}"
                
                raise Exception(error_msg)
            
            # Exponential backoff sleep
            time.sleep(backoff_factor ** attempt)


class MarketauxNewsFetcher:
    """
    Client for fetching financial news articles from the Marketaux API.
    
    Implements robust error handling, rate limiting, and parameter validation
    for all Marketaux API interactions.
    
    Attributes:
        api_token (str): Marketaux API authentication token.
        base_url (str): Base URL for Marketaux API endpoints.
        rate_limiter (MarketauxRateLimiter): Rate limiting instance to prevent API throttling.
    """
    
    # API endpoint URLs
    NEWS_API_URL = "https://api.marketaux.com/v1/news/all"
    ENTITY_SEARCH_URL = "https://api.marketaux.com/v1/entity/search"
    ENTITY_TYPE_URL = "https://api.marketaux.com/v1/entity/type/list"
    INDUSTRY_LIST_URL = "https://api.marketaux.com/v1/entity/industry/list"
    SOURCES_URL = "https://api.marketaux.com/v1/news/sources"
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the Marketaux client with API token.
        
        Args:
            api_token: Optional API token override. If not provided, 
                       loads from environment variable.
        
        Raises:
            ValueError: If API token cannot be found.
        """
        # Load environment variables if token not explicitly provided
        if api_token is None:
            load_dotenv()
            api_token = os.getenv("MARKETAUX_API_KEY")
        
        if not api_token:
            raise ValueError(
                "Marketaux API token is required. Set MARKETAUX_API_KEY in .env file "
                "or pass api_token to the constructor."
            )
        
        self.api_token = api_token
        self.rate_limiter = MarketauxRateLimiter()
    
    def fetch_news(self, 
                  symbols: Optional[List[str]] = None,
                  entity_types: Optional[List[str]] = None, 
                  industries: Optional[List[str]] = None,
                  countries: Optional[List[str]] = None,
                  sentiment_gte: Optional[float] = None,
                  sentiment_lte: Optional[float] = None,
                  min_match_score: Optional[float] = None,
                  filter_entities: bool = False,
                  must_have_entities: bool = False,
                  group_similar: bool = True,
                  search: Optional[str] = None,
                  domains: Optional[List[str]] = None,
                  exclude_domains: Optional[List[str]] = None,
                  source_ids: Optional[List[str]] = None,
                  exclude_source_ids: Optional[List[str]] = None,
                  language: Optional[List[str]] = None,
                  published_before: Optional[Union[str, datetime]] = None,
                  published_after: Optional[Union[str, datetime]] = None,
                  published_on: Optional[Union[str, datetime]] = None,
                  sort: Optional[str] = None,
                  sort_order: str = "desc",
                  limit: Optional[int] = None,
                  page: int = 1) -> Dict[str, Any]:
        """
        Fetch financial news articles from Marketaux API with comprehensive filtering options.
        
        Args:
            symbols: List of entity symbols (e.g., ["TSLA", "AMZN", "MSFT"]).
            entity_types: List of entity types (e.g., ["index", "equity"]).
            industries: List of industries (e.g., ["Technology", "Industrials"]).
            countries: List of country codes for exchanges (e.g., ["us", "ca"]).
            sentiment_gte: Minimum sentiment score (-1 to 1).
            sentiment_lte: Maximum sentiment score (-1 to 1).
            min_match_score: Minimum entity match score.
            filter_entities: If True, only relevant entities to query are returned per article.
            must_have_entities: If True, only articles with at least one entity are returned.
            group_similar: If True, group similar articles.
            search: Search terms or advanced query (AND/OR/NOT, phrase, prefix, precedence).
            domains: List of domains to include.
            exclude_domains: List of domains to exclude.
            source_ids: List of source IDs to include.
            exclude_source_ids: List of source IDs to exclude.
            language: List of language codes (e.g., ["en", "es"]).
            published_before: Find articles published before this date/time.
            published_after: Find articles published after this date/time.
            published_on: Find articles published on this specific date.
            sort: Sort by "published_on", "entity_match_score", "entity_sentiment_score", or "relevance_score".
            sort_order: "desc" or "asc" (only with entity_match_score or entity_sentiment_score).
            limit: Number of articles to return (max depends on API plan).
            page: Page number for pagination.
            
        Returns:
            Dict containing response data with articles and metadata.
            
        Raises:
            Exception: For API errors or failed retries.
        """
        # Parameter validation and normalization
        params = {"api_token": self.api_token}
        
        # Convert lists to comma-separated strings as required by the API
        if symbols:
            params["symbols"] = ",".join(symbols)
        
        if entity_types:
            params["entity_types"] = ",".join(entity_types)
            
        if industries:
            params["industries"] = ",".join(industries)
            
        if countries:
            params["countries"] = ",".join(countries)
            
        if sentiment_gte is not None:
            params["sentiment_gte"] = sentiment_gte
            
        if sentiment_lte is not None:
            params["sentiment_lte"] = sentiment_lte
            
        if min_match_score is not None:
            params["min_match_score"] = min_match_score
            
        if filter_entities:
            params["filter_entities"] = "true"
            
        if must_have_entities:
            params["must_have_entities"] = "true"
            
        if not group_similar:
            params["group_similar"] = "false"
            
        if search:
            params["search"] = search
            
        if domains:
            params["domains"] = ",".join(domains)
            
        if exclude_domains:
            params["exclude_domains"] = ",".join(exclude_domains)
            
        if source_ids:
            params["source_ids"] = ",".join(source_ids)
            
        if exclude_source_ids:
            params["exclude_source_ids"] = ",".join(exclude_source_ids)
            
        if language:
            params["language"] = ",".join(language)
        
        # Handle date parameters with datetime conversion if needed
        for date_param, date_value in [
            ("published_before", published_before),
            ("published_after", published_after),
            ("published_on", published_on)
        ]:
            if date_value:
                if isinstance(date_value, datetime):
                    # Convert datetime to ISO format for API
                    if date_param == "published_on":
                        # For published_on we only need the date part
                        params[date_param] = date_value.strftime("%Y-%m-%d")
                    else:
                        # For before/after we include time
                        params[date_param] = date_value.isoformat()
                else:
                    # Assume it's already a properly formatted string
                    params[date_param] = date_value
        
        if sort:
            valid_sort_options = ["published_on", "entity_match_score", 
                                "entity_sentiment_score", "relevance_score"]
            if sort not in valid_sort_options:
                raise ValueError(f"Invalid sort option. Must be one of: {valid_sort_options}")
            params["sort"] = sort
            
        if sort_order and sort_order.lower() in ["asc", "desc"]:
            params["sort_order"] = sort_order.lower()
        
        if limit:
            params["limit"] = limit
            
        params["page"] = page
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Make API request with retry logic
        return fetch_marketaux_with_retry(self.NEWS_API_URL, params)
    
    def search_entities(self,
                       search: Optional[str] = None,
                       symbols: Optional[List[str]] = None,
                       exchanges: Optional[List[str]] = None,
                       types: Optional[List[str]] = None,
                       industries: Optional[List[str]] = None,
                       countries: Optional[List[str]] = None,
                       page: int = 1) -> Dict[str, Any]:
        """
        Search for entities (companies, indices, currencies) via Marketaux API.
        
        Args:
            search: Search string to find entities.
            symbols: Specific symbols to return.
            exchanges: Filter by exchanges (comma-separated).
            types: Filter by entity types (comma-separated).
            industries: Filter by industries (comma-separated).
            countries: Filter by country codes (comma-separated).
            page: Page number for pagination.
            
        Returns:
            Dict containing response data with entities and metadata.
            
        Raises:
            Exception: For API errors or failed retries.
        """
        params = {"api_token": self.api_token, "page": page}
        
        if search:
            params["search"] = search
            
        if symbols:
            params["symbols"] = ",".join(symbols)
            
        if exchanges:
            params["exchanges"] = ",".join(exchanges)
            
        if types:
            params["types"] = ",".join(types)
            
        if industries:
            params["industries"] = ",".join(industries)
            
        if countries:
            params["countries"] = ",".join(countries)
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Make API request with retry logic
        return fetch_marketaux_with_retry(self.ENTITY_SEARCH_URL, params)
    
    def get_entity_types(self) -> Dict[str, Any]:
        """
        Retrieve all supported entity types.
        
        Returns:
            Dict containing the list of supported entity types.
            
        Raises:
            Exception: For API errors or failed retries.
        """
        params = {"api_token": self.api_token}
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Make API request with retry logic
        return fetch_marketaux_with_retry(self.ENTITY_TYPE_URL, params)
    
    def get_industry_list(self) -> Dict[str, Any]:
        """
        Retrieve all supported entity industries.
        
        Returns:
            Dict containing the list of supported industries.
            
        Raises:
            Exception: For API errors or failed retries.
        """
        params = {"api_token": self.api_token}
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Make API request with retry logic
        return fetch_marketaux_with_retry(self.INDUSTRY_LIST_URL, params)
    
    def get_news_sources(self,
                        distinct_domain: bool = False,
                        language: Optional[List[str]] = None,
                        page: int = 1) -> Dict[str, Any]:
        """
        View available news sources for use in other API requests.
        
        Args:
            distinct_domain: If True, group distinct domains (source_id will be null).
            language: List of language codes (e.g., ["en", "es"]).
            page: Page number for pagination.
            
        Returns:
            Dict containing response data with sources and metadata.
            
        Raises:
            Exception: For API errors or failed retries.
        """
        params = {"api_token": self.api_token, "page": page}
        
        if distinct_domain:
            params["distinct_domain"] = "true"
            
        if language:
            params["language"] = ",".join(language)
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Make API request with retry logic
        return fetch_marketaux_with_retry(self.SOURCES_URL, params)
    
    def normalize_article_data(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize the article data from Marketaux API to a consistent format.
        
        Args:
            articles: List of raw article data from Marketaux API.
            
        Returns:
            List of normalized article dictionaries.
        """
        normalized_articles = []
        
        for article in articles:
            normalized = {
                "uuid": article.get("uuid", ""),
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "snippet": article.get("snippet", ""),
                "content": article.get("snippet", ""),  # Use snippet as content for now
                "url": article.get("url", ""),
                "image_url": article.get("image_url", ""),
                "language": article.get("language", ""),
                "published_at": article.get("published_at", ""),
                "source": article.get("source", ""),
                "relevance_score": article.get("relevance_score"),
                "entities": article.get("entities", []),
                "similar": article.get("similar", [])
            }
            
            # Additional processing of entities if needed
            if normalized["entities"]:
                for entity in normalized["entities"]:
                    # Ensure all required fields are present
                    entity.setdefault("symbol", "")
                    entity.setdefault("name", "")
                    entity.setdefault("type", "")
                    entity.setdefault("industry", "")
                    entity.setdefault("sentiment_score", 0.0)
                    entity.setdefault("match_score", 0.0)
            
            normalized_articles.append(normalized)
        
        return normalized_articles
