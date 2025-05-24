"""
EODHD API Client Module

This module provides functionality to fetch financial news articles from the EODHD API.
It includes robust error handling, rate limiting, and normalization of API responses.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests


class EODHDApiError(Exception):
    """Custom exception for EODHD API errors."""
    pass


class EODHDClient:
    """
    Client for interacting with the EODHD API to fetch financial news articles.
    
    This client handles authentication, request construction, response parsing,
    error handling, and rate limiting for the EODHD financial news API.
    """
    
    def __init__(
        self,
        api_key: str,
        api_url: str = 'https://eodhd.com/api/news',
        default_timeout: int = 100,
        default_max_retries: int = 3,
        default_backoff_factor: float = 1.5,
        default_limit: int = 50
    ):
        """
        Initialize the EODHD API client.
        
        Args:
            api_key: EODHD API key (required).
            api_url: EODHD API endpoint URL.
            default_timeout: Default request timeout in seconds.
            default_max_retries: Default maximum number of retry attempts for failed requests.
            default_backoff_factor: Default backoff factor for retry timing.
            default_limit: Default number of articles to return per request.
        
        Raises:
            ValueError: If no API key is provided.
        """
        if not api_key:
            raise ValueError("EODHD API key is required.")
        
        self.api_key = api_key
        self.api_url = api_url
        self.default_timeout = default_timeout
        self.default_max_retries = default_max_retries
        self.default_backoff_factor = default_backoff_factor
        self.default_limit = default_limit
    
    def fetch_news(
        self,
        symbol: Optional[str] = None,
        tag: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        max_retries: Optional[int] = None,
        backoff_factor: Optional[float] = None
    ) -> Dict:
        """
        Fetch financial news articles from EODHD API.
        
        Args:
            symbol: A single ticker symbol to filter news for (e.g., "AAPL.US").
            tag: Topic tag to filter news for (e.g., "mergers and acquisitions").
            from_date: Start date for filtering news (YYYY-MM-DD).
            to_date: End date for filtering news (YYYY-MM-DD).
            limit: Number of results to return (min: 1, max: 1000).
            offset: Offset for pagination (default: 0).
            max_retries: Maximum number of retry attempts for failed requests.
            backoff_factor: Backoff factor for retry timing (seconds = backoff_factor * (2 ** attempt)).
        
        Returns:
            Dictionary containing:
            - "articles": List of normalized article dictionaries
            - "status_code": HTTP status code of the API response
            - "success": Boolean indicating if the API call was successful
            - "error_message": Error message if the call failed, otherwise None
            
        Raises:
            ValueError: If neither symbols nor tag is provided.
            ValueError: If limit is outside the valid range of 1-1000.
            ValueError: If from_date or to_date has an invalid format.
            EODHDApiError: If the API request fails after retries.
        """
        # Use instance defaults if parameters not provided
        limit_to_use = limit if limit is not None else self.default_limit
        max_retries_to_use = max_retries if max_retries is not None else self.default_max_retries
        backoff_factor_to_use = backoff_factor if backoff_factor is not None else self.default_backoff_factor
        
        # Validate limit parameter
        if not 1 <= limit_to_use <= 1000:
            raise ValueError("'limit' must be between 1 and 1000")
        
        # Validate date format
        date_format = "%Y-%m-%d"
        if from_date:
            try:
                datetime.strptime(from_date, date_format)
            except ValueError:
                raise ValueError("'from_date' must be in YYYY-MM-DD format")
        
        if to_date:
            try:
                datetime.strptime(to_date, date_format)
            except ValueError:
                raise ValueError("'to_date' must be in YYYY-MM-DD format")
        
        # Construct parameters
        if not symbol and not tag:
            raise ValueError("Either 'symbol' or 'tag' must be provided.")
        
        params = {
            'api_token': self.api_key,
            'fmt': 'json',
            'limit': limit_to_use,
            'offset': offset
        }
        
        if tag:
            params['t'] = tag
        elif symbol:
            params['s'] = symbol
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        # Fetch with retry
        raw_articles = self._fetch_with_retry(self.api_url, params, max_retries_to_use, backoff_factor_to_use)
        
        # Normalize each article
        normalized_articles = []
        for article in raw_articles.get("articles", []):
            # Pass query type and value based on the parameters used
            if 't' in params:
                normalized_article = self._normalize_article(article, query_type='tag', query_value=params['t'])
            elif 's' in params:
                normalized_article = self._normalize_article(article, query_type='symbol', query_value=params['s'])
            else:
                normalized_article = self._normalize_article(article)
            
            normalized_articles.append(normalized_article)
        
        # Update the result dictionary with normalized articles
        raw_articles["articles"] = normalized_articles
        
        return raw_articles
    
    def _fetch_with_retry(
        self, 
        url: str, 
        params: Dict, 
        max_retries: int, 
        backoff_factor: float
    ) -> Dict:
        """
        Fetch data from the API with retry logic and exponential backoff.
        
        Args:
            url: API endpoint URL.
            params: Query parameters for the request.
            max_retries: Maximum number of retry attempts.
            backoff_factor: Backoff factor for retry timing.
            
        Returns:
            Dictionary containing:
            - "articles": List of article dictionaries (or empty list)
            - "status_code": HTTP status code (int)
            - "success": Boolean indicating if the API call was successful
            - "error_message": Error message string or None if successful
        """
        result = {
            "articles": [],
            "status_code": None,
            "success": False,
            "error_message": None
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.default_timeout)
                
                # Capture the status code
                result["status_code"] = response.status_code
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Check if response is empty
                if not response.text.strip():
                    error_msg = "Empty response received from API"
                    result["error_message"] = error_msg
                    raise ValueError(error_msg)
                
                # Parse JSON response
                try:
                    data = response.json()
                except requests.exceptions.JSONDecodeError as json_err:
                    # Additional error info for debugging JSON decode issues
                    error_msg = f"JSON decode error: {str(json_err)}. Response content: {response.text[:100]}..."
                    result["error_message"] = error_msg
                    raise ValueError(error_msg)
                
                # Check if the response is an error message
                if isinstance(data, dict) and 'code' in data and 'message' in data:
                    result["error_message"] = f"EODHD API Error: {data['message']} (Code: {data['code']})"
                    result["success"] = False
                    return result
                
                # If we received an empty list, it's valid but means no articles found
                if data == [] and 's' in params:
                    logging.info(f"No articles found for symbols: {params['s']}")
                
                result["articles"] = data if isinstance(data, list) else []
                result["success"] = True
                return result
                
            except (requests.exceptions.RequestException, ValueError) as e:
                # Log the error with attempt information
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                logging.error(error_msg)  # Log the error message
                
                # Keep track of the latest error message
                result["error_message"] = str(e)
                
                # If this was the last attempt, return the failed result
                if attempt == max_retries - 1:
                    return result
                
                # Exponential backoff before next attempt
                sleep_time = backoff_factor * (2 ** attempt)
                time.sleep(sleep_time)
        
        # This should not be reached if max_retries > 0, but included for safety
        result["error_message"] = "Failed to fetch data from EODHD API."
        return result
    
    def _normalize_article(self, article: Dict, query_type: Optional[str] = None, query_value: Optional[str] = None) -> Dict:
        """
        Normalize an article from the EODHD API response.
        
        Args:
            article: Raw article dictionary from the API response.
            query_type: Type of query used to fetch the article ('tag' or 'symbol').
            query_value: Value of the query used to fetch the article.
            
        Returns:
            Normalized article dictionary with consistent field names.
        """
        # Normalize date format
        try:
            # Parse the ISO 8601 date
            published_at = datetime.fromisoformat(article['date'].replace('Z', '+00:00'))
            # Store it in ISO format for consistent representation
            published_at_iso = published_at.isoformat()
        except (KeyError, ValueError):
            # Fallback if date is missing or invalid
            published_at_iso = datetime.now(timezone.utc).isoformat()
        
        # Build normalized article structure
        normalized = {
            'title': article.get('title', ''),
            'raw_content': article.get('content', ''),
            'url': article.get('link', ''),
            'published_at': published_at_iso,
            'source_api': 'EODHD',
            'symbols': article.get('symbols', []),
            'tags': article.get('tags', []),
            'sentiment': article.get('sentiment', {})
        }
        
        # Add source query information based on query type
        if query_type == 'tag' and query_value:
            normalized['source_query_tag'] = query_value
        elif query_type == 'symbol' and query_value:
            normalized['source_query_symbol'] = query_value
        
        return normalized
