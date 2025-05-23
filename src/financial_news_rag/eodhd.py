"""
EODHD API Client Module

This module provides functionality to fetch financial news articles from the EODHD API.
It includes robust error handling, rate limiting, and normalization of API responses.
"""

import os
import time
import warnings
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_URL = 'https://eodhd.com/api/news'
DEFAULT_TIMEOUT = 100  # Increased timeout due to potential delays as per documentation
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.5
DEFAULT_LIMIT = 50  # Default articles per request

class EODHDApiError(Exception):
    """Custom exception for EODHD API errors."""
    pass

class EODHDClient:
    """
    Client for interacting with the EODHD API to fetch financial news articles.
    
    This client handles authentication, request construction, response parsing,
    error handling, and rate limiting for the EODHD financial news API.
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the EODHD API client.
        
        Args:
            api_key: EODHD API key. If None, will look for EODHD_API_KEY in environment variables.
            timeout: Request timeout in seconds.
        
        Raises:
            ValueError: If no API key is provided or found in environment variables.
        """
        self.api_key = api_key or os.getenv('EODHD_API_KEY')
        if not self.api_key:
            raise ValueError("EODHD API key not found. Please provide it or set EODHD_API_KEY environment variable.")
        
        self.timeout = timeout
    
    def fetch_news(
        self,
        symbols: Optional[Union[str, List[str]]] = None,
        tag: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    ) -> List[Dict]:
        """
        Fetch financial news articles from EODHD API.
        
        Args:
            symbols: Ticker symbol(s) to filter news for. Can be a string (single symbol), a comma-separated string, 
                   or a list of symbols. If a list or a comma-separated string is provided, a warning will be triggered, 
                   and only the first symbol will be used.
            tag: Topic tag to filter news for (e.g., "mergers and acquisitions").
            from_date: Start date for filtering news (YYYY-MM-DD).
            to_date: End date for filtering news (YYYY-MM-DD).
            limit: Number of results to return (default: 50, min: 1, max: 1000).
            offset: Offset for pagination (default: 0).
            max_retries: Maximum number of retry attempts for failed requests.
            backoff_factor: Backoff factor for retry timing (seconds = backoff_factor * (2 ** attempt)).
        
        Returns:
            List of news article dictionaries with normalized fields.
            
        Raises:
            ValueError: If neither symbols nor tag is provided.
            ValueError: If limit is outside the valid range of 1-1000.
            ValueError: If from_date or to_date has an invalid format.
            EODHDApiError: If the API request fails after retries.
        """
        # Validate limit parameter
        if not 1 <= limit <= 1000:
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
        if not symbols and not tag:
            raise ValueError("Either 'symbols' or 'tag' must be provided.")
        
        # Handle symbols parameter
        if isinstance(symbols, list):
            if len(symbols) > 1:
                warnings.warn(
                    "EODHD API only supports a single symbol despite official documentation. "
                    "Only the first symbol will be used. For multiple symbols, make separate API calls for each.",
                    UserWarning
                )
                symbols = symbols[0]  # Take only the first symbol
            else:
                symbols = symbols[0] if symbols else None
        elif symbols and ',' in symbols:
            warnings.warn(
                "EODHD API only supports a single symbol despite official documentation. "
                "Only the first symbol will be used. For multiple symbols, make separate API calls for each.",
                UserWarning
            )
            symbols = symbols.split(',')[0].strip()  # Take only the first symbol before any comma
        
        params = {
            'api_token': self.api_key,
            'fmt': 'json',
            'limit': limit,
            'offset': offset
        }
        
        if tag:
            params['t'] = tag
        elif symbols:
            params['s'] = symbols
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        # Fetch with retry
        raw_articles = self._fetch_with_retry(API_URL, params, max_retries, backoff_factor)
        
        # Normalize each article
        normalized_articles = []
        for article in raw_articles:
            # Pass query type and value based on the parameters used
            if 't' in params:
                normalized_article = self._normalize_article(article, query_type='tag', query_value=params['t'])
            elif 's' in params:
                normalized_article = self._normalize_article(article, query_type='symbol', query_value=params['s'])
            else:
                normalized_article = self._normalize_article(article)
            
            normalized_articles.append(normalized_article)
        
        return normalized_articles
    
    def _fetch_with_retry(
        self, 
        url: str, 
        params: Dict, 
        max_retries: int = DEFAULT_MAX_RETRIES, 
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    ) -> List[Dict]:
        """
        Fetch data from the API with retry logic and exponential backoff.
        
        Args:
            url: API endpoint URL.
            params: Query parameters for the request.
            max_retries: Maximum number of retry attempts.
            backoff_factor: Backoff factor for retry timing.
            
        Returns:
            JSON response from the API as a list of dictionaries.
            
        Raises:
            EODHDApiError: If all retry attempts fail.
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Check if response is empty
                if not response.text.strip():
                    raise ValueError("Empty response received from API")
                
                # Log response for debugging if needed
                # print(f"Response text: {response.text[:100]}...")  # Uncomment for debugging
                
                # Parse JSON response
                try:
                    data = response.json()
                except requests.exceptions.JSONDecodeError as json_err:
                    # Additional error info for debugging JSON decode issues
                    raise ValueError(f"JSON decode error: {str(json_err)}. Response content: {response.text[:100]}...")
                
                # Check if the response is an error message
                if isinstance(data, dict) and 'code' in data and 'message' in data:
                    raise EODHDApiError(f"EODHD API Error: {data['message']} (Code: {data['code']})")
                
                # If we received an empty list, it's valid but means no articles found
                if data == [] and 's' in params:
                    logging.info(f"No articles found for symbols: {params['s']}")
                
                return data if isinstance(data, list) else []
                
            except (requests.exceptions.RequestException, ValueError) as e:
                # Log the error with attempt information
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                logging.error(error_msg)  # Log the error message
                
                # If this was the last attempt, raise the error
                if attempt == max_retries - 1:
                    raise EODHDApiError(f"Failed after {max_retries} attempts: {str(e)}")
                
                # Exponential backoff before next attempt
                sleep_time = backoff_factor * (2 ** attempt)
                time.sleep(sleep_time)
        
        # This should not be reached if max_retries > 0
        raise EODHDApiError("Failed to fetch data from EODHD API.")
    
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
