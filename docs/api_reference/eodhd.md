# EODHD API Client (`eodhd.py`)

The `eodhd.py` module provides the `EODHDClient` class, which is responsible for fetching financial news articles from the EOD Historical Data (EODHD) API. It handles API authentication, request construction, response parsing, error handling (including retries with exponential backoff), and normalization of the data received from the API.

## `EODHDApiError` Exception

```python
class EODHDApiError(Exception):
    """Custom exception for EODHD API errors."""
    pass
```
A custom exception class used to indicate errors specific to the EODHD API interactions.

## `EODHDClient` Class

This class is the main interface for interacting with the EODHD news API.

### Initialization

```python
class EODHDClient:
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://eodhd.com/api/news",
        default_timeout: int = 30,
        default_max_retries: int = 3,
        default_backoff_factor: float = 1.5,
        default_limit: int = 50,
    ):
        """
        Initialize the EODHD API client.
        ...
        """
        # ...
```

The constructor initializes the `EODHDClient`.

**Parameters:**

*   `api_key` (str): Your EODHD API key. This is **required**.
*   `api_url` (str, optional): The base URL for the EODHD news API. Defaults to `"https://eodhd.com/api/news"`.
*   `default_timeout` (int, optional): The default timeout in seconds for API requests. Defaults to 30.
*   `default_max_retries` (int, optional): The default maximum number of times to retry a failed API request. Defaults to 3.
*   `default_backoff_factor` (float, optional): The default factor used to calculate the delay between retries (delay = `backoff_factor * (2 ** attempt_number)`). Defaults to 1.5.
*   `default_limit` (int, optional): The default number of articles to request per API call if not specified in `fetch_news`. Defaults to 50.

**Raises:**

*   `ValueError`: If `api_key` is not provided.

### Methods

#### `fetch_news(symbol: Optional[str] = None, tag: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: Optional[int] = None, offset: int = 0, max_retries: Optional[int] = None, backoff_factor: Optional[float] = None) -> Dict`

```python
def fetch_news(
    self,
    symbol: Optional[str] = None,
    tag: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
) -> Dict:
    """
    Fetch financial news articles from EODHD API.
    ...
    Returns:
        Dictionary containing:
        - "articles": List of normalized article dictionaries
        - "status_code": HTTP status code of the API response
        - "success": Boolean indicating if the API call was successful
        - "error_message": Error message if the call failed, otherwise None
    """
    # ...
```
Fetches news articles from the EODHD API based on the provided filters.

**Parameters:**

*   `symbol` (Optional[str], optional): A single stock ticker symbol (e.g., "AAPL.US") to filter news for. Cannot be used with `tag`.
*   `tag` (Optional[str], optional): A topic tag (e.g., "mergers and acquisitions") to filter news for. Cannot be used with `symbol`.
*   `from_date` (Optional[str], optional): The start date for filtering news, in "YYYY-MM-DD" format.
*   `to_date` (Optional[str], optional): The end date for filtering news, in "YYYY-MM-DD" format.
*   `limit` (Optional[int], optional): The number of articles to return. Must be between 1 and 1000. If not provided, `default_limit` is used.
*   `offset` (int, optional): The offset for pagination. Defaults to 0.
*   `max_retries` (Optional[int], optional): Overrides `default_max_retries` for this specific call.
*   `backoff_factor` (Optional[float], optional): Overrides `default_backoff_factor` for this specific call.

**Returns:**

*   `Dict`: A dictionary with the following keys:
    *   `"articles"` (List[Dict]): A list of normalized article dictionaries. Each dictionary represents an article and contains fields like `title`, `raw_content`, `url`, `published_at`, `source_api`, `symbols`, `tags`, `sentiment`, `source_query_tag` (if `tag` was used), and `source_query_symbol` (if `symbol` was used).
    *   `"status_code"` (int or None): The HTTP status code of the API response.
    *   `"success"` (bool): `True` if the API call was successful and data was retrieved, `False` otherwise.
    *   `"error_message"` (str or None): An error message if the call failed, otherwise `None`.

**Raises:**

*   `ValueError`: If neither `symbol` nor `tag` is provided, if both are provided, if `limit` is out of range (1-1000), or if `from_date` or `to_date` has an invalid format.
*   `EODHDApiError`: If the API request ultimately fails after all retry attempts (though typically, the method returns a dictionary with `success: False` in such cases).

#### Normalization

The `_normalize_article` internal method is used to transform the raw article data from the EODHD API into a standardized format. This includes:

*   **Date Parsing:** Converts the `date` field from the API (which can be in various formats) into a consistent ISO 8601 format (`YYYY-MM-DDTHH:MM:SS.ffffff`). If parsing fails or the date is missing, it defaults to the current UTC time.
*   **Field Mapping:** Maps API fields to a consistent schema:
    *   `title` -> `title`
    *   `content` -> `raw_content`
    *   `link` -> `url`
    *   `date` -> `published_at` (normalized)
    *   Hardcoded `source_api`: "EODHD"
    *   `symbols` -> `symbols` (list)
    *   `tags` -> `tags` (list)
    *   `sentiment` -> `sentiment` (dict)
*   **Source Query Information:** Adds `source_query_tag` or `source_query_symbol` to the normalized article based on how the `fetch_news` method was called.
