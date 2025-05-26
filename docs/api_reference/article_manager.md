[← Back to Main Documentation Index](../index.md)  

[← API reference Index](./index.md)

# Article Manager (`article_manager.py`)

The `article_manager.py` module provides the `ArticleManager` class, which is responsible for managing financial news articles within a SQLite database. This includes storing, retrieving, and updating articles, as well as logging API calls related to article fetching.

## `ArticleManager` Class

The `ArticleManager` class handles all interactions with the SQLite database concerning articles.

### Initialization

```python
class ArticleManager:
    def __init__(self, db_path: str):
        """
        Initialize the article manager.

        Args:
            db_path: Path to SQLite database file.
        """
        # ...
```

The constructor initializes the `ArticleManager` and sets up the connection to the SQLite database. If the database or the required tables do not exist, they are created.

**Parameters:**

*   `db_path` (str): The file path to the SQLite database.

### Database Schema

The `ArticleManager` uses two main tables:

1.  **`articles`**: Stores information about each financial news article.
    *   `url_hash` (TEXT PRIMARY KEY NOT NULL): SHA-256 hash of the article URL.
    *   `title` (TEXT): Title of the article.
    *   `raw_content` (TEXT NOT NULL): The original, unprocessed content of the article.
    *   `processed_content` (TEXT): The cleaned and processed content of the article.
    *   `url` (TEXT NOT NULL): The original URL of the article.
    *   `published_at` (TEXT NOT NULL): Publication date and time of the article.
    *   `added_at` (TEXT NOT NULL): Timestamp when the article was added to the database.
    *   `source_api` (TEXT): The API from which the article was fetched (e.g., "EODHD").
    *   `symbols` (TEXT): JSON string of stock symbols associated with the article.
    *   `tags` (TEXT): JSON string of tags or categories associated with the article.
    *   `sentiment` (TEXT): JSON string representing sentiment analysis results for the article.
    *   `source_query_tag` (TEXT): The tag used in the API query that retrieved this article.
    *   `source_query_symbol` (TEXT): The symbol used in the API query that retrieved this article.
    *   `status_text_processing` (TEXT DEFAULT 'PENDING' NOT NULL): Status of text processing (e.g., 'PENDING', 'SUCCESS', 'FAILED').
    *   `status_embedding` (TEXT DEFAULT 'PENDING' NOT NULL): Status of embedding generation (e.g., 'PENDING', 'SUCCESS', 'FAILED').
    *   `embedding_model` (TEXT): Name of the embedding model used.
    *   `vector_db_id` (TEXT): ID of the article's embedding in the vector database.

2.  **`api_call_log`**: Logs details of API calls made to fetch articles.
    *   `log_id` (INTEGER PRIMARY KEY AUTOINCREMENT): Unique ID for the log entry.
    *   `query_type` (TEXT NOT NULL): Type of query ('tag' or 'symbol').
    *   `query_value` (TEXT NOT NULL): The tag or symbol used in the query.
    *   `last_fetched_timestamp` (TEXT NOT NULL): Timestamp of when the API call was made.
    *   `from_date_param` (TEXT): The 'from_date' parameter used in the API call.
    *   `to_date_param` (TEXT): The 'to_date' parameter used in the API call.
    *   `limit_param` (INTEGER): The 'limit' parameter used in the API call.
    *   `offset_param` (INTEGER): The 'offset' parameter used in the API call.
    *   `articles_retrieved_count` (INTEGER): Number of articles retrieved by this call.
    *   `oldest_article_date_in_batch` (TEXT): Publication date of the oldest article in the retrieved batch.
    *   `newest_article_date_in_batch` (TEXT): Publication date of the newest article in the retrieved batch.
    *   `api_call_successful` (INTEGER NOT NULL): 1 if the API call was successful, 0 otherwise.
    *   `http_status_code` (INTEGER): HTTP status code of the API response.
    *   `error_message` (TEXT): Error message if the API call failed.

### Methods

#### `close_connection()`

```python
def close_connection(self) -> None:
    """Close the database connection if open."""
    # ...
```
Closes the active SQLite database connection. It's good practice to call this when the `ArticleManager` is no longer needed.

#### `get_article_by_hash(url_hash: str) -> dict`

```python
def get_article_by_hash(self, url_hash: str) -> dict:
    """
    Get complete article data by URL hash.

    Args:
        url_hash: SHA-256 hash of the article URL

    Returns:
        dict: Complete article data or None if not found
    """
    # ...
```
Retrieves a single article from the database based on its `url_hash`.

**Parameters:**

*   `url_hash` (str): The SHA-256 hash of the article's URL.

**Returns:**

*   `dict`: A dictionary containing all fields of the article if found, otherwise `None`.

#### `get_processed_articles_for_embedding(status: str = "PENDING", limit: int = 100) -> List[Dict]`

```python
def get_processed_articles_for_embedding(
    self, status: str = "PENDING", limit: int = 100
) -> List[Dict]:
    """
    Get articles that have been processed and are ready for embedding,
    or have a specific embedding status.

    Args:
        status: The embedding status to filter articles by (e.g., 'PENDING', 'FAILED').
        limit: Maximum number of articles to retrieve

    Returns:
        List of article dictionaries with processed content
    """
    # ...
```
Fetches articles that have successfully undergone text processing (`status_text_processing = 'SUCCESS'`) and match the specified embedding `status`. This is typically used to get articles ready to be embedded.

**Parameters:**

*   `status` (str, optional): The `status_embedding` to filter by. Defaults to "PENDING".
*   `limit` (int, optional): The maximum number of articles to return. Defaults to 100.

**Returns:**

*   `List[Dict]`: A list of dictionaries, where each dictionary represents an article and includes fields like `url_hash`, `processed_content`, `title`, etc.

#### `update_article_processing_status(url_hash: str, processed_content: Optional[str] = None, status: str = "SUCCESS", error_message: Optional[str] = None) -> None`

```python
def update_article_processing_status(
    self,
    url_hash: str,
    processed_content: Optional[str] = None,
    status: str = "SUCCESS",
    error_message: Optional[str] = None,
) -> None:
    """
    Update an article's text processing status in the database.

    Args:
        url_hash: SHA-256 hash of the article URL
        processed_content: Cleaned and processed article content
        status: Processing status ('SUCCESS', 'FAILED', etc.)
        error_message: Error message if processing failed
    """
    # ...
```
Updates the text processing status of an article. If processing was successful, the `processed_content` is also stored.

**Parameters:**

*   `url_hash` (str): The SHA-256 hash of the article's URL.
*   `processed_content` (Optional[str], optional): The processed text content of the article. Required if `status` is 'SUCCESS'. Defaults to `None`.
*   `status` (str, optional): The new text processing status (e.g., 'SUCCESS', 'FAILED'). Defaults to "SUCCESS".
*   `error_message` (Optional[str], optional): An error message if processing failed. Defaults to `None`.

#### `update_article_embedding_status(url_hash: str, status: str = "SUCCESS", embedding_model: Optional[str] = None, vector_db_id: Optional[str] = None, error_message: Optional[str] = None) -> None`

```python
def update_article_embedding_status(
    self,
    url_hash: str,
    status: str = "SUCCESS",
    embedding_model: Optional[str] = None,
    vector_db_id: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Update an article's embedding status in the database.

    Args:
        url_hash: SHA-256 hash of the article URL
        status: Embedding status ('SUCCESS', 'FAILED', etc.)
        embedding_model: Name of the embedding model used
        vector_db_id: ID in the vector database
        error_message: Error message if embedding failed
    """
    # ...
```
Updates the embedding status of an article. If embedding was successful, the `embedding_model` and `vector_db_id` are also stored.

**Parameters:**

*   `url_hash` (str): The SHA-256 hash of the article's URL.
*   `status` (str, optional): The new embedding status (e.g., 'SUCCESS', 'FAILED'). Defaults to "SUCCESS".
*   `embedding_model` (Optional[str], optional): The name of the model used for embedding. Defaults to `None`.
*   `vector_db_id` (Optional[str], optional): The ID assigned to the article's embedding in the vector database. Defaults to `None`.
*   `error_message` (Optional[str], optional): An error message if embedding failed. Defaults to `None`.

#### `store_articles(articles: List[Dict], replace_existing: bool = False) -> int`

```python
def store_articles(
    self, articles: List[Dict], replace_existing: bool = False
) -> int:
    """
    Store articles in the database.

    Args:
        articles: List of article dictionaries from EODHDClient
        replace_existing: Whether to replace existing articles

    Returns:
        int: Number of articles stored
    """
    # ...
```
Stores a list of new articles in the database. It generates a URL hash for each article.

**Parameters:**

*   `articles` (List[Dict]): A list of article dictionaries. Each dictionary should contain keys like `url`, `title`, `raw_content`, `published_at`, `source_api`, `symbols`, `tags`, `sentiment`, `source_query_tag`, `source_query_symbol`.
*   `replace_existing` (bool, optional): If `True`, existing articles with the same `url_hash` will be replaced. If `False`, duplicates will be ignored. Defaults to `False`.

**Returns:**

*   `int`: The number of articles successfully stored in the database.

#### `log_api_call(query_type: str, query_value: str, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None, articles_retrieved_count: int = 0, fetched_articles: Optional[List[Dict[str, Any]]] = None, api_call_successful: bool = True, http_status_code: Optional[int] = None, error_message: Optional[str] = None) -> int`

```python
def log_api_call(
    self,
    query_type: str,
    query_value: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    articles_retrieved_count: int = 0,
    fetched_articles: Optional[List[Dict[str, Any]]] = None,
    api_call_successful: bool = True,
    http_status_code: Optional[int] = None,
    error_message: Optional[str] = None,
) -> int:
    """
    Log an API call to the api_call_log table.
    ...
    Returns:
        int: ID of the logged API call (log_id)
    """
    # ...
```
Logs the details of an API call made to fetch articles into the `api_call_log` table.

**Parameters:**

*   `query_type` (str): Type of query (e.g., 'tag', 'symbol').
*   `query_value` (str): The specific tag or symbol queried.
*   `from_date` (Optional[str], optional): The 'from_date' parameter used.
*   `to_date` (Optional[str], optional): The 'to_date' parameter used.
*   `limit` (Optional[int], optional): The 'limit' parameter used.
*   `offset` (Optional[int], optional): The 'offset' parameter used.
*   `articles_retrieved_count` (int, optional): Number of articles retrieved.
*   `fetched_articles` (Optional[List[Dict[str, Any]]], optional): The list of articles fetched. Used to determine date ranges.
*   `api_call_successful` (bool, optional): Whether the API call was successful.
*   `http_status_code` (Optional[int], optional): HTTP status code of the response.
*   `error_message` (Optional[str], optional): Error message if the call failed.

**Returns:**

*   `int`: The `log_id` of the newly created log entry, or -1 if an error occurred.

#### `get_articles_by_processing_status(status: str, limit: int = 100) -> List[Dict]`

```python
def get_articles_by_processing_status(
    self, status: str, limit: int = 100
) -> List[Dict]:
    """
    Get articles by their text processing status.

    Args:
        status: The status to filter articles by (e.g., 'FAILED', 'SUCCESS', 'PENDING')
        limit: Maximum number of articles to retrieve

    Returns:
        List of article dictionaries matching the specified status
    """
    # ...
```
Retrieves articles based on their `status_text_processing`.

**Parameters:**

*   `status` (str): The text processing status to filter by (e.g., 'PENDING', 'SUCCESS', 'FAILED').
*   `limit` (int, optional): The maximum number of articles to return. Defaults to 100.

**Returns:**

*   `List[Dict]`: A list of article dictionaries matching the criteria.

#### `delete_article_by_hash(url_hash: str) -> bool`

```python
def delete_article_by_hash(self, url_hash: str) -> bool:
    """
    Delete an article from the database by its URL hash.

    Args:
        url_hash: SHA-256 hash of the article URL

    Returns:
        bool: True if an article was successfully deleted, False otherwise
    """
    # ...
```
Deletes an article from the `articles` table using its `url_hash`.

**Parameters:**

*   `url_hash` (str): The SHA-256 hash of the article's URL.

**Returns:**

*   `bool`: `True` if the article was successfully deleted, `False` otherwise (e.g., article not found or database error).

#### `get_database_statistics() -> Dict[str, Any]`

```python
def get_database_statistics(self) -> Dict[str, Any]:
    """
    Get statistics about the article database.

    Returns:
        Dict containing article database statistics including total counts,
        processing statuses, embedding statuses, article tags, symbols,
        date ranges, and API call data.
    """
    # ...
```
Retrieves various statistics about the articles and API calls stored in the database.

**Returns:**

*   `Dict[str, Any]`: A dictionary containing statistics such as:
    *   `total_articles`: Total number of articles.
    *   `text_processing_status`: Counts of articles by text processing status.
    *   `embedding_status`: Counts of articles by embedding status.
    *   `articles_by_tag`: Counts of articles grouped by `source_query_tag`.
    *   `articles_by_symbol`: Counts of articles grouped by `source_query_symbol`.
    *   `date_range`: Oldest and newest article publication dates.
    *   `api_calls`: Statistics about logged API calls (total calls, total articles retrieved).
    *   May also include an `error` key if fetching statistics failed.
