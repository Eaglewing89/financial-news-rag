[← Back to Main Documentation Index](../index.md)  

[← API reference Index](./index.md)

# ChromaDB Manager (`chroma_manager.py`)

The `chroma_manager.py` module provides the `ChromaDBManager` class, designed to handle all interactions with a ChromaDB vector database. This includes initializing the database, adding and querying article embeddings, managing metadata, and deleting entries.

## `ChromaDBManager` Class

This class encapsulates the logic for managing embeddings in ChromaDB, which is used for similarity searches on financial news articles.

### Initialization

```python
class ChromaDBManager:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_dimension: int,
        in_memory: bool = False,
    ):
        """
        Initialize the ChromaDBManager with connection parameters.

        Args:
            persist_directory: Path for persistent storage of embeddings.
            collection_name: Name of the ChromaDB collection to use.
            embedding_dimension: Dimension of the embedding vectors.
            in_memory: Whether to use an in-memory ChromaDB instance (for testing).
        """
        # ...
```

The constructor sets up the `ChromaDBManager`.

**Parameters:**

*   `persist_directory` (str): The file system path where ChromaDB should persist its data. If `in_memory` is `True`, this is ignored.
*   `collection_name` (str): The name of the collection within ChromaDB to use for storing embeddings.
*   `embedding_dimension` (int): The dimensionality of the embedding vectors that will be stored (e.g., 768 for Gemini embeddings).
*   `in_memory` (bool, optional): If `True`, ChromaDB will run in-memory (useful for testing or temporary instances). Defaults to `False`.

Upon initialization, it either connects to an existing ChromaDB collection or creates a new one if it doesn't exist.

### Methods

#### `close_connection()`

```python
def close_connection(self) -> None:
    """
    Close the ChromaDB client connection and release resources.
    """
    # ...
```
Closes the connection to the ChromaDB client. This should be called to ensure resources are properly released, especially when using a persistent client.

#### `add_article_chunks(article_url_hash: str, chunk_texts: List[str], chunk_vectors: List[List[float]], article_data: Dict[str, Any]) -> bool`

```python
def add_article_chunks(
    self,
    article_url_hash: str,
    chunk_texts: List[str],
    chunk_vectors: List[List[float]],
    article_data: Dict[str, Any],
) -> bool:
    """
    Add chunk data for an article to ChromaDB.
    ...
    Returns:
        bool: True if successful, False otherwise
    """
    # ...
```
Adds or updates (upserts) chunks of an article into the ChromaDB collection. Each chunk consists of its text, its embedding vector, and associated metadata.

**Parameters:**

*   `article_url_hash` (str): A unique identifier for the article (typically the URL hash from the SQLite database).
*   `chunk_texts` (List[str]): A list of text strings, where each string is a chunk of the article.
*   `chunk_vectors` (List[List[float]]): A list of embedding vectors, where each vector corresponds to a chunk in `chunk_texts`.
*   `article_data` (Dict[str, Any]): A dictionary containing metadata for the article. This can include:
    *   `published_at` (str, optional): The publication date of the article in ISO format. This is converted to a Unix timestamp and stored as `published_at_timestamp` in the metadata for efficient range queries.
    *   `source_query_tag` (str, optional): The tag used when querying the news API for this article.
    *   `source_query_symbol` (str, optional): The stock symbol used when querying the news API for this article.

**Returns:**

*   `bool`: `True` if the chunks were successfully added/updated, `False` otherwise.

**Metadata for each chunk includes:**

*   `article_url_hash`: The hash of the parent article.
*   `chunk_index`: The 0-based index of the chunk within the article.
*   `published_at_timestamp` (int, optional): The publication date as a Unix timestamp (seconds since epoch).
*   `source_query_tag` (str, optional).
*   `source_query_symbol` (str, optional).

#### `query_embeddings(query_embedding: List[float], n_results: int = 5, filter_metadata: Optional[Dict[str, Any]] = None, from_date_str: Optional[str] = None, to_date_str: Optional[str] = None, return_similarity_score: bool = False) -> List[Dict[str, Any]]`

```python
def query_embeddings(
    self,
    query_embedding: List[float],
    n_results: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
    from_date_str: Optional[str] = None,
    to_date_str: Optional[str] = None,
    return_similarity_score: bool = False,
) -> List[Dict[str, Any]]:
    """
    Query ChromaDB for the most similar embeddings to the query embedding.
    ...
    Returns:
        List of results, each including chunk_id, distance/similarity_score, metadata, and text
    """
    # ...
```
Searches the ChromaDB collection for chunks whose embeddings are most similar to the provided `query_embedding`.

**Parameters:**

*   `query_embedding` (List[float]): The embedding vector of the search query.
*   `n_results` (int, optional): The maximum number of similar chunks to retrieve. Defaults to 5.
*   `filter_metadata` (Optional[Dict[str, Any]], optional): A dictionary to filter results based on metadata fields (e.g., `{"article_url_hash": "some_hash"}`).
*   `from_date_str` (Optional[str], optional): An ISO format date string (e.g., "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SSZ"). Filters results to include chunks from articles published on or after this date. Uses the `published_at_timestamp` metadata field.
*   `to_date_str` (Optional[str], optional): An ISO format date string. Filters results to include chunks from articles published on or before this date. Uses the `published_at_timestamp` metadata field.
*   `return_similarity_score` (bool, optional): If `True`, the results will include a `similarity_score` (0 to 1, where 1 is most similar) instead of `distance`. The similarity is calculated as `1.0 - (distance / 2.0)` assuming cosine distance. Defaults to `False`.

**Returns:**

*   `List[Dict[str, Any]]`: A list of result dictionaries. Each dictionary contains:
    *   `chunk_id` (str): The unique ID of the chunk.
    *   `metadata` (dict): The metadata associated with the chunk.
    *   `text` (str): The text content of the chunk.
    *   `distance` (float, if `return_similarity_score` is `False`): The distance metric (e.g., cosine distance) between the query embedding and the chunk's embedding.
    *   `similarity_score` (float, if `return_similarity_score` is `True`): The calculated similarity score.

#### `get_collection_status() -> Dict[str, Any]`

```python
def get_collection_status(self) -> Dict[str, Any]:
    """
    Get status information about the ChromaDB collection.

    Returns:
        Dictionary with collection stats
    """
    # ...
```
Retrieves statistics and information about the current ChromaDB collection.

**Returns:**

*   `Dict[str, Any]`: A dictionary containing:
    *   `collection_name` (str): Name of the collection.
    *   `total_chunks` (int): Total number of individual chunks stored in the collection.
    *   `unique_articles` (int): Number of unique articles (based on `article_url_hash`) represented in the collection.
    *   `embedding_dimension` (int): The dimension of the embeddings.
    *   `is_empty` (bool): `True` if the collection contains no items.
    *   `persist_directory` (str or None): The path to the persistence directory, or `None` if in-memory.
    *   May also include an `error` key if fetching status failed.

#### `delete_embeddings_by_article(article_url_hash: str) -> bool`

```python
def delete_embeddings_by_article(self, article_url_hash: str) -> bool:
    """
    Delete all embeddings associated with a given article_url_hash.

    Args:
        article_url_hash: The unique identifier of the article

    Returns:
        bool: True if successful, False otherwise
    """
    # ...
```
Deletes all chunks (and their embeddings) associated with a specific `article_url_hash` from the collection.

**Parameters:**

*   `article_url_hash` (str): The unique identifier of the article whose chunks should be deleted.

**Returns:**

*   `bool`: `True` if deletion was successful or if no matching chunks were found, `False` if an error occurred.

#### `get_article_hashes_by_date_range(older_than_timestamp: Optional[int] = None, newer_than_timestamp: Optional[int] = None) -> List[str]`

```python
def get_article_hashes_by_date_range(
    self,
    older_than_timestamp: Optional[int] = None,
    newer_than_timestamp: Optional[int] = None,
) -> List[str]:
    """
    Retrieve article hashes based on published_at_timestamp within a specified date range.

    Args:
        older_than_timestamp: Optional upper bound timestamp (inclusive).
        newer_than_timestamp: Optional lower bound timestamp (inclusive).

    Returns:
        List[str]: A list of unique article_url_hash values that match the criteria.
    """
    # ...
```
Retrieves a list of unique `article_url_hash` values for articles whose `published_at_timestamp` falls within the specified range.

**Parameters:**

*   `older_than_timestamp` (Optional[int], optional): A Unix timestamp. If provided, only articles published on or before this timestamp are considered.
*   `newer_than_timestamp` (Optional[int], optional): A Unix timestamp. If provided, only articles published on or after this timestamp are considered.

**Returns:**

*   `List[str]`: A list of unique `article_url_hash` strings. Returns an empty list if no criteria are specified or if an error occurs.
