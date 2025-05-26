[← Back to Main Documentation Index](../index.md)  

[← API reference Index](./index.md)

# FinancialNewsRAG Orchestrator API Reference

## Overview

The `FinancialNewsRAG` class is the central orchestrator for the financial news retrieval and generation system. It integrates various components to provide a high-level interface for fetching, processing, storing, embedding, and searching financial news articles.

This class coordinates the functionalities of:

-   [`EODHDClient`](./eodhd.md): For fetching financial news articles from the EODHD API.
-   [`ArticleManager`](./article_manager.md): For storing and managing article metadata and content in an SQLite database.
-   [`TextProcessor`](./text_processor.md): For cleaning, validating, and chunking article text.
-   [`EmbeddingsGenerator`](./embeddings.md): For creating vector embeddings from text chunks using Gemini models.
-   [`ChromaDBManager`](./chroma_manager.md): For storing and querying vector embeddings in ChromaDB.
-   [`ReRanker`](./reranker.md): For re-ranking search results using a Gemini language model to improve relevance.

## `FinancialNewsRAG` Class

### `__init__(self, eodhd_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None, db_path: Optional[str] = None, chroma_persist_dir: Optional[str] = None, chroma_collection_name: Optional[str] = None, max_tokens_per_chunk: Optional[int] = None)`

Initializes the `FinancialNewsRAG` orchestrator.

**Arguments:**

-   `eodhd_api_key` (Optional[str]): Your EODHD API key. If not provided, it will be loaded from the configuration (environment variable `EODHD_API_KEY`).
-   `gemini_api_key` (Optional[str]): Your Gemini API key. If not provided, it will be loaded from the configuration (environment variable `GEMINI_API_KEY`).
-   `db_path` (Optional[str]): Path to the SQLite database file for storing article metadata. If `None`, uses the path from [`Config`](./config.md) (`DATABASE_PATH`).
-   `chroma_persist_dir` (Optional[str]): Directory to persist ChromaDB data. If `None`, uses the path from [`Config`](./config.md) (`CHROMA_DEFAULT_PERSIST_DIRECTORY`).
-   `chroma_collection_name` (Optional[str]): Name of the ChromaDB collection. If `None`, uses the name from [`Config`](./config.md) (`CHROMA_DEFAULT_COLLECTION_NAME`).
-   `max_tokens_per_chunk` (Optional[int]): Maximum number of tokens per text chunk for processing. If `None`, uses the value from [`Config`](./config.md) (`TEXTPROCESSOR_MAX_TOKENS_PER_CHUNK`).

**Raises:**

-   `ValueError`: If `eodhd_api_key` or `gemini_api_key` is not provided either as an argument or in the configuration.

---

### `fetch_and_store_articles(self, tag: Optional[str] = None, symbol: Optional[str] = None, from_date: Optional[str] = None, to_date: Optional[str] = None, limit: int = 50) -> Dict[str, Any]`

Fetches financial news articles from the EODHD API based on a tag or symbol and stores them in the SQLite database via the [`ArticleManager`](./article_manager.md).

**Arguments:**

-   `tag` (Optional[str]): News tag to filter articles (e.g., "TECHNOLOGY"). Mutually exclusive with `symbol`.
-   `symbol` (Optional[str]): Stock symbol to fetch news for (e.g., "AAPL"). Only a single symbol string is accepted. Mutually exclusive with `tag`.
-   `from_date` (Optional[str]): Start date for fetching articles in "YYYY-MM-DD" format.
-   `to_date` (Optional[str]): End date for fetching articles in "YYYY-MM-DD" format.
-   `limit` (int): Maximum number of articles to fetch. Defaults to 50.

**Returns:**

-   `Dict[str, Any]`: A dictionary containing an operation summary:
    -   `articles_fetched` (int): Number of articles fetched from the API.
    -   `articles_stored` (int): Number of new articles successfully stored in the database.
    -   `status` (str): "SUCCESS" or "FAILED".
    -   `errors` (List[str]): A list of error messages if any occurred.

**Raises:**

-   `ValueError`: If neither `tag` nor `symbol` is provided, or if both are provided.

---

### `process_articles_by_status(self, status: str = "PENDING", limit: int = 100) -> Dict[str, Any]`

Processes articles retrieved from the SQLite database that have a specific processing status (e.g., "PENDING", "FAILED"). Processing involves cleaning and validating the raw article content using the [`TextProcessor`](./text_processor.md).

**Arguments:**

-   `status` (str): The processing status to filter articles by (e.g., "PENDING", "FAILED"). Defaults to "PENDING".
-   `limit` (int): Maximum number of articles to process. Defaults to 100.

**Returns:**

-   `Dict[str, Any]`: A dictionary containing an operation summary:
    -   `articles_processed` (int): Number of articles successfully processed.
    -   `articles_failed` (int): Number of articles that failed processing.
    -   `status` (str): Overall operation status ("SUCCESS" or "FAILED").
    -   `errors` (List[str]): A list of error messages if any occurred.

---

### `embed_processed_articles(self, status: str = "PENDING", limit: int = 100) -> Dict[str, Any]`

Generates embeddings for processed articles and stores them in ChromaDB. It fetches articles based on their embedding status ("PENDING" or "FAILED") from the SQLite database.

For each article:
1.  Retrieves the processed content.
2.  Splits the content into chunks using [`TextProcessor`](./text_processor.md).
3.  Generates embeddings for these chunks using [`EmbeddingsGenerator`](./embeddings.md).
4.  Stores the chunks and their embeddings in ChromaDB via [`ChromaDBManager`](./chroma_manager.md).
5.  Updates the article's embedding status in the SQLite database.

**Arguments:**

-   `status` (str): The embedding status of articles to process (e.g., "PENDING" for initial embedding, "FAILED" for re-attempting). Defaults to "PENDING".
-   `limit` (int): Maximum number of articles to embed. Defaults to 100.

**Returns:**

-   `Dict[str, Any]`: A dictionary containing an operation summary:
    -   `articles_embedding_succeeded` (int): Number of articles successfully embedded.
    -   `articles_failed` (int): Number of articles that failed the embedding process.
    -   `status` (str): Overall operation status ("SUCCESS" or "FAILED").
    -   `errors` (List[str]): A list of error messages if any occurred.

---

### `search_articles(self, query: str, n_results: int = 5, sort_by_metadata: Optional[Dict[str, str]] = None, rerank: bool = False, from_date_str: Optional[str] = None, to_date_str: Optional[str] = None) -> List[Dict[str, Any]]`

Searches for articles relevant to the given `query`.

The process involves:
1.  Generating an embedding for the `query` using [`EmbeddingsGenerator`](./embeddings.md).
2.  Querying [`ChromaDBManager`](./chroma_manager.md) to find the `n_results` (or more if reranking) most similar article chunks, potentially filtered by `from_date_str`, `to_date_str` and `sort_by_metadata`.
3.  If `rerank` is `True`, the initial search results (chunks) are re-ranked using the [`ReRanker`](./reranker.md) to improve relevance.
4.  Retrieving full article details from [`ArticleManager`](./article_manager.md) for the top results.

**Arguments:**

-   `query` (str): The search query.
-   `n_results` (int): Maximum number of results to return. Defaults to 5.
-   `sort_by_metadata` (Optional[Dict[str, str]]): Dictionary for filtering/sorting based on metadata (e.g., `{"published_at_timestamp": "desc"}`). Currently, this is noted as needing proper implementation for range filters.
-   `rerank` (bool): Whether to apply re-ranking with a Gemini LLM. Defaults to `False`.
-   `from_date_str` (Optional[str]): ISO format string (YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD) for filtering articles published on or after this date.
-   `to_date_str` (Optional[str]): ISO format string (YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD) for filtering articles published on or before this date.

**Returns:**

-   `List[Dict[str, Any]]`: A list of article dictionaries, each including details like title, URL, content snippet, and similarity/rerank scores.

---

### `get_article_database_status(self) -> Dict[str, Any]`

Retrieves statistics about the article database (SQLite) managed by [`ArticleManager`](./article_manager.md).

**Returns:**

-   `Dict[str, Any]`: A dictionary containing article database statistics (e.g., total articles). Returns `{"error": str(e), "status": "FAILED"}` on failure.

---

### `get_vector_database_status(self) -> Dict[str, Any]`

Retrieves statistics about the vector database (ChromaDB) managed by [`ChromaDBManager`](./chroma_manager.md).

**Returns:**

-   `Dict[str, Any]`: A dictionary containing vector database statistics (e.g., total chunks). Returns `{"error": str(e), "status": "FAILED"}` on failure.

---

### `delete_articles_older_than(self, days: int = 180) -> Dict[str, Any]`

Deletes articles older than the specified number of days from both the SQLite database and ChromaDB.

**Arguments:**

-   `days` (int): Number of days to use as the age threshold. Articles older than this will be deleted. Defaults to 180.

**Returns:**

-   `Dict[str, Any]`: A dictionary containing an operation summary:
    -   `targeted_articles` (int): Number of articles targeted for deletion.
    -   `deleted_from_sqlite` (int): Number of articles successfully deleted from SQLite.
    -   `deleted_from_chroma` (int): Number of article embeddings successfully deleted from ChromaDB.
    -   `status` (str): "SUCCESS", "PARTIAL_FAILURE", or "FAILED".
    -   `errors` (List[str]): A list of error messages if any occurred.

---

### `close(self) -> None`

Closes database connections and cleans up resources. This includes the SQLite connection in [`ArticleManager`](./article_manager.md) and the ChromaDB connection in [`ChromaDBManager`](./chroma_manager.md).

It's good practice to call this when the orchestrator is no longer needed.

**Returns:**

-   `None`

---

*This documentation was auto-generated based on the docstrings in `financial_news_rag.orchestrator`.*
