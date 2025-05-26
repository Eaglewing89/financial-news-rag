# Usage Guide: FinancialNewsRAG Orchestrator

This guide provides a comprehensive overview of how to use the `FinancialNewsRAG` orchestrator, the central component for interacting with the financial news RAG system. It covers initialization, fetching and storing articles, processing content, generating embeddings, searching, and managing data.

An alternative to this guide is the [jupyter notebook example](../examples/financial_news_rag_example.ipynb) for a more interactive tutorial. 

## 1. Initialization

First, you need to initialize the `FinancialNewsRAG` orchestrator. This requires API keys for EODHD and Gemini. These keys are typically loaded from environment variables (via a `.env` file in your project root) by the centralized `Config` class.

You can also specify custom paths for the SQLite database (where article metadata and processed text are stored) and the ChromaDB persistence directory (where text embeddings are stored).

**Key Initialization Points:**

*   **API Keys:** Ensure `EODHD_API_KEY` and `GEMINI_API_KEY` are set in your environment or `.env` file.
*   **Database Paths:** You can override the default database and ChromaDB paths if needed.
*   **Centralized Configuration:** The system uses a `Config` class (`config.py`) to manage all settings. Parameters provided to the `FinancialNewsRAG` constructor will override these centralized configurations.

```python
import os
from financial_news_rag import FinancialNewsRAG

# Using unique DB paths for this example to avoid conflicts
example_db_path = "financial_news_rag_usage_guide.db"
example_chroma_persist_dir = "chroma_db_usage_guide"

try:
    # Initialize the orchestrator
    # API keys will be loaded from environment variables by default.
    # We are overriding DB paths here for demonstration.
    orchestrator = FinancialNewsRAG(
        db_path=example_db_path,
        chroma_persist_dir=example_chroma_persist_dir
    )
    print("FinancialNewsRAG orchestrator initialized successfully.")
    print(f"SQLite DB will be created/used at: {os.path.abspath(example_db_path)}")
    print(f"ChromaDB will persist data in: {os.path.abspath(example_chroma_persist_dir)}")

except ValueError as e:
    print(f"Initialization failed: {e}")
    print("Please ensure EODHD_API_KEY and GEMINI_API_KEY are set in your .env file or environment variables.")

```

### Logging

It's helpful to configure logging to see the orchestrator's activity. You can adjust the logging level as needed.

```python
import logging

# Basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Optionally, set a higher level for the financial_news_rag logger to reduce its verbosity
logging.getLogger('financial_news_rag').setLevel(logging.WARNING)

print("Logging configured.")
```

## 2. Fetching and Storing Articles

The `fetch_and_store_articles` method fetches news articles from the EODHD API and stores their raw content and metadata in the SQLite database.

You can fetch articles by:
*   `tag`: (e.g., 'TECHNOLOGY', 'M&A')
*   `symbol`: (e.g., 'AAPL.US', 'MSFT.US')
*   `from_date` and `to_date`: To specify a date range.
*   `limit`: To control the maximum number of articles fetched per call.

```python
if 'orchestrator' in locals() and orchestrator:
    # Example 1: Fetch articles by tag (e.g., 'MERGERS AND ACQUISITIONS')
    print("Fetching articles by tag 'MERGERS AND ACQUISITIONS'...")
    fetch_results_tag = orchestrator.fetch_and_store_articles(tag="MERGERS AND ACQUISITIONS", limit=5) # Using a small limit
    print(f"Tag fetch results: {fetch_results_tag}")

    # Example 2: Fetch articles by symbol (e.g., 'MSFT.US')
    print("\nFetching articles by symbol 'MSFT.US'...")
    fetch_results_symbol = orchestrator.fetch_and_store_articles(symbol="MSFT.US", limit=5)
    print(f"Symbol fetch results: {fetch_results_symbol}")

    # Example 3: Fetch articles with a date range
    from datetime import datetime, timedelta
    to_date_str = datetime.now().strftime('%Y-%m-%d')
    from_date_str = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    print(f"\nFetching articles for 'AAPL.US' from {from_date_str} to {to_date_str}...")
    fetch_results_date = orchestrator.fetch_and_store_articles(
        symbol="AAPL.US",
        from_date=from_date_str,
        to_date=to_date_str,
        limit=5
    )
    print(f"Date range fetch results: {fetch_results_date}")
else:
    print("Orchestrator not initialized. Skipping fetch examples.")
```

## 3. Processing Articles

The `process_articles_by_status` method retrieves articles from the database based on their processing status (e.g., 'PENDING' for newly fetched articles, 'FAILED' for articles that failed previous processing attempts).

Processing involves:
*   Cleaning HTML from the raw article content.
*   Extracting the main textual content.
*   Validating the content.

Successfully processed articles are updated in the database with the cleaned content and their status is changed to 'SUCCESS'.

```python
if 'orchestrator' in locals() and orchestrator:
    print("\nProcessing 'PENDING' articles...")
    # Process articles that are currently 'PENDING' (newly fetched)
    # The limit here applies to how many articles are processed in this batch.
    processing_results_pending = orchestrator.process_articles_by_status(status='PENDING', limit=10)
    print(f"Pending processing results: {processing_results_pending}")

    # You could also re-process 'FAILED' articles:
    # print("\nAttempting to re-process 'FAILED' articles...")
    # processing_results_failed = orchestrator.process_articles_by_status(status='FAILED', limit=5)
    # print(f"Failed processing results: {processing_results_failed}")
else:
    print("Orchestrator not initialized. Skipping processing examples.")
```

## 4. Embedding Articles

The `embed_processed_articles` method takes articles that have been successfully processed (content status 'SUCCESS') and generates embeddings for their textual content using the configured embedding model (e.g., Gemini).

These embeddings are then stored in ChromaDB, making the articles searchable. This method can process articles whose embedding status is 'PENDING' or re-attempt those that are 'FAILED'.

```python
if 'orchestrator' in locals() and orchestrator:
    print("\nEmbedding articles with 'PENDING' embedding status...")
    # Embed articles whose content has been processed and are 'PENDING' embedding
    embedding_results_pending = orchestrator.embed_processed_articles(status='PENDING', limit=10)
    print(f"Pending embedding results: {embedding_results_pending}")

    # Optionally, re-attempt embedding for articles that 'FAILED' previously
    # print("\nRe-attempting to embed articles with 'FAILED' embedding status...")
    # embedding_results_failed = orchestrator.embed_processed_articles(status='FAILED', limit=5)
    # print(f"Failed embedding results: {embedding_results_failed}")
else:
    print("Orchestrator not initialized. Skipping embedding examples.")
```

## 5. Checking Database Status

You can monitor the state of your data using status-checking methods:

*   `get_article_database_status()`: Provides a summary of articles in the SQLite database, including counts by processing and embedding status.
*   `get_vector_database_status()`: Provides a summary of the ChromaDB vector store, including the total number of embedded chunks.

```python
if 'orchestrator' in locals() and orchestrator:
    print("\nFetching article database status...")
    article_db_status = orchestrator.get_article_database_status()
    print(f"Article DB Status: {article_db_status}")

    print("\nFetching vector database status...")
    vector_db_status = orchestrator.get_vector_database_status()
    print(f"Vector DB Status: {vector_db_status}")
else:
    print("Orchestrator not initialized. Skipping database status examples.")
```

## 6. Searching Articles

The `search_articles` method allows you to find articles relevant to a given query. The process involves:
1.  Generating an embedding for your search query.
2.  Searching ChromaDB for the most similar article chunks based on their embeddings.
3.  Retrieving the full article details from SQLite for these chunks.
4.  Optionally, re-ranking the search results using a Gemini LLM for enhanced relevance (this is slower but can be more accurate).

You can also filter search results by date.

```python
if 'orchestrator' in locals() and orchestrator and vector_db_status.get('total_chunks', 0) > 0:
    # Example 1: Basic search
    query1 = "latest advancements in AI by Microsoft"
    print(f"\nSearching for: '{query1}'...")
    search_results1 = orchestrator.search_articles(query=query1, n_results=3)
    print("Search Results 1 (basic):")
    for i, article in enumerate(search_results1):
        print(f"  {i+1}. Title: {article.get('title')}, Score: {article.get('similarity_score')}")
        # print(f"     URL: {article.get('link')}") # 'link' or 'url' depending on data source
        # print(f"     Published: {article.get('published_at')}")
        # print(f"     Content Snippet: {article.get('processed_content', '')[:150]}...")

    # Example 2: Search with re-ranking
    query2 = "Acquisitions in the tech industry related to safety and security"
    print(f"\nSearching for: '{query2}' with re-ranking...")
    search_results2 = orchestrator.search_articles(query=query2, n_results=3, rerank=True)
    print("Search Results 2 (re-ranked):")
    for i, article in enumerate(search_results2):
        print(f"  {i+1}. Title: {article.get('title')}, Score: {article.get('relevance_score', article.get('similarity_score'))}")
        print(f"     URL: {article.get('url')}") # Reranker might use 'url'
        print(f"     Published: {article.get('published_at')}")
        # print(f"     Content Snippet: {article.get('processed_content', '')[:150]}...")

    # Example 3: Search with date filtering
    from datetime import datetime, timedelta
    # Ensure dates are in ISO format with 'Z' for UTC, as expected by ChromaDB metadata filters
    from_date_search = (datetime.now() - timedelta(days=3)).isoformat() + "Z"
    to_date_search = datetime.now().isoformat() + "Z"
    query3 = "market trends"
    print(f"\nSearching for: '{query3}' between {from_date_search} and {to_date_search}...")
    search_results3 = orchestrator.search_articles(
        query=query3,
        n_results=3,
        from_date_str=from_date_search,
        to_date_str=to_date_search
    )
    print("Search Results 3 (date filtered):")
    for i, article in enumerate(search_results3):
        print(f"  {i+1}. Title: {article.get('title')}, Published: {article.get('published_at')}, Score: {article.get('similarity_score')}")

elif 'orchestrator' in locals() and orchestrator:
    print("\nVector database is empty. Skipping search examples. Run fetch, process, and embed steps first.")
else:
    print("Orchestrator not initialized. Skipping search examples.")
```

## 7. Deleting Old Articles

The `delete_articles_older_than` method removes articles from both the SQLite database and ChromaDB that were published earlier than a specified number of days ago. This is useful for data retention and managing storage.

```python
if 'orchestrator' in locals() and orchestrator:
    # Example: Delete articles older than 1 day (for demonstration)
    # Be cautious with this in a real scenario.
    # If articles fetched were published more than 1 day ago, they will be deleted.
    print("\nAttempting to delete articles older than 1 day (for demonstration)...")
    delete_results = orchestrator.delete_articles_older_than(days=1)
    print(f"Deletion results: {delete_results}")

    # Check status again after deletion
    print("\nFetching article database status after potential deletion...")
    article_db_status_after_delete = orchestrator.get_article_database_status()
    print(f"Article DB Status: {article_db_status_after_delete}")

    print("\nFetching vector database status after potential deletion...")
    vector_db_status_after_delete = orchestrator.get_vector_database_status()
    print(f"Vector DB Status: {vector_db_status_after_delete}")
else:
    print("Orchestrator not initialized. Skipping deletion examples.")
```

## 8. Closing Connections

When you are finished using the orchestrator, call the `close()` method to properly close database connections (SQLite and ChromaDB) and release any other resources.

```python
if 'orchestrator' in locals() and orchestrator:
    print("\nClosing orchestrator connections...")
    orchestrator.close()
    print("Orchestrator connections closed.")

    # Clean up the example database and chroma directory created by this guide
    # Comment out if you want to inspect the files.
    if os.path.exists(example_db_path):
        os.remove(example_db_path)
        print(f"Removed example SQLite DB: {example_db_path}")
    if os.path.exists(example_chroma_persist_dir):
        import shutil
        shutil.rmtree(example_chroma_persist_dir)
        print(f"Removed example ChromaDB directory: {example_chroma_persist_dir}")
else:
    print("Orchestrator not initialized. Skipping close example.")
```

## 9. Customizing Configuration

The `FinancialNewsRAG` system uses a centralized `Config` class for managing settings. You can customize these settings in several ways:

1.  **Environment Variables:** Set variables in your `.env` file (e.g., `EODHD_API_KEY`, `GEMINI_API_KEY`, `TEXTPROCESSOR_MAX_TOKENS_PER_CHUNK`). These are loaded automatically.
2.  **Direct Overrides in Constructor:** Pass parameters directly to the `FinancialNewsRAG` constructor. These take precedence over environment variables and defaults.
    ```python
    # Example of overriding configuration values at initialization
    # custom_orchestrator = FinancialNewsRAG(
    #     eodhd_api_key="your_custom_eodhd_key",
    #     gemini_api_key="your_custom_gemini_key",
    #     db_path="custom_financial_news.db",
    #     chroma_persist_dir="custom_chroma_data",
    #     max_tokens_per_chunk=1024  # Overrides TEXTPROCESSOR_MAX_TOKENS_PER_CHUNK
    # )
    ```
3.  **Custom Config Class (Advanced):** For more complex scenarios, you could potentially subclass the `Config` object, though direct overrides or environment variables are usually sufficient.

Refer to `configuration.md` for a detailed list of all configurable parameters, their corresponding environment variables, and default values.

This concludes the usage guide for the `FinancialNewsRAG` orchestrator. You can adapt these examples to build more complex workflows for your financial news analysis and retrieval tasks.
