# Text Processing and Article Management

This document outlines the components responsible for processing raw text data from news articles and managing their storage and state within the financial news RAG system. These responsibilities were formerly handled by a single `TextProcessingPipeline` class but have been refactored into two distinct classes: `TextProcessor` and `ArticleManager` for better separation of concerns.

## 1. `ArticleManager`

The `ArticleManager` class is responsible for all interactions with the SQLite database (`financial_news.db`). Its key duties include:

- **Storing Articles**: Saving new articles fetched from data sources (e.g., EODHD).
- **Managing Processing Status**: Tracking whether an article has been cleaned, chunked, or embedded.
- **Managing Embedding Status**: Specifically noting if embeddings have been generated and stored for an article's chunks.
- **Logging API Calls**: Recording details of API calls made (e.g., to EODHD or embedding services).
- **Retrieving Articles**: Fetching articles or their chunks based on various criteria (e.g., unprocessed articles).

**Key Methods (Conceptual):**

- `store_articles(articles_data)`: Stores a list of articles.
- `get_articles_to_process()`: Retrieves articles that haven't been processed.
- `update_article_status(article_id, status_field, new_value)`: Updates status fields.
- `get_chunks_for_embedding()`: Retrieves text chunks ready for embedding.
- `mark_chunks_as_embedded(chunk_ids)`: Updates the status of chunks once embeddings are stored.
- `log_api_call(service_name, endpoint, params, response_data)`: Logs API interactions.

## 2. `TextProcessor`

The `TextProcessor` class focuses solely on the textual manipulation of article content. Its responsibilities are:

- **Text Cleaning**: Removing unwanted HTML, normalizing whitespace, and other cleaning operations to prepare text for analysis.
- **Text Chunking**: Dividing cleaned article text into smaller, manageable chunks suitable for embedding. This often involves strategies to respect sentence boundaries and manage chunk size.

**Key Methods (Conceptual):**

- `clean_text(html_content)`: Takes raw HTML or text and returns cleaned text.
- `chunk_text(text, strategy='semantic')`: Splits the cleaned text into chunks based on a specified strategy.

## Workflow Example

The typical workflow involving these classes, as demonstrated in `examples/end_to_end_pipeline_example.py`, is:

1. Fetch news data (e.g., using `EODHDClient`).
2. Store the raw articles using `ArticleManager`.
3. Retrieve articles needing processing from `ArticleManager`.
4. For each article:
   a. Clean its content using `TextProcessor`.
   b. Chunk the cleaned text using `TextProcessor`.
   c. Store these chunks and update processing statuses using `ArticleManager`.
5. Retrieve chunks needing embeddings from `ArticleManager`.
6. Generate embeddings for these chunks (e.g., using `EmbeddingsGenerator`).
7. Store embeddings (e.g., in ChromaDB via `ChromaDBManager`).
8. Update embedding statuses in `ArticleManager`.

This separation allows for more modularity and easier maintenance. For instance, the text cleaning or chunking logic in `TextProcessor` can be modified without impacting the database interaction logic in `ArticleManager`.

## Future Enhancements

- More sophisticated chunking strategies in `TextProcessor`.
- Enhanced status tracking and error handling in `ArticleManager`.

The primary example script `examples/end_to_end_pipeline_example.py` provides a comprehensive demonstration of how `ArticleManager` and `TextProcessor` are used in conjunction with other components of the system.
