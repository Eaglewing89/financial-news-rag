[← Back to Main Documentation Index](../index.md)  

[← API reference Index](./index.md)

# `config` Module API Reference

The `config` module is responsible for managing all application configurations. It provides a centralized `Config` class that loads settings from environment variables (supporting `.env` files) and offers default values for various parameters essential for the application's operation.

## `Config` Class

The `Config` class is the core of the configuration management system. An instance of this class is automatically created and made available as `config` when the module is imported.

```python
from financial_news_rag.config import config
```

### Initialization

When a `Config` object is initialized (which happens automatically for the global `config` instance), it performs the following actions:
1.  Loads environment variables from a `.env` file in the project root, if one exists.
2.  Reads specific environment variables to set up configurations for various components of the application.
3.  Provides default values if certain environment variables are not set (except for required ones like API keys).

### Configuration Properties

The `Config` class exposes its settings through read-only properties. Below is a list of available properties, their corresponding environment variables, and default values.

#### EODHD API Configuration

-   **`eodhd_api_key`**: `str`
    -   Description: Your EODHD API key. This is a **required** setting.
    -   Environment Variable: `EODHD_API_KEY`
-   **`eodhd_api_url`**: `str`
    -   Description: The base URL for the EODHD news API.
    -   Environment Variable: `EODHD_API_URL_OVERRIDE`
    -   Default: `"https://eodhd.com/api/news"`
-   **`eodhd_default_timeout`**: `int`
    -   Description: Default timeout in seconds for EODHD API requests.
    -   Environment Variable: `EODHD_DEFAULT_TIMEOUT_OVERRIDE`
    -   Default: `100`
-   **`eodhd_default_max_retries`**: `int`
    -   Description: Default maximum number of retries for failed EODHD API requests.
    -   Environment Variable: `EODHD_DEFAULT_MAX_RETRIES_OVERRIDE`
    -   Default: `3`
-   **`eodhd_default_backoff_factor`**: `float`
    -   Description: Default backoff factor for retrying EODHD API requests.
    -   Environment Variable: `EODHD_DEFAULT_BACKOFF_FACTOR_OVERRIDE`
    -   Default: `1.5`
-   **`eodhd_default_limit`**: `int`
    -   Description: Default limit for the number of news articles to fetch from EODHD.
    -   Environment Variable: `EODHD_DEFAULT_LIMIT_OVERRIDE`
    -   Default: `50`

#### Gemini API Configuration

-   **`gemini_api_key`**: `str`
    -   Description: Your Google Gemini API key. This is a **required** setting.
    -   Environment Variable: `GEMINI_API_KEY`

#### Embeddings Configuration

-   **`embeddings_default_model`**: `str`
    -   Description: The default model to use for generating text embeddings.
    -   Environment Variable: `EMBEDDINGS_DEFAULT_MODEL`
    -   Default: `"text-embedding-004"`
-   **`embeddings_default_task_type`**: `str`
    -   Description: The default task type for text embeddings (e.g., "SEMANTIC_SIMILARITY").
    -   Environment Variable: `EMBEDDINGS_DEFAULT_TASK_TYPE`
    -   Default: `"SEMANTIC_SIMILARITY"`
-   **`embeddings_model_dimensions`**: `Dict[str, int]`
    -   Description: A dictionary mapping embedding model names to their output dimensions.
    -   Environment Variable: `EMBEDDINGS_MODEL_DIMENSIONS` (as a JSON string, e.g., `'''{"text-embedding-004": 768, "custom-model": 1024}'''`)
    -   Default: `{"text-embedding-004": 768}` (Custom values merged with/override defaults)

#### ReRanker Configuration

-   **`reranker_default_model`**: `str`
    -   Description: The default model to use for the ReRanker component.
    -   Environment Variable: `RERANKER_DEFAULT_MODEL`
    -   Default: `"gemini-2.0-flash"`

#### TextProcessor Configuration

-   **`textprocessor_max_tokens_per_chunk`**: `int`
    -   Description: The maximum number of tokens per chunk when processing text.
    -   Environment Variable: `TEXTPROCESSOR_MAX_TOKENS_PER_CHUNK`
    -   Default: `2048`

#### Database Configuration

-   **`database_path`**: `str`
    -   Description: The file path for the SQLite database used by the `ArticleManager`.
    -   Environment Variable: `DATABASE_PATH_OVERRIDE`
    -   Default: `os.path.join(os.getcwd(), "financial_news.db")` (i.e., `financial_news.db` in the current working directory)

#### ChromaDB Configuration

-   **`chroma_default_collection_name`**: `str`
    -   Description: The default name for the collection in ChromaDB.
    -   Environment Variable: `CHROMA_DEFAULT_COLLECTION_NAME`
    -   Default: `"financial_news_embeddings"`
-   **`chroma_default_persist_directory`**: `str`
    -   Description: The directory where ChromaDB should persist its data.
    -   Environment Variable: `CHROMA_DEFAULT_PERSIST_DIRECTORY`
    -   Default: `os.path.join(os.getcwd(), "chroma_db")` (i.e., `chroma_db` in the current working directory)
-   **`chroma_default_embedding_dimension`**: `int`
    -   Description: The embedding dimension to configure ChromaDB with, derived from the `embeddings_default_model` and its dimension specified in `embeddings_model_dimensions`.
    -   Default: `768` (based on the default `text-embedding-004` model)

### Methods

#### `get(key: str, default: Optional[Any] = None) -> Any`

Retrieves a configuration value by its key. This method is a generic way to access configuration properties.

-   **Parameters:**
    -   `key` (`str`): The configuration key (e.g., `"eodhd_api_key"`). The method internally converts this to the corresponding attribute name (e.g., `_eodhd_api_key`).
    -   `default` (`Optional[Any]`): The value to return if the key is not found. Defaults to `None`.
-   **Returns:** (`Any`) The configuration value or the default.

### Helper Methods (Internal)

The class also uses internal helper methods `_get_required_env(key: str)` and `_get_env(key: str, default: str)` to fetch environment variables, with the former raising a `ValueError` if a required variable is missing. These are not intended for direct external use.

## Global `config` Instance

A global instance of the `Config` class, named `config`, is created and available for direct import and use throughout the application:

```python
from financial_news_rag.config import config

# Example usage:
api_key = config.eodhd_api_key
print(f"Using EODHD API Key: {api_key}")
```
This instance provides easy access to all configuration properties.
