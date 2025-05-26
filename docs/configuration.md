# Configuration (`config.py`)

This document provides an in-depth explanation of the `Config` class found in `financial_news_rag.config`, which manages the configuration for the Financial News RAG application. Understanding these settings is crucial for customizing the behavior of the system, especially when initializing the `FinancialNewsRAG` orchestrator.

## Overview

The `Config` class centralizes configuration management by:

1.  Loading settings from environment variables.
2.  Utilizing a `.env` file in the root of your project directory for ease of development.
3.  Providing default values for settings that are not explicitly set.

An instance of this class is automatically created as `config` when you import `from financial_news_rag.config import config`. The `FinancialNewsRAG` orchestrator uses this `config` instance by default but also allows overriding these settings via its constructor parameters.

## Loading Configuration

Configurations are loaded in the following order of precedence:

1.  **Environment Variables:** Explicitly set system environment variables.
2.  **`.env` File:** Variables defined in a `.env` file located in your project's root directory. This file is loaded automatically if it exists.
3.  **Default Values:** Predefined defaults within the `Config` class if no environment variable or `.env` entry is found.

### Required API Keys

Certain API keys are essential for the application to function:

*   `EODHD_API_KEY`: For fetching financial news from EODHD.
*   `GEMINI_API_KEY`: For accessing Google Gemini models (used for embeddings and re-ranking).

If these are not set either as environment variables or in your `.env` file, the `Config` class will raise a `ValueError` upon initialization.

**Example `.env` file:**

```env
EODHD_API_KEY="your_eodhd_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"

# Optional Overrides
# EODHD_API_URL_OVERRIDE="https://custom.eodhd.com/api/news"
# DATABASE_PATH_OVERRIDE="/path/to/your/financial_news.db"
# CHROMA_DEFAULT_PERSIST_DIRECTORY="/path/to/your/chroma_db"
```

## Configuration Parameters

Below are the configurable parameters, their corresponding environment variables, default values, and descriptions. You can access these values through the properties of the `config` object (e.g., `config.eodhd_api_key`).

### 1. EODHD API Configuration

Settings related to the EOD Historical Data API client.

| Property                          | Environment Variable                     | Default Value                      | Description                                                                 |
| :-------------------------------- | :--------------------------------------- | :--------------------------------- | :-------------------------------------------------------------------------- |
| `eodhd_api_key`                   | `EODHD_API_KEY`                          | **Required**                       | Your API key for EOD Historical Data.                                       |
| `eodhd_api_url`                   | `EODHD_API_URL_OVERRIDE`                 | `"https://eodhd.com/api/news"`     | The base URL for the EODHD news API.                                        |
| `eodhd_default_timeout`           | `EODHD_DEFAULT_TIMEOUT_OVERRIDE`         | `100` (seconds)                    | Default timeout for API requests to EODHD.                                  |
| `eodhd_default_max_retries`       | `EODHD_DEFAULT_MAX_RETRIES_OVERRIDE`     | `3`                                | Default maximum number of retries for failed EODHD API requests.            |
| `eodhd_default_backoff_factor`    | `EODHD_DEFAULT_BACKOFF_FACTOR_OVERRIDE`  | `1.5`                              | Backoff factor for retrying EODHD API requests (e.g., 1.5 means 50% longer wait). |
| `eodhd_default_limit`             | `EODHD_DEFAULT_LIMIT_OVERRIDE`           | `50`                               | Default limit for the number of news articles to fetch per API call.        |

### 2. Gemini API Configuration

Settings related to Google Gemini API.

| Property           | Environment Variable | Default Value | Description                                  |
| :----------------- | :------------------- | :------------ | :------------------------------------------- |
| `gemini_api_key`   | `GEMINI_API_KEY`     | **Required**  | Your API key for Google Gemini services.     |

### 3. Embeddings Configuration

Settings for generating text embeddings.

| Property                          | Environment Variable                | Default Value                     | Description                                                                                                |
| :-------------------------------- | :---------------------------------- | :-------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| `embeddings_default_model`        | `EMBEDDINGS_DEFAULT_MODEL`          | `"text-embedding-004"`            | The default Gemini model to use for generating text embeddings.                                            |
| `embeddings_default_task_type`    | `EMBEDDINGS_DEFAULT_TASK_TYPE`      | `"SEMANTIC_SIMILARITY"`           | The default task type for embeddings, influencing how embeddings are generated.                            |
| `embeddings_model_dimensions`     | `EMBEDDINGS_MODEL_DIMENSIONS`       | `{"text-embedding-004": 768}`     | A JSON string defining the embedding dimensions for different models. Custom values merge with and override defaults. Example: `''''{"your-model": 1024, "text-embedding-004": 768}''''' |
| `chroma_default_embedding_dimension`| (Derived)                           | `768` (for default model)         | The embedding dimension used by ChromaDB, derived from `embeddings_default_model` and `embeddings_model_dimensions`. |

**Note on `EMBEDDINGS_MODEL_DIMENSIONS`:**
If you use a custom embedding model, you might need to specify its dimension here. The value should be a JSON string. For example, if your model `custom-model-xyz` has 1024 dimensions, you would set the environment variable like this:
`EMBEDDINGS_MODEL_DIMENSIONS='{"custom-model-xyz": 1024, "text-embedding-004": 768}'`
The application will merge this with its internal defaults. If the JSON is invalid, a warning is printed, and only default dimensions are used.

### 4. ReRanker Configuration

Settings for the re-ranking component.

| Property                 | Environment Variable       | Default Value          | Description                                          |
| :----------------------- | :------------------------- | :--------------------- | :--------------------------------------------------- |
| `reranker_default_model` | `RERANKER_DEFAULT_MODEL`   | `"gemini-2.0-flash"`   | The default Gemini model to use for re-ranking search results. |

### 5. TextProcessor Configuration

Settings for text processing and chunking.

| Property                             | Environment Variable                     | Default Value | Description                                                              |
| :----------------------------------- | :--------------------------------------- | :------------ | :----------------------------------------------------------------------- |
| `textprocessor_max_tokens_per_chunk` | `TEXTPROCESSOR_MAX_TOKENS_PER_CHUNK`     | `2048`        | The maximum number of tokens allowed in a single chunk of text before embedding. |

### 6. Database Configuration (SQLite)

Settings for the SQLite database used to store article metadata and content.

| Property        | Environment Variable        | Default Value                               | Description                                                                   |
| :-------------- | :-------------------------- | :------------------------------------------ | :---------------------------------------------------------------------------- |
| `database_path` | `DATABASE_PATH_OVERRIDE`    | `"./financial_news.db"` (in current dir)    | The file path for the SQLite database. Defaults to `financial_news.db` in the current working directory. |

### 7. ChromaDB Configuration

Settings for the ChromaDB vector store.

| Property                             | Environment Variable                     | Default Value                               | Description                                                                                             |
| :----------------------------------- | :--------------------------------------- | :------------------------------------------ | :------------------------------------------------------------------------------------------------------ |
| `chroma_default_collection_name`     | `CHROMA_DEFAULT_COLLECTION_NAME`         | `"financial_news_embeddings"`               | The default name for the collection within ChromaDB where embeddings are stored.                        |
| `chroma_default_persist_directory`   | `CHROMA_DEFAULT_PERSIST_DIRECTORY`       | `"./chroma_db"` (in current dir)            | The directory where ChromaDB should persist its data. Defaults to `chroma_db/` in the current working directory. |

## Customization via `FinancialNewsRAG` Orchestrator

When you initialize the `FinancialNewsRAG` orchestrator, you can override many of these configuration settings by passing them as arguments to its constructor.

For example:

```python
from financial_news_rag.orchestrator import FinancialNewsRAG

# Initialize with custom settings, overriding defaults or .env values
rag_system = FinancialNewsRAG(
    eodhd_api_key="your_eodhd_key",  # Explicitly provided
    gemini_api_key="your_gemini_key", # Explicitly provided
    db_path="/custom/path/to/articles.db",
    chroma_persist_dir="/custom/path/to/vector_store",
    chroma_collection_name="my_custom_collection",
    max_tokens_per_chunk=1024
)

# If a parameter is not provided, it will fall back to the value from Config
# (which loads from environment variables or uses defaults)
# e.g., rag_system_default = FinancialNewsRAG() will use all settings from Config
```

Refer to the `FinancialNewsRAG` class documentation in `orchestrator.md` (once available) or its docstrings for details on which parameters can be overridden.

## Accessing Configuration Values

While the `FinancialNewsRAG` orchestrator handles most configuration internally, you can directly access configuration values using the global `config` instance:

```python
from financial_news_rag.config import config

print(f"EODHD API Key: {config.eodhd_api_key}")
print(f"Default Embedding Model: {config.embeddings_default_model}")

# Generic get method (though properties are generally preferred)
# custom_setting = config.get("MY_CUSTOM_SETTING_ENV_VAR", "default_value_if_not_set")
```

This provides flexibility if you need to access configuration settings outside the main orchestrator flow.
