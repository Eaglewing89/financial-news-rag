# Table of Contents

1. [Project Overview](#project-overview)
2. [Core Objectives](#core-objectives)
3. [Technical Implementation](#technical-implementation)
   - [Tech Stack](#tech-stack)
   - [API and Deployment (Future Scope)](#api-and-deployment-future-scope)
   - [Data Sources](#data-sources)
   - [Data Management](#data-management)
4. [Configuration Management](#configuration-management)
5. [Error Handling Strategy](#error-handling-strategy)
6. [Text Processing Pipeline](#text-processing-pipeline)
7. [Core Module Interface (MVP)](#core-module-interface-mvp)
8. [ChromaDB Integration Details](#chromadb-integration-details)
9. [Gemini API Integration](#gemini-api-integration)
10. [Embedding Model](#embedding-model)
11. [Testing Strategy](#testing-strategy)
    - [Testing Plan and Documentation](#testing-plan-and-documentation)
12. [Documentation](#documentation)
13. [Implementation Plan](#implementation-plan)
14. [Required Packages](#required-packages)
15. [References](#references)

# Financial News RAG Module Specification

## Project Overview

**MVP Requirements:**
- Implement a basic Retrieval Augmented Generation (RAG) system for financial news as a focused Python module
- Fetch, process, and enable semantic search over financial news articles
- Demonstrate core RAG concepts and workflows as a school project
- Provide clean, well-documented interfaces for future integration
- Design with flexibility to be used directly in Python projects or later wrapped in an API

**Future Enhancements:**
- Full integration with a portfolio management system
- Support for multiple AI agents with different information needs
- Advanced analytics and sentiment analysis capabilities
- Comprehensive monitoring and scaling features
- Potential implementation as a standalone API service (separate project)

## Core Objectives

**MVP Requirements:**
1. Fetch financial news articles from a reliable API that provides **full article content** (EODHD API).
2. Process and clean text content of news articles
3. Generate embeddings using a serverless embedding model (Google's text-embedding-004)
4. Store news articles and embeddings in ChromaDB
5. Implement basic semantic search functionality
6. Use Google's Gemini 2.0 Flash API for re-ranking search results
7. Create a well-structured Python module with clean interfaces

**Future Enhancements:**
1. Integrate multiple news sources with fallback mechanisms (e.g., Marketaux for snippets if full content isn't needed for a specific task, Finnhub.io, Newsfilter.io).
2. Implement advanced filtering by company, sector, and time range
3. Add source credibility scoring and article relevance metrics
4. Create automated data refresh and maintenance workflows
5. Develop performance monitoring and cost optimization
6. Potential deployment as a FastAPI service (separate project)

## Technical Implementation

### Tech Stack

**MVP Requirements:**
- Python 3.10+
- ChromaDB for vector storage
- Serverless embedding API (Google's text-embedding-004) 
- Google Gemini API for re-ranking (gemini-2.0-flash)
- Basic Python libraries for text processing (NLTK, BeautifulSoup)
- Simple dependency management (requirements.txt)
- Environment management with Python venv
- See [model_details.md](model_details.md) for detailed model specifications

**Future Enhancements:**
- More sophisticated embedding models as needed
- Advanced monitoring tools
- CI/CD pipeline integration
  
### API and Deployment (Future Scope)
> The core module is designed with clean interfaces that could later be exposed via API without major refactoring, but the API layer itself is intentionally separated from this project's scope.

**Potential Future API Implementation:**
- FastAPI for robust API interfaces
- Docker for containerization
- API authentication and rate limiting
- Swagger/OpenAPI documentation

### Data Sources

**MVP Requirements:**
- **Primary Source:** [EODHD API](https://eodhd.com/financial-apis/stock-market-financial-news-api) for fetching **full news articles**.
  - Specific endpoint: `GET https://eodhd.com/api/news`
  - Authentication via API token in query parameters (`api_token`).
  - Free tier usage with appropriate rate limit handling (e.g., 5 API calls per news request, 20 calls/day limit).
  - See [eodhd_api.md](eodhd_api.md) for detailed integration, parameters, and response structure.
- Basic error handling for API failures.
- Simple caching to minimize redundant calls (especially important for EODHD due to call costs).

**Future Enhancements:**
- Integrate other financial news APIs (Finnhub.io, Newsfilter.io, etc.) for broader coverage or fallback.
- **Secondary/Potential Future Source (Not for RAG MVP):** [Marketaux.com](https://www.marketaux.com/) for article **snippets**, entity/sentiment data if needed for other tasks.
  - Marketaux is **not** used for the primary RAG pipeline due to providing only snippets.
  - See [marketaux_api.md](marketaux_api.md) for its details.
- SEC EDGAR filings integration
- Source credibility scoring system
- Rate-limiting and advanced request management

### Data Management

**MVP Requirements:**
- **Primary Data Store (SQLite):** A local SQLite database will serve as the primary store for all fetched article metadata and content (both raw and processed). This includes detailed information as outlined in the `articles` table schema within the [EODHD API Integration Guide](./eodhd_api.md#articles-table). This approach ensures data persistence, facilitates detailed pipeline status tracking, and allows for robust recovery from processing failures.
- **Vector Store (ChromaDB):** ChromaDB will be used exclusively for storing embeddings of the processed article content. Each entry in ChromaDB will contain the embedding vector and a reference (e.g., `url_hash`) back to the corresponding article record in the SQLite database.
- Article metadata schema for SQLite is detailed in [eodhd_api.md](./eodhd_api.md#articles-table).
- Simple local file storage for configurations (.env files).
- Manual refresh process for updating data in SQLite and subsequently in ChromaDB.

**Future Enhancements:**
- Automated data refresh strategies for both SQLite and ChromaDB
- Data retention policies
- Versioning for embeddings
- Performance monitoring for database
- Backup and recovery procedures

## Configuration Management

**MVP Requirements:**
- Environment variable management using python-dotenv
- Standard .env file with required API keys:
  ```
  GEMINI_API_KEY=your_gemini_api_key
  EODHD_API_KEY=your_eodhd_api_key
  MARKETAUX_API_KEY=your_marketaux_api_key # Optional, if Marketaux is used for secondary tasks
  ```
- Configuration loading pattern:
  ```python
  from dotenv import load_dotenv
  import os
  
  load_dotenv()
  gemini_api_key = os.getenv('GEMINI_API_KEY')
  eodhd_api_key = os.getenv('EODHD_API_KEY')
  marketaux_api_key = os.getenv('MARKETAUX_API_KEY') # Load if used
  ```
- Configuration validation on module initialization
- Separate configuration for development/testing environments
- Documentation of required environment variables

**Future Enhancements:**
- Secrets management for production environments
- Configuration versioning
- Dynamic configuration reloading
- Service discovery integration

## Error Handling Strategy

**MVP Requirements:**
- Comprehensive exception handling for external API calls, retry logic, and rate limiting. See [eodhd_api.md#error-codes--handling](eodhd_api.md#error-codes--handling) for specific error codes, rate limits, and detailed strategies.
- Graceful degradation for non-critical failures.
- Detailed logging of errors with context information.
- User-friendly error messages for common failure scenarios.

**Future Enhancements:**
- Circuit breaker pattern for failing services
- Fallback to alternative data sources when primary is unavailable
- Comprehensive error monitoring and alerting system
- Error aggregation and analysis for identifying patterns

## Text Processing Pipeline

**MVP Requirements:**
- Standardized text cleaning and normalization, sentence splitting, deduplication, and Unicode normalization. The raw article content will be stored in the SQLite database, and a processed version will also be stored there before being used for embedding. See [text_processing_pipeline.md](text_processing_pipeline.md) for the full implementation pipeline and code examples.
- Entity extraction and normalization will primarily rely on data from the EODHD API response (e.g., `symbols`, `tags`), stored in SQLite.  
- Chunking strategy and tokenization are described in [model_details.md](model_details.md#chunking-strategy) and [text_processing_pipeline.md](text_processing_pipeline.md#chunking-strategies-for-rag).
- Optional summarization for long articles.

**Future Enhancements:**
- Advanced NLP for entity relationship extraction
- Topic modeling for automatic categorization
- Sentiment analysis specific to financial texts
- Named entity recognition fine-tuned for financial domain
- Multilingual support for global financial news

## Core Module Interface (MVP)

**MVP Requirements:**
The financial news RAG system will be implemented as a focused Python module with clean, well-documented interfaces:

```python
def search_news(
    query: str,
    max_results: int = 5,
    rerank: bool = True,
    entity_filter: Optional[List[str]] = None,
    date_range: Optional[Tuple[datetime, datetime]] = None
) -> List[Dict]:
    """
    Search for financial news articles based on semantic similarity to query.
    
    Args:
        query: The search query text
        max_results: Maximum number of results to return
        rerank: Whether to apply Gemini re-ranking
        entity_filter: Optional list of entity symbols to filter by
        date_range: Optional date range to filter articles
        
    Returns:
        List of dictionaries containing:
            - uuid: Unique article identifier
            - title: Article title
            - source: News source name
            - date: Publication date
            - url: Link to original article
            - content: Full or snippet of article text
            - relevance_score: Similarity score
            - entities: List of related financial entities
    """
    pass

def add_news_articles(articles: List[Dict]) -> Dict[str, int]:
    """
    Add new articles to the vector database.
    
    Args:
        articles: List of article dictionaries with title, content, etc.
        
    Returns:
        Statistics about added articles (total, succeeded, failed)
    """
    pass

def refresh_news_data(
    days_back: int = 3,
    symbols: Optional[List[str]] = None,
    min_sentiment: float = -1.0
) -> Dict[str, int]:
    """
    Manually trigger refresh of news database.
    
    Args:
        days_back: Number of days of news to fetch
        symbols: Optional list of ticker symbols to focus on
        min_sentiment: Minimum sentiment score threshold
        
    Returns:
        Statistics about the refresh operation
    """
    pass

def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the current news collection.
    
    Returns:
        Dictionary with statistics like:
            - total_articles: Total number of articles
            - oldest_article: Date of oldest article
            - newest_article: Date of newest article
            - top_entities: Most common entities
            - storage_size: Approximate storage size
    """
    pass
```

**Future Enhancements to Core Module:**

```python
def get_company_news(
    ticker: str, 
    time_range: Optional[Tuple[datetime, datetime]] = None,
    max_results: int = 10
) -> List[Dict]:
    """Get news specifically about a company by ticker."""
    pass

def get_sector_news(
    sector: str,
    time_range: Optional[Tuple[datetime, datetime]] = None,
    max_results: int = 10
) -> List[Dict]:
    """Get news about a specific market sector."""
    pass

def get_market_sentiment(
    topic: Optional[str] = None,
    time_range: Optional[Tuple[datetime, datetime]] = None
) -> Dict:
    """Analyze sentiment across news for a topic or market overall."""
    pass
```

> **Note on API Implementation:** The module is designed to be easily wrapped by a FastAPI service in the future or integrated into multi-agent systems, but the API implementation is outside the current project scope.

## ChromaDB Integration Details

**Implementation:**
- ChromaDB setup and management is implemented in [`src/financial_news_rag/chroma_manager.py`](../src/financial_news_rag/chroma_manager.py) as the `ChromaDBManager` class.
- Main functionalities include:
  - Collection initialization and persistence management
  - Adding and updating embeddings with reference to SQLite
  - Querying similar vectors with optional metadata filtering
  - Collection status reporting
  - Deleting embeddings by article ID

**Example Usage:**
```python
from financial_news_rag.chroma_manager import ChromaDBManager

# Initialize with default or custom parameters
chroma_manager = ChromaDBManager(
    persist_directory="./chroma_db",  # Custom persistence location
    collection_name="financial_news_embeddings"
)

# Add embeddings for an article
chroma_manager.add_embeddings(
    article_url_hash="unique_article_hash",
    chunk_embeddings=[{
        "chunk_id": "unique_article_hash_0",
        "embedding": [0.1, 0.2, ...],  # 768-dimension vector
        "text": "Article chunk text content",
        "metadata": {
            "article_url_hash": "unique_article_hash",
            "chunk_index": 0,
            "published_at_timestamp": 1678886400
        }
    }]
)

# Query for similar embeddings
results = chroma_manager.query_embeddings(
    query_embedding=[0.1, 0.2, ...],  # Query vector
    n_results=5,
    filter_metadata={"source_query_tag": "TECHNOLOGY"}  # Optional filtering
)

# Get collection statistics
status = chroma_manager.get_collection_status()
```

For a complete end-to-end example of embedding generation and storage in ChromaDB, see [`examples/generate_and_store_embeddings_example.py`](../examples/generate_and_store_embeddings_example.py).

For comprehensive unit tests, see [`tests/test_chroma_manager.py`](../tests/test_chroma_manager.py).

## Gemini API Integration

**MVP Requirements:**
- Use Gemini 2.0 Flash for re-ranking search results (see [model_details.md#gemini-llm-gemini-20-flash](model_details.md#gemini-llm-gemini-20-flash) for detailed specifications and the `rerank_with_gemini` function implementation in [model_details.md#re-ranking-implementation](model_details.md#re-ranking-implementation)).
- Simple prompt template for relevance assessment
- Basic error handling for API failures
- Environment variable for API key management

**Future Enhancements:**
- Upgrade to newer Gemini models as they become available
- Advanced prompt engineering for better results
- Fallback mechanisms to alternative models
- Cost tracking and optimization

### Example use Gemini 2.0 Flash
See [model_details.md#example-use-gemini-20-flash](model_details.md#example-use-gemini-20-flash) for an example.

### Reranking Implementation
The `rerank_with_gemini` function is defined in [model_details.md#re-ranking-implementation](model_details.md#re-ranking-implementation).

## Embedding Model

**MVP Requirements:**
- Use Google's text-embedding-004 via Gemini API, implemented in [`src/financial_news_rag/embeddings.py`](../src/financial_news_rag/embeddings.py) as `EmbeddingsGenerator`.
- Store embeddings alongside article metadata in ChromaDB.
- Reuse the same API key configuration for both embedding and re-ranking.

**Example usage:** See [`examples/generate_embeddings_example.py`](../examples/generate_embeddings_example.py).

**Unit tests:** See [`tests/test_embeddings.py`](../tests/test_embeddings.py).

**Future Enhancements:**
- Evaluate custom domain-tuned embedding models for financial texts
- Implement more sophisticated chunking strategies with overlap
- Benchmark against alternative embedding providers

## Testing Strategy

**MVP Requirements:**
- Comprehensive unit and integration testing suite using pytest. See the `tests/` directory for implementation details and [technical_design.md](technical_design.md#testing) for testing strategy and metrics.
- End-to-end workflow testing.
- Coverage reporting with minimum threshold of 80%.
- Performance benchmarks for critical operations.
- **All details and up-to-date instructions are maintained in [testing.md](testing.md).**

**Testing Quality Metrics:**
- Precision: Accuracy of retrieved articles relative to the query
- Recall: Percentage of relevant articles actually retrieved
- F1 Score: Harmonic mean of precision and recall
- Latency: Response time for typical queries
- Throughput: Number of queries handled per unit time

### Testing Plan and Documentation

See [docs/testing.md](testing.md) for the full testing implementation plan, test types, quality metrics, and instructions for running tests. This is the single source of truth for all testing practices in this project.

## Documentation

**MVP Requirements:**
- Clear README with setup instructions
- Function-level documentation
- Example usage notebooks
- Architecture diagram
- Module import and usage examples
- Model specifications and details (see [model_details.md](model_details.md))

**Future Enhancements:**
- Detailed integration guide
- Performance optimization guide
- Contribution guidelines
- API implementation guide (for separate API project)

## Implementation Plan

**MVP Implementation Steps:**
1. Set up project structure and dependencies as a Python module
   ```
   financial-news-rag/
   ├── src/
   │   └── financial_news_rag/   # Core module with all RAG functionality
   │       ├── __init__.py
   │       ├── search.py
   │       ├── embeddings.py
   │       ├── data.py           # Will handle EODHD and potentially Marketaux fetching
   │       ├── eodhd.py          # Specific EODHD API interaction logic (new or in data.py)
   │       ├── marketaux.py      # Specific Marketaux API interaction logic (if kept for other tasks)
   │       ├── config.py
   │       └── utils.py
   ├── tests/
   │   ├── test_search.py
   │   ├── test_embeddings.py
   │   ├── test_data.py        # Tests for data fetching and processing
   │   ├── test_eodhd.py       # (new or in test_data.py)
   │   └── test_marketaux.py   # (if kept)
   ├── docs/
   │   └── ...
   ├── examples/
   │   ├── basic_search.ipynb
   │   └── refresh_data.ipynb
   ├── requirements.txt
   ├── pyproject.toml
   ├── setup.py
   └── README.md
   ```
2. Implement news fetching from the EODHD API (primary source for full articles).
3. Optionally, retain or adapt Marketaux API fetching for snippets/metadata if specific secondary uses are identified.
4. Create text processing and embedding pipeline (focused on full content from EODHD).
5. Implement basic search functionality
6. Add Gemini re-ranking capability
7. Create clean, well-documented module interfaces
8. Write tests and documentation

**Future Considerations:**
- The module is designed to be API-friendly for potential future deployment as a standalone service, but the API layer itself is intentionally separated from this project's scope
- Potential deployment options include:
  - Direct import in other Python projects
  - Integration in multi-agent systems
  - Wrapped in a FastAPI service (as a separate project)
  - Employer showcase in various configurations

## Required Packages

```
# Core functionality
chromadb
google-generativeai
requests
python-dotenv

# Text processing
nltk
beautifulsoup4

# Testing
pytest
pytest-cov

# Development
black
isort
flake8
```

## References

- [eodhd_api.md](eodhd_api.md): EODHD API usage, field definitions, error codes, and filtering options (Primary RAG source).
- [marketaux_api.md](marketaux_api.md): Marketaux API usage, field definitions, error codes, and filtering options (Secondary/potential source for snippets).
- [model_details.md](model_details.md): Embedding and LLM model specifications, chunking/tokenization details, example use.
- [technical_design.md](technical_design.md): System architecture, ChromaDB schema, error handling strategies.
- [text_processing_pipeline.md](text_processing_pipeline.md): Text cleaning, chunking, and preprocessing pipeline details.

## Current implementation and outlook

## 1. Overview

This project aims to build a Retrieval Augmented Generation (RAG) system focused on financial news. The system will fetch news articles, process them, generate embeddings, store them in a vector database, and use this information to answer user queries about financial topics, potentially with a focus on specific companies or market trends.

## 2. Core Components (Low-Level Classes)

The system will be built upon a set of modular, low-level classes, each responsible for a specific part of the pipeline:

1.  **`EODHDClient`**: Responsible for fetching financial news articles and potentially other financial data from the EODHD API.
2.  **`ArticleManager`**: Manages the storage and status of articles in a local SQLite database (`financial_news.db`). This includes tracking processing stages (cleaned, chunked, embedded) and logging API calls.
3.  **`TextProcessor`**: Handles the cleaning of raw article text (e.g., removing HTML, normalizing whitespace) and chunking the cleaned text into manageable pieces suitable for embedding.
4.  **`EmbeddingsGenerator`**: Generates vector embeddings for the text chunks using a chosen sentence transformer model.
5.  **`ChromaDBManager`**: Manages the storage and retrieval of text chunks and their corresponding embeddings in a ChromaDB vector database.

## 3. Key Features

*   **Data Ingestion**: Fetch news from EODHD API.
*   **Data Processing**: Clean and chunk news articles.
*   **Embedding Generation**: Create embeddings for text chunks.
*   **Vector Storage**: Store and index embeddings in ChromaDB.
*   **Information Retrieval**: Retrieve relevant text chunks based on user query embeddings.
*   **Answer Generation (Future)**: Use a Large Language Model (LLM) to synthesize answers based on retrieved chunks.
*   **Modular Design**: Clearly separated components for easier development, testing, and maintenance.
*   **Status Tracking**: `ArticleManager` will maintain the state of each article through the processing pipeline.

## 4. Workflow

1.  **Fetch**: `EODHDClient` fetches news (e.g., for a specific ticker and date range).
2.  **Store & Pre-process**: 
    *   `ArticleManager` stores the raw articles.
    *   `TextProcessor` cleans and chunks the articles. 
    *   `ArticleManager` updates the status of articles and stores chunks.
3.  **Embed & Store**: 
    *   `EmbeddingsGenerator` creates embeddings for the chunks.
    *   `ChromaDBManager` stores the chunks and their embeddings.
    *   `ArticleManager` updates the embedding status.
4.  **Query & Retrieve (RAG)**:
    *   User provides a query.
    *   Query is embedded using `EmbeddingsGenerator`.
    *   `ChromaDBManager` retrieves the most relevant chunks from the vector store.
    *   (Future) Retrieved chunks are passed to an LLM to generate a concise answer.

## 5. Example Workflow Demonstration

The script `examples/end_to_end_pipeline_example.py` demonstrates the complete flow from fetching news using `EODHDClient`, processing and storing articles via `ArticleManager` and `TextProcessor`, generating embeddings with `EmbeddingsGenerator`, and storing them in ChromaDB using `ChromaDBManager`.

## 6. Future Enhancements

*   **Re-ranking Class**: A component to re-rank the retrieved chunks for relevance before passing them to the LLM.
*   **High-Level Orchestrator Class**: A class to manage the end-to-end RAG pipeline, simplifying its execution.
*   **LLM Integration**: Incorporate a language model for answer synthesis.
*   **Advanced Querying**: Support for more complex query types and filtering.
*   **User Interface**: A simple UI for interacting with the system.
*   **Scalability**: Improvements for handling larger volumes of data and user traffic.

## 7. Technology Stack (Initial)

*   Python
*   SQLite (for `ArticleManager`)
*   Sentence Transformers (for `EmbeddingsGenerator`)
*   ChromaDB (for `ChromaDBManager`)
*   EODHD API (for data acquisition)
*   NLTK (for text processing tasks like sentence tokenization within `TextProcessor`)

This specification provides a foundational outline. Details may evolve as the project progresses.
