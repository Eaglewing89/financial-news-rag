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
1. Fetch financial news articles from a single reliable API (Marketaux.com)
2. Process and clean text content of news articles
3. Generate embeddings using a serverless embedding model (Google's text-embedding-004)
4. Store news articles and embeddings in ChromaDB
5. Implement basic semantic search functionality
6. Use Google's Gemini API for re-ranking search results
7. Create a well-structured Python module with clean interfaces

**Future Enhancements:**
1. Integrate multiple news sources with fallback mechanisms (Finnhub.io, Newsfilter.io)
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
- [Marketaux.com](https://www.marketaux.com/) as primary source
  - Specific endpoint: `GET https://api.marketaux.com/v1/news/all`
  - Authentication via API token in query parameters
  - Free tier usage with appropriate rate limit handling
- Basic error handling for API failures
- Simple caching to minimize redundant calls

**Future Enhancements:**
- Multiple financial news APIs (Finnhub.io, Newsfilter.io, etc.)
- SEC EDGAR filings integration
- Source credibility scoring system
- Rate-limiting and advanced request management

### Data Management

**MVP Requirements:**
- ChromaDB for storing articles and embeddings
- Article metadata schema (see [technical_design.md](technical_design.md#chromadb-schema-definitions) and [marketaux_api.md](marketaux_api.md#response-fields) for full details)
- Simple local file storage for configurations (.env files)
- Manual refresh process for updating data

**Future Enhancements:**
- Automated data refresh strategies
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
  MARKETAUX_API_KEY=your_marketaux_api_key
  ```
- Configuration loading pattern:
  ```python
  from dotenv import load_dotenv
  import os
  
  load_dotenv()
  gemini_api_key = os.getenv('GEMINI_API_KEY')
  marketaux_api_key = os.getenv('MARKETAUX_API_KEY')
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
- Comprehensive exception handling for external API calls, retry logic, and rate limiting. See [marketaux_api.md](marketaux_api.md#error-codes--handling) and [technical_design.md](technical_design.md#error-handling-strategies) for error codes, rate limits, and detailed strategies.
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
- Standardized text cleaning and normalization, sentence splitting, deduplication, and Unicode normalization. See [text_processing_pipeline.md](text_processing_pipeline.md) for the full implementation pipeline and code examples.
- Entity extraction and normalization are based on Marketaux API metadata (see [marketaux_api.md](marketaux_api.md#response-fields)).
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

**MVP Requirements:**
- ChromaDB setup and configuration:
  ```python
  import chromadb
  
  # Initialize client with persistence
  client = chromadb.PersistentClient(path="./chroma_db")
  
  # Create collection with appropriate schema
  collection = client.create_collection(
      name="financial_news",
      metadata={"description": "Financial news articles for RAG"},
  )
  ```
- Vector storage and retrieval:
  ```python
  # Add documents to collection
  collection.add(
      documents=["Article content here..."],
      metadatas=[{"source": "Marketaux", "published_at": "2025-05-14T10:00:00Z"}],
      ids=["article-uuid"]
  )
  
  # Query the collection
  results = collection.query(
      query_texts=["What's happening with Tesla stock?"],
      n_results=5
  )
  ```
- Collection management:
  ```python
  # List collections
  collections = client.list_collections()
  
  # Get collection
  collection = client.get_collection("financial_news")
  
  # Delete old items
  collection.delete(
      where={"published_at": {"$lt": "2025-01-01T00:00:00Z"}}
  )
  ```
- Migrations and schema evolution strategy

**Future Enhancements:**
- Advanced ChromaDB filtering
- Multi-collection design for different news categories or time periods
- Backup and restore procedures for ChromaDB
- Performance tuning for larger datasets

## Gemini API Integration

**MVP Requirements:**
- Use Gemini 2.0 Flash for re-ranking search results (see [model_details.md](model_details.md) for detailed specifications)
- Simple prompt template for relevance assessment
- Basic error handling for API failures
- Environment variable for API key management

**Future Enhancements:**
- Upgrade to newer Gemini models as they become available
- Advanced prompt engineering for better results
- Fallback mechanisms to alternative models
- Cost tracking and optimization

### Example use Gemini 2.0 Flash
```python
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=gemini_api_key)

input_config=types.GenerateContentConfig(
        system_instruction="You are a wizard. You speak in riddles.",
        max_output_tokens=500,
        temperature=0.1
        )

input_contents="Explain how RAG works in a few words"

response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents=input_contents,
    config=input_config
)
print(response.text)
# From dusty tomes, knowledge I glean, then weave it with queries, a vibrant scene.
```

### Reranking Implementation

```python
def rerank_with_gemini(query, results, top_n=5):
    """Rerank search results using Gemini's understanding of relevance."""
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Prepare prompt for reranking
    system_instruction = """
    You are a financial analyst assistant. Your task is to evaluate the relevance 
    of financial news articles to a user's query. Rate each article on a scale of 0-10, 
    where 10 means the article is perfectly relevant to answering the query, and 0 means 
    it's completely irrelevant. Consider specificity, recency, and depth of information.
    
    Return a JSON array with each article's ID and its relevance score:
    [{"id": "article-id-1", "score": 8.5}, {"id": "article-id-2", "score": 6.2}, ...]
    """
    
    # Format articles for evaluation
    articles_text = "\n\n".join([
        f"Article ID: {result['id']}\nTitle: {result['metadata']['title']}\n"
        f"Date: {result['metadata']['published_at']}\n"
        f"Content: {result['content'][:500]}..."
        for result in results
    ])
    
    input_contents = f"""
    USER QUERY: {query}
    
    ARTICLES TO EVALUATE:
    {articles_text}
    
    Please rate the relevance of each article to the query.
    """
    
    # Generate relevance ratings
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=input_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=1024,
            temperature=0.1
        )
    )
    
    # Parse results and rerank
    try:
        scores = json.loads(response.text)
        # Create a mapping of article IDs to scores
        score_map = {item["id"]: item["score"] for item in scores}
        
        # Sort results by score
        sorted_results = sorted(
            results, 
            key=lambda x: score_map.get(x["id"], 0), 
            reverse=True
        )
        
        return sorted_results[:top_n]
    except:
        # Fallback to original ranking if parsing fails
        return results[:top_n]
```

### Sources
- [Google Text generation Docs](https://ai.google.dev/gemini-api/docs/text-generation)
- [Google GenerateContentConfig Docs](https://ai.google.dev/api/generate-content)

## Embedding Model

**MVP Requirements:**
- Use Google's text-embedding-004 via Gemini API (see [model_details.md](model_details.md#gemini-embedding-model-text-embedding-004) for detailed specifications and chunking/tokenization constraints).
- Store embeddings alongside article metadata in ChromaDB.
- Reuse the same API key configuration for both embedding and re-ranking.

**Future Enhancements:**
- Evaluate custom domain-tuned embedding models for financial texts
- Implement more sophisticated chunking strategies with overlap
- Benchmark against alternative embedding providers

### Example use text-embedding-004
```python
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=gemini_api_key)

input_contents="What is the meaning of life?"

input_config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")

result = client.models.embed_content(
        model="text-embedding-004",
        contents=input_contents,
        config=input_config
)
print(result.embeddings)
# [ContentEmbedding(values=[-0.0001, 0.0002, 0.0003, ...], statistics=None)]
```

## Testing Strategy

**MVP Requirements:**
- Comprehensive unit and integration testing suite using pytest. See the `tests/` directory for implementation details and [technical_design.md](technical_design.md#testing) for testing strategy and metrics.
- End-to-end workflow testing.
- Coverage reporting with minimum threshold of 80%.
- Performance benchmarks for critical operations.

**Testing Quality Metrics:**
- Precision: Accuracy of retrieved articles relative to the query
- Recall: Percentage of relevant articles actually retrieved
- F1 Score: Harmonic mean of precision and recall
- Latency: Response time for typical queries
- Throughput: Number of queries handled per unit time

**Future Enhancements:**
- Comprehensive test suite with integration tests
- Automated evaluation pipelines
- Performance benchmarking for scale
- User feedback collection system
- A/B testing framework for model improvements

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
   │       ├── data.py
   │       ├── config.py
   │       └── utils.py
   ├── tests/
   │   ├── test_search.py
   │   ├── test_embeddings.py
   │   └── test_data.py
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
2. Implement news fetching from the Marketaux API
3. Create text processing and embedding pipeline
4. Set up ChromaDB integration
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

- [marketaux_api.md](marketaux_api.md): Marketaux API usage, field definitions, error codes, and filtering options.
- [model_details.md](model_details.md): Embedding and LLM model specifications, chunking/tokenization details.
- [technical_design.md](technical_design.md): System architecture, ChromaDB schema, error handling strategies.
- [text_processing_pipeline.md](text_processing_pipeline.md): Text cleaning, chunking, and preprocessing pipeline details.
