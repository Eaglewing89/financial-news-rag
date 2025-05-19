# Financial News RAG Module

A Retrieval Augmented Generation (RAG) system for financial news. This Python module fetches, processes, and enables semantic search over financial news articles with re-ranking capabilities.

## Features

- Fetch complete financial news articles from EODHD API
- Filter news by ticker symbols, topic tags, date ranges, and more
- Process and clean article text for embedding generation
- Store articles and vectors in ChromaDB for efficient semantic search
- Re-rank results using Gemini 2.0 Flash for improved relevance
- Robust error handling with exponential backoff retry mechanism
- SQLite database for tracking processed articles and API usage

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-news-rag.git
cd financial-news-rag

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Configuration

Create a `.env` file in the root directory with your API keys:

```
EODHD_API_KEY=your_eodhd_api_key
GEMINI_API_KEY=your_gemini_api_key
```

You can obtain an EODHD API key by signing up at [eodhd.com](https://eodhd.com/).

## Usage Examples

### Fetching News Articles with EODHD API

```python
from financial_news_rag import EODHDClient

# Create EODHD client
client = EODHDClient()

# Fetch news articles by topic tag
merger_news = client.fetch_news(
    tag="MERGERS AND ACQUISITIONS",
    from_date="2025-05-01",
    to_date="2025-05-19",
    limit=5
)

# Fetch news for a specific ticker symbol
tech_news = client.fetch_news(
    symbols=["AAPL.US"],  # Use a single symbol
    from_date="2025-05-01",
    to_date="2025-05-19",
    limit=10
)

# Process the normalized articles
for article in merger_news:
    print(f"Title: {article['title']}")
    print(f"Published: {article['published_at']}")
    print(f"URL: {article['url']}")
    print(f"Sentiment: {article['sentiment']['polarity']}")
    print("---")
```

### Semantic Search with Re-ranking

```python
from financial_news_rag import search_news

# Search for news about AI acquisitions
results = search_news(
    query="AI companies acquiring startups",
    max_results=5,
    rerank=True,
    date_range=("2025-01-01", "2025-05-19")
)

# Search with entity filter
results = search_news(
    query="earnings report",
    entity_filter=["AAPL.US", "GOOGL.US"],
    max_results=5
)
```
from financial_news_rag import search_entities

# Search for AI-related companies
entities = search_entities(
    search="AI",
    entity_types=["equity"],
    countries=["us"]
)
```

## Component Architecture

The Financial News RAG system consists of several interconnected components:

1. **News Fetcher**: Primarily uses EODHD API to fetch full news articles. Marketaux API is available as a secondary source for news snippets.
2. **Text Processing Pipeline**: Cleans and normalizes article content for embedding generation.
3. **SQLite Database**: Tracks processed articles and API usage to prevent duplicates and optimize API calls.
4. **Embedding Generator**: Generates vector embeddings for articles using Google's text-embedding-004 model.
5. **Vector Database**: Stores articles and embeddings in ChromaDB for efficient semantic search.
6. **Semantic Search Engine**: Finds relevant articles based on query similarity.
7. **Gemini Re-ranker**: Uses Gemini 2.0 Flash model to improve search result relevance.

## Documentation

See the `docs/` directory for detailed documentation:

- [Project Specification](docs/project_spec.md)
- [Technical Design](docs/technical_design.md)
- [EODHD API Integration](docs/eodhd_api.md)
- [Text Processing Pipeline](docs/text_processing_pipeline.md)
- [Model Details](docs/model_details.md)

For development guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Example Notebooks

The `examples/` directory contains Jupyter notebooks demonstrating various features:

- [EODHD Example](examples/eodhd_example.ipynb): Working with the EODHD API
- [Basic Search](examples/basic_search.ipynb): Demonstrates semantic search capabilities
- [Refresh Data](examples/refresh_data.ipynb): Shows how to refresh the news database

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=financial_news_rag
```
