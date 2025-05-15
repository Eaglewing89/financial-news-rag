# Financial News RAG Module

A Retrieval Augmented Generation (RAG) system for financial news. This Python module fetches, processes, and enables semantic search over financial news articles.

## Features

- Fetch financial news articles from Marketaux API
- Filter by company symbols, sentiment, date ranges, and more
- Entity information extraction
- Robust error handling with retries and rate limiting
- Clean and normalized article data structure

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
MARKETAUX_API_KEY=your_marketaux_api_key
GEMINI_API_KEY=your_gemini_api_key
```

You can obtain a Marketaux API key by signing up at [marketaux.com](https://www.marketaux.com/).

## Usage Examples

### Fetching News Articles

```python
from financial_news_rag import fetch_financial_news

# Fetch recent news for Tesla and Apple
news = fetch_financial_news(
    symbols=["TSLA", "AAPL"],
    days_back=7,
    language=["en"],
    filter_entities=True
)

# Access the articles
for article in news["data"]:
    print(f"Title: {article['title']}")
    print(f"Source: {article['source']}")
    print(f"URL: {article['url']}")
    print(f"Published: {article['published_at']}")
    print("---")
```

### Filtering by Sentiment

```python
from financial_news_rag import fetch_financial_news

# Get positive news (sentiment >= 0.2)
positive_news = fetch_financial_news(
    symbols=["MSFT"],
    sentiment_gte=0.2,
    days_back=7
)

# Get negative news (sentiment <= -0.2)
negative_news = fetch_financial_news(
    symbols=["MSFT"],
    sentiment_lte=-0.2,
    days_back=7
)
```

### Text Search

```python
from financial_news_rag import fetch_financial_news

# Search for specific terms
results = fetch_financial_news(
    search="AI earnings",
    days_back=7,
    language=["en"]
)
```

### Entity Search

```python
from financial_news_rag import search_entities

# Search for AI-related companies
entities = search_entities(
    search="AI",
    entity_types=["equity"],
    countries=["us"]
)
```

## Documentation

See the `docs/` directory for detailed documentation:

- [Project Specification](docs/project_spec.md)
- [Technical Design](docs/technical_design.md)
- [Marketaux API Integration](docs/marketaux_api.md)
- [Model Details](docs/model_details.md)
- [Text Processing Pipeline](docs/text_processing_pipeline.md)

For development guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Example Notebooks

The `examples/` directory contains Jupyter notebooks demonstrating various features:

- [Basic Search](examples/basic_search.ipynb): Demonstrates fetching and filtering news
- [Refresh Data](examples/refresh_data.ipynb): Shows how to refresh the news database

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=financial_news_rag
```
