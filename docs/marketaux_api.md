# Marketaux API Integration Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
    - [Finance & Market News](#finance--market-news)
    - [Additional API Endpoints](#additional-api-endpoints)
4. [Request & Response Examples](#request--response-examples)
5. [Error Codes & Handling](#error-codes--handling)
6. [Rate Limiting & Usage Limits](#rate-limiting--usage-limits)
7. [Free Plan Capabilities](#free-plan-capabilities)
8. [Integration Patterns](#integration-patterns)
9. [References](#references)

---

## Overview

The Marketaux API provides access to global financial news article snippets, supporting entity, sentiment, and metadata filtering. This guide details its integration into the Financial News RAG module.

**Important Note on Usage:** The Marketaux API returns only **snippets** of articles, not full content. Due to this limitation, it is **no longer the primary source for populating our RAG vector database**. The EODHD API, which provides full article content, has replaced Marketaux for this core purpose.

Marketaux may still be used for other potential features, such as:
- Simple entity sentiment analysis.
- Quick checks for news mentions without needing full content.
- Exploring broader news trends via snippets if full content processing is too resource-intensive for that specific task.

However, it will **not be included in the MVP for the RAG system's primary news ingestion pipeline.**

This guide covers authentication, endpoint usage, error handling, and best practices for any continued or future use of the Marketaux API.

---

## Authentication

- **API Token Required**: All requests require an `api_token` as a query parameter.
- **How to Obtain**: [Sign up](https://www.marketaux.com/signup) for a free account and retrieve your token from the dashboard.
- **Usage Example**: `GET ...?api_token=YOUR_API_TOKEN`
- **Best Practice**: Store your API token in a `.env` file and load it using `python-dotenv`. See [technical_design.md#configuration-management](technical_design.md#configuration-management) for the standard loading pattern.

---

## API Endpoints

### Finance & Market News
- **Endpoint**: `GET https://api.marketaux.com/v1/news/all`
- **Available on**: All plans
- **Purpose**: Retrieve snippets of the latest global financial news and filter by entities, sentiment, industry, and more. Entity analysis is provided for each identified entity in articles. Not all articles have entities; use `must_have_entities=true` or entity filters for more concise results. **Note: Returns article snippets, not full content.**

#### HTTP GET Parameters
| Name                | Required | Description |
|---------------------|----------|-------------|
| `api_token`         | Yes      | Your API token from your account dashboard. |
| `symbols`           | No       | Comma-separated entity symbols (e.g., `TSLA,AMZN,MSFT`). |
| `entity_types`      | No       | Comma-separated entity types (e.g., `index,equity`). |
| `industries`        | No       | Comma-separated industries (e.g., `Technology,Industrials`). |
| `countries`         | No       | Comma-separated country codes for exchanges (e.g., `us,ca`). |
| `sentiment_gte`     | No       | Minimum sentiment score (range: -1 to 1). E.g., `sentiment_gte=0` finds neutral or positive. |
| `sentiment_lte`     | No       | Maximum sentiment score (range: -1 to 1). E.g., `sentiment_lte=0` finds neutral or negative. |
| `min_match_score`   | No       | Minimum entity match score. |
| `filter_entities`   | No       | If `true`, only relevant entities to your query are returned per article. Default: `false`. |
| `must_have_entities`| No       | If `true`, only articles with at least one identified entity are returned. Default: `false`. |
| `group_similar`     | No       | If `true`, group similar articles. Default: `true`. |
| `search`            | No       | Search terms or advanced query (AND/OR/NOT, phrase, prefix, precedence). E.g., `"ipo" -nyse`. |
| `domains`           | No       | Comma-separated list of domains to include. |
| `exclude_domains`   | No       | Comma-separated list of domains to exclude. |
| `source_ids`        | No       | Comma-separated list of source IDs to include. |
| `exclude_source_ids`| No       | Comma-separated list of source IDs to exclude. |
| `language`          | No       | Comma-separated list of languages (e.g., `en,es`). Default: all. |
| `published_before`  | No       | Find articles published before a date. Formats: `Y-m-dTH:i:s`, `Y-m-d`, etc. |
| `published_after`   | No       | Find articles published after a date. Formats: `Y-m-dTH:i:s`, `Y-m-d`, etc. |
| `published_on`      | No       | Find articles published on a specific date. Format: `Y-m-d`. |
| `sort`              | No       | Sort by `published_on`, `entity_match_score`, `entity_sentiment_score`, or `relevance_score`. |
| `sort_order`        | No       | `desc` or `asc`. Only with `sort=entity_match_score` or `entity_sentiment_score`. Default: `desc`. |
| `limit`             | No       | Number of articles to return. Max based on plan. Default: plan max. |
| `page`              | No       | Page number for pagination. Default: 1. |

#### Advanced Search Usage
- `+` = AND, `|` = OR, `-` = NOT, `"phrase"`, `*` = prefix, `()` = precedence, `\` = escape.
- Example: `search="ipo" -nyse` (articles must include "ipo" but not NYSE)

#### Pagination
- Use `limit` and `page` to paginate results. E.g., `limit=3&page=2`.
- The response `meta` object provides `found`, `returned`, `limit`, and `page`.

#### Response Fields
| Field                                 | Description |
|----------------------------------------|-------------|
| `meta > found`                        | Number of articles found for the request. |
| `meta > returned`                     | Number of articles returned on the page. |
| `meta > limit`                        | The limit used for the request. |
| `meta > page`                         | The page number. |
| `data > uuid`                         | Unique article identifier. |
| `data > title`                        | Article title. |
| `data > description`                  | Article meta description. |
| `data > keywords`                     | Article meta keywords. |
| `data > snippet`                      | Short snippet of article body. |
| `data > url`                          | URL to the article. |
| `data > image_url`                    | URL to the article image. |
| `data > language`                     | Language of the source. |
| `data > published_at`                 | Datetime published. |
| `data > source`                       | Domain of the source. |
| `data > relevance_score`              | Relevance score (if search used). |
| `data > entities > symbol`            | Symbol of identified entity. |
| `data > entities > name`              | Name of identified entity. |
| `data > entities > exchange`          | Exchange identifier. |
| `data > entities > exchange_long`     | Exchange name. |
| `data > entities > country`           | Exchange country. |
| `data > entities > type`              | Type of entity. |
| `data > entities > industry`          | Industry of entity. |
| `data > entities > match_score`       | Strength of entity match. |
| `data > entities > sentiment_score`   | Average sentiment for entity. |
| `data > entities > highlights`        | Array of highlights for the entity. |
| `data > entities > highlights > highlight` | Snippet of text where entity found. |
| `data > entities > highlights > sentiment` | Sentiment of the highlight. |
| `data > entities > highlights > highlighted_in` | Where highlight was found (title/main_text). |
| `data > similar`                      | Array of similar articles. |

#### Example Request
```http
GET https://api.marketaux.com/v1/news/all?symbols=TSLA,AMZN,MSFT&filter_entities=true&language=en&api_token=YOUR_API_TOKEN
```

#### Example Response
```json
{
    "meta": {
        "found": 140037,
        "returned": 3,
        "limit": 3,
        "page": 1
    },
    "data": [
        {
            "uuid": "70cb577e-c2dd-4dde-b501-f713823a4939",
            "title": "Trump wins 2024, markets surge globally",
            "description": "Global markets experience a significant surge following Trump's victory in the 2024 election.",
            "keywords": "",
            "snippet": "Donald Trump has won the 2024 presidential election, defeating Vice President Kamala Harris. Trump secured the 270 electoral votes needed for victory after winn...",
            "url": "https://www.killerstartups.com/trump-wins-2024-markets-surge-globally/",
            "image_url": "https://images.killerstartups.com/wp-content/uploads/2024/11/Trump-Wins.jpg",
            "language": "en",
            "published_at": "2024-11-08T01:24:00.000000Z",
            "source": "killerstartups.com",
            "relevance_score": null,
            "entities": [
                {
                    "symbol": "TSLA",
                    "name": "Tesla, Inc.",
                    "exchange": null,
                    "exchange_long": null,
                    "country": "us",
                    "type": "equity",
                    "industry": "Consumer Cyclical",
                    "match_score": 12.133104,
                    "sentiment_score": 0.7783,
                    "highlights": [
                        {
                            "highlight": "., majority-owned by Trump, and Tesl[+253 characters]",
                            "sentiment": 0.7783,
                            "highlighted_in": "main_text"
                        }
                    ]
                }
            ],
            "similar": []
        },
        {
            "uuid": "ed35bdcd-6f6a-4007-9949-b769fbe2e36d",
            "title": "Amazon.com mulls new multi-billion dollar investment in Anthropic, the Information reports By Reuters",
            "description": "Amazon.com mulls new multi-billion dollar investment in Anthropic, the Information reports",
            "keywords": "",
            "snippet": "(Reuters) -Amazon is in talks for its second multi-billion dollar investment in artificial intelligence startup Anthropic, the Information reported on Thursday,...",
            "url": "https://www.investing.com/news/stock-market-news/amazoncom-mulls-new-multibillion-dollar-investment-in-anthropic-the-information-reports-3710319",
            "image_url": "https://i-invdn-com.investing.com/news/amazon_800x533_L_1411373482.jpg",
            "language": "en",
            "published_at": "2024-11-07T23:49:09.000000Z",
            "source": "investing.com",
            "relevance_score": null,
            "entities": [
                {
                    "symbol": "AMZN",
                    "name": "Amazon.com, Inc.",
                    "exchange": null,
                    "exchange_long": null,
                    "country": "us",
                    "type": "equity",
                    "industry": "Consumer Cyclical",
                    "match_score": 34.292408,
                    "sentiment_score": 0,
                    "highlights": [
                        {
                            "highlight": "<em>Amazon.com</em> mulls new multi-billion dollar investment in Anthropic, the Information reports By Reuters",
                            "sentiment": 0,
                            "highlighted_in": "title"
                        }
                    ]
                }
            ],
            "similar": []
        },
        {
            "uuid": "2ca2cbbf-c613-4d1c-b470-9d1bac3a256a",
            "title": "Market Soars to Record Highs: November 7, 2024 Stock Market Recap",
            "description": "The U.S. stock market experienced a historic surge on Thursday, November 7, 2024, as investors reacted to Donald Trump's unexpected victory in the 2024 U.S.",
            "keywords": "",
            "snippet": "Why Was the Market Up Today? Trump’s Victory Sparks Rally\n\nThe U.S. stock market experienced a historic surge on Thursday, November 7, 2024, as investors reac...",
            "url": "https://thestockmarketwatch.com/stock-market-news/market-soars-to-record-highs-november-7-2024-stock-market-recap/48362/",
            "image_url": "https://thestockmarketwatch.com/stock-market-news/wp-content/uploads/2024/08/5.jpg",
            "language": "en",
            "published_at": "2024-11-07T22:28:28.000000Z",
            "source": "thestockmarketwatch.com",
            "relevance_score": null,
            "entities": [
                {
                    "symbol": "TSLA",
                    "name": "Tesla, Inc.",
                    "exchange": null,
                    "exchange_long": null,
                    "country": "us",
                    "type": "equity",
                    "industry": "Consumer Cyclical",
                    "match_score": 17.491323,
                    "sentiment_score": 0.7783,
                    "highlights": [
                        {
                            "highlight": "<em>Tesla</em>, <em>Inc</em>. (TSLA), wh[+166 characters]",
                            "sentiment": 0.7783,
                            "highlighted_in": "main_text"
                        }
                    ]
                }
            ],
            "similar": []
        }
    ]
}
```

---

## Additional API Endpoints

### Entity Search
- **Endpoint**: `GET https://api.marketaux.com/v1/entity/search`
- **Purpose**: Search for all supported entities (e.g., companies, indices, currencies). Useful for discovering valid symbols for use in news queries.
- **Limit**: 50 results per request.

#### Parameters
| Name         | Required | Description |
|--------------|----------|-------------|
| `api_token`  | Yes      | Your API token. |
| `search`     | No       | Search string to find entities. |
| `symbols`    | No       | Specific symbols to return. |
| `exchanges`  | No       | Filter by exchanges (comma-separated). |
| `types`      | No       | Filter by entity types (comma-separated). |
| `industries` | No       | Filter by industries (comma-separated). |
| `countries`  | No       | Filter by ISO 3166-1 country code (comma-separated). |
| `page`       | No       | Page number for pagination. Default: 1. |

#### Response Fields
| Field                | Description |
|----------------------|-------------|
| `meta > found`       | Number of entities found. |
| `meta > returned`    | Number of entities returned. |
| `meta > limit`       | Always 50. |
| `meta > page`        | Page number. |
| `data > symbol`      | Entity symbol (ticker). |
| `data > name`        | Entity name. |
| `data > type`        | Entity type. |
| `data > industry`    | Entity industry. |
| `data > exchange`    | Exchange identifier. |
| `data > exchange_long` | Exchange name. |
| `data > country`     | Exchange country code. |

#### Example Request
```http
GET https://api.marketaux.com/v1/entity/search?search=tsla&countries=us&api_token=YOUR_API_TOKEN
```

#### Example Response
```json
{
    "meta": {
        "found": 1,
        "returned": 1,
        "limit": 50,
        "page": 1
    },
    "data": [
        {
            "symbol": "TSLA",
            "name": "Tesla, Inc.",
            "type": "equity",
            "industry": "Consumer Cyclical",
            "exchange": null,
            "exchange_long": null,
            "country": "us"
        }
    ]
}
```

---

### Entity Type List
- **Endpoint**: `GET https://api.marketaux.com/v1/entity/type/list`
- **Purpose**: Retrieve all supported entity types (e.g., equity, index, etf, etc.). Useful for building type filters in queries.

#### Parameters
| Name        | Required | Description |
|-------------|----------|-------------|
| `api_token` | Yes      | Your API token. |

#### Response Fields
| Field   | Description |
|---------|-------------|
| `data`  | Array of entity types. |

#### Example Request
```http
GET https://api.marketaux.com/v1/entity/type/list?api_token=YOUR_API_TOKEN
```

#### Example Response
```json
{
    "data": [
        "equity",
        "index",
        "etf",
        "mutualfund",
        "currency",
        "cryptocurrency"
    ]
}
```

---

### Industry List
- **Endpoint**: `GET https://api.marketaux.com/v1/entity/industry/list`
- **Purpose**: Retrieve all supported entity industries. Useful for filtering entities or news by industry.

#### Parameters
| Name        | Required | Description |
|-------------|----------|-------------|
| `api_token` | Yes      | Your API token. |

#### Response Fields
| Field   | Description |
|---------|-------------|
| `data`  | Array of industries. |

#### Example Request
```http
GET https://api.marketaux.com/v1/entity/industry/list?api_token=YOUR_API_TOKEN
```

#### Example Response
```json
{
    "data": [
        "Technology",
        "Industrials",
        "N/A",
        "Consumer Cyclical",
        "Healthcare",
        "Communication Services",
        "Financial Services",
        "Consumer Defensive",
        "Basic Materials",
        "Real Estate",
        "Energy",
        "Utilities",
        "Financial",
        "Services",
        "Consumer Goods",
        "Industrial Goods"
    ]
}
```

---

### Sources
- **Endpoint**: `GET https://api.marketaux.com/v1/news/sources`
- **Purpose**: View available news sources for use in other API requests. Useful for filtering news by domain or source ID.
- **Limit**: 50 results per request.

#### Parameters
| Name             | Required | Description |
|------------------|----------|-------------|
| `api_token`      | Yes      | Your API token. |
| `distinct_domain`| No       | If `true`, group distinct domains (source_id will be null). |
| `language`       | No       | Comma-separated list of languages. Default: all. |
| `page`           | No       | Page number for pagination. Default: 1. |

#### Response Fields
| Field                | Description |
|----------------------|-------------|
| `meta > found`       | Number of sources found. |
| `meta > returned`    | Number of sources returned. |
| `meta > limit`       | Always 50. |
| `meta > page`        | Page number. |
| `data > source_id`   | Unique source feed ID (for use in news endpoints). |
| `data > domain`      | Domain of the source. |
| `data > language`    | Source language. |

#### Example Request
```http
GET https://api.marketaux.com/v1/news/sources?api_token=YOUR_API_TOKEN&language=en
```

#### Example Response
```json
{
    "meta": {
        "found": 5327,
        "returned": 50,
        "limit": 50,
        "page": 1
    },
    "data": [
        {
            "source_id": "adweek.com-1",
            "domain": "adweek.com",
            "language": "en"
        },
        {
            "source_id": "adage.com-1",
            "domain": "adage.com",
            "language": "en"
        },
        {
            "source_id": "avc.com-1",
            "domain": "avc.com",
            "language": "en"
        }
        // ...more sources...
    ]
}
```

---

## Request & Response Examples

**Basic Example:**
```http
GET https://api.marketaux.com/v1/news/all?api_token=YOUR_API_TOKEN&symbols=TSLA,AMZN&sentiment_gte=0.2&must_have_entities=true
```

**Retrieve articles for AAPL and TSLA, filter entities:**
```http
GET https://api.marketaux.com/v1/news/all?symbols=AAPL,TSLA&filter_entities=true&api_token=YOUR_API_TOKEN
```

**Find positive sentiment articles in English:**
```http
GET https://api.marketaux.com/v1/news/all?sentiment_gte=0.1&language=en&api_token=YOUR_API_TOKEN
```

**Find neutral sentiment articles in English:**
```http
GET https://api.marketaux.com/v1/news/all?sentiment_gte=0&sentiment_lte=0&language=en&api_token=YOUR_API_TOKEN
```

**Find negative sentiment articles in English:**
```http
GET https://api.marketaux.com/v1/news/all?sentiment_lte=-0.1&language=en&api_token=YOUR_API_TOKEN
```

**Sample JSON Response:**
```json
{
  "data": [
    {
      "uuid": "article-uuid",
      "title": "Tesla's Market Performance Surges",
      "description": "An overview of Tesla's recent stock performance...",
      "published_at": "2025-05-14T10:00:00Z",
      "url": "https://news.source.com/article",
      "entities": [
        {
          "symbol": "TSLA",
          "name": "Tesla Inc.",
          "type": "equity",
          "sentiment_score": 0.85,
          "match_score": 0.95
        }
      ]
    }
  ]
}
```

---

## Error Codes & Handling

| Error Code              | HTTP Status | Description                                                                 |
|-------------------------|-------------|-----------------------------------------------------------------------------|
| malformed_parameters    | 400         | Validation of parameters failed.                                            |
| invalid_api_token       | 401         | Invalid API token.                                                          |
| usage_limit_reached     | 402         | Usage limit of your plan has been reached. See `X-UsageLimit-Limit` header. |
| endpoint_access_restricted | 403      | Endpoint not available on your plan.                                        |
| resource_not_found      | 404         | Resource not found.                                                         |
| invalid_api_endpoint    | 404         | API route does not exist.                                                   |
| rate_limit_reached      | 429         | Too many requests in 60 seconds. See `X-RateLimit-Limit` header.            |
| server_error            | 500         | Server error occurred.                                                      |
| maintenance_mode        | 503         | Service under maintenance.                                                  |

**Error Handling Patterns:**
- Use robust exception handling for all API calls.
- Implement retry logic with exponential backoff for transient errors (see `fetch_with_retry` below):
  ```python
  import requests
  import time

  def fetch_with_retry(url, params, max_retries=3, backoff_factor=1.5):
      for attempt in range(max_retries):
          try:
              response = requests.get(url, params=params, timeout=10)
              response.raise_for_status()
              return response.json()
          except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
              if attempt == max_retries - 1:
                  # Consider logging the error here
                  raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
              time.sleep(backoff_factor ** attempt)
  ```
- For rate limiting, use a simple in-memory rate limiter (see `RateLimiter` below):
  ```python
  import time

  class RateLimiter:
      def __init__(self, calls_per_minute=60):
          self.calls_per_minute = calls_per_minute
          self.call_times = []
      def wait_if_needed(self):
          now = time.time()
          # Remove calls older than 60 seconds
          self.call_times = [t for t in self.call_times if now - t < 60]
          if len(self.call_times) >= self.calls_per_minute:
              oldest_call_in_window = self.call_times[0] # Oldest call within the current 60s window
              sleep_time = 60 - (now - oldest_call_in_window)
              if sleep_time > 0:
                  time.sleep(sleep_time)
          self.call_times.append(time.time())
  ```
- Always log errors with context and provide user-friendly messages for common scenarios.

---

## Rate Limiting & Usage Limits

- **Free Plan Limits:**
  - 100 requests per day
  - 3 articles per request (global market news)
  - No access to Market Stats API
  - Full metadata access
  - 200,000+ entities, 5,000+ sources, 80+ markets, 30+ languages
- **Headers:**
  - `X-RateLimit-Limit`: Max requests per 60 seconds
  - `X-UsageLimit-Limit`: Max requests per day
- **Best Practices:**
  - Monitor these headers to avoid hitting limits
  - Implement graceful degradation if limits are reached
  - Consider caching results to minimize redundant calls

---

## Free Plan Capabilities

- ✓ 100 requests daily
- ✓ 3 articles per request (global market news)
- ✓ Instant news access
- ✓ Full metadata access
- ✓ 200,000+ entities
- ✓ 5,000+ news sources
- ✓ 80+ global markets
- ✓ 30+ languages
- ✗ Market Stats API access
- ✗ Technical support

---

## Integration Patterns

**Typical Workflow (for secondary uses, not RAG pipeline):**
1. Fetch article snippets from Marketaux using `/v1/news/all` with appropriate filters if a quick overview or sentiment check is needed.
2. Process the snippet and associated metadata (e.g., entities, sentiment scores).
3. Use this information for tasks like dashboarding sentiment trends or identifying articles for later full-text retrieval via another source if necessary.
4. Implement robust error and rate limit handling as described.

**Note:** For the primary RAG pipeline, refer to the `eodhd_api.md` documentation for fetching and processing full articles.

**Python Example (for fetching snippets):**
```python
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_TOKEN = os.getenv('MARKETAUX_API_KEY')
API_URL = 'https://api.marketaux.com/v1/news/all'

def fetch_marketaux_news_snippets(symbols, sentiment_threshold=0.2):
    if not API_TOKEN:
        print("MARKETAUX_API_KEY not found in environment variables.")
        return []
    params = {
        'api_token': API_TOKEN,
        'symbols': ','.join(symbols),
        'sentiment_gte': sentiment_threshold,
        'must_have_entities': 'true',
        'limit': 3 # Example limit, adjust as needed per plan
    }
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Marketaux snippets: {e}")
        return []

# Example usage:
# snippets = fetch_marketaux_news_snippets(['TSLA', 'AAPL'])
# for snippet_data in snippets:
#     print(f"Title: {snippet_data['title']}, Snippet: {snippet_data['snippet']}")
```

---

## References
- [Marketaux API Documentation](https://www.marketaux.com/documentation)
- [EODHD API Integration Guide](./eodhd_api.md) (for primary RAG news source)
