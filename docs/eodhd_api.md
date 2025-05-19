# EODHD Financial News API Integration Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
    - [Financial News](#financial-news)
4. [Supported Topic Tags](#supported-topic-tags)
5. [Request & Response Examples](#request--response-examples)
6. [Error Codes & Handling](#error-codes--handling)
7. [Rate Limiting & Usage Limits](#rate-limiting--usage-limits)
8. [Free Plan Capabilities](#free-plan-capabilities)
9. [Integration Patterns](#integration-patterns)
    - [API Call Strategy](#api-call-strategy)
    - [Database for Tracking API Usage](#database-for-tracking-api-usage)
10. [References](#references)

---

## Overview
The EODHD Financial News API provides access to global financial news articles, including full article content, which is crucial for our RAG (Retrieval Augmented Generation) system. This API will replace the Marketaux API as the primary source for populating our vector database with news articles, as Marketaux only provides snippets. EODHD supports filtering by ticker symbol, topic tag, date, and offers pagination.

---

## Authentication
- **API Token Required**: All requests to the EODHD API require an `api_token` as a query parameter.
- **How to Obtain**: Sign up for an account on the [EODHD website](https://eodhd.com/) and retrieve your API token from your account dashboard.
- **Usage Example**: `GET https://eodhd.com/api/news?...&api_token=YOUR_API_TOKEN`
- **Best Practice**: Store your API token in a `.env` file and load it into your application using a library like `python-dotenv`. This practice is aligned with our project's [technical_design.md#configuration-management](technical_design.md#configuration-management).

---

## API Endpoints

### Financial News
- **Endpoint**: `GET https://eodhd.com/api/news`
- **Purpose**: Retrieve the latest global financial news, filterable by ticker symbols, topic tags, and date ranges. This endpoint provides full article content, making it suitable for our RAG pipeline.

#### HTTP GET Parameters
| Parameter  | Required           | Type    | Description                                                                 |
|------------|--------------------|---------|-----------------------------------------------------------------------------|
| `s`        | Yes (if `t` not set) | string  | The ticker code to retrieve news for, e.g., `AAPL.US`. **Important**: Despite official documentation, only a single symbol is supported. Multiple symbols in a comma-separated list will result in empty responses. |
| `t`        | Yes (if `s` not set) | string  | The topic tag to retrieve news for, e.g., `technology`. Use `%20` for spaces in multi-word tags (e.g., `mergers%20and%20acquisitions`). |
| `from`     | No                 | string  | Start date for filtering news (YYYY-MM-DD).                                 |
| `to`       | No                 | string  | End date for filtering news (YYYY-MM-DD).                                   |
| `limit`    | No                 | integer | Number of results to return (default: 50, min: 1, max: 1000). Note: The API may not always return the exact number specified. |
| `offset`   | No                 | integer | Offset for pagination (default: 0).                                         |
| `fmt`      | No                 | string  | Response format: `json` or `xml` (default: `json`).                         |
| `api_token`| Yes                | string  | Your unique API access token.                                               |

#### Pagination
- Use the `limit` and `offset` parameters to paginate through results.
- For example, to get the second page of 10 articles: `limit=10&offset=10`.

#### Response Fields (JSON)
Each article in the JSON response array includes:
| Field     | Type                | Description                                      |
|-----------|---------------------|--------------------------------------------------|
| `date`    | string (ISO 8601)   | Publication date and time of the article.         |
| `title`   | string              | Headline of the news article.                    |
| `content` | string              | Full article body.                               |
| `link`    | string              | Direct URL to the article.                       |
| `symbols` | array               | List of ticker symbols mentioned in the article. **May be empty.** |
| `tags`    | array               | Article topic tags. **May be empty.** (max 20 shown, alphabetically sorted). |
| `sentiment`| object             | Sentiment scores: `polarity`, `neg`, `neu`, `pos`.|

Our `EODHDClient` processes the raw API response and returns a list of dictionaries, where each dictionary represents an article with the following structure. Note that `url_hash`, `fetched_at`, `source_api` are added by our client, and `published_at` is a normalized version of the original `date` field.

| Field          | Type                | Description                                                                                                                               |
|----------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `url_hash`     | string              | A SHA-256 hash of the article's URL. This can be used as a unique identifier or primary key. **(Custom field)**                             |
| `title`        | string              | Headline of the news article.                                                                                                             |
| `content`      | string              | Full article body.                                                                                                                        |
| `url`          | string              | Direct URL to the article.                                                                                                                |
| `published_at` | string (ISO 8601)   | Normalized publication date and time of the article. Original `date` field is parsed and stored in a consistent ISO 8601 format.           |
| `fetched_at`   | string (ISO 8601)   | Timestamp of when the article was fetched by our client. **(Custom field)**                                                                 |
| `source_api`   | string              | Indicates the API source, e.g., 'EODHD'. **(Custom field)**                                                                               |
| `symbols`      | array               | List of ticker symbols mentioned in the article. May be empty.                                                                            |
| `tags`         | array               | Article topic tags. May be empty.                                                                                                         |
| `sentiment`    | object              | Sentiment scores: `polarity`, `neg`, `neu`, `pos`.                                                                                        |

**Note on `published_at` normalization:**
The `date` field from the EODHD API response is normalized into the `published_at` field. The normalization process involves:
```python
# Pseudo-code for normalization
try:
    # Parse the ISO 8601 date
    published_at = datetime.fromisoformat(article['date'].replace('Z', '+00:00'))
    # Store it in ISO format for consistent representation
    published_at_iso = published_at.isoformat()
except (KeyError, ValueError):
    # Fallback if date is missing or invalid
    published_at_iso = datetime.now(timezone.utc).isoformat()
```

#### Example Request
```http
GET https://eodhd.com/api/news?t=mergers%20and%20acquisitions&limit=5&from=2025-05-01&to=2025-05-18&api_token=YOUR_API_TOKEN&fmt=json
```

#### Example Response
```json
[
    {
        "date": "2025-05-15T12:28:10+00:00",
        "title": "Tech Giant A Acquires Startup B in Multi-Billion Dollar Deal",
        "content": "Full article text detailing the acquisition, synergies, and market impact will be present here...",
        "link": "https://www.example-news.com/tech-giant-a-acquires-startup-b",
        "symbols": ["TICK.US", "STRT.US"],
        "tags": [
            "ACQUISITION",
            "MERGER",
            "TECHNOLOGY"
        ],
        "sentiment": {"polarity": 0.65, "neg": 0.05, "neu": 0.80, "pos": 0.15}
    }
    // ... more articles ...
]
```

---

## Supported Topic Tags
The EODHD API allows filtering news by topic tags using the `t` parameter. While the official documentation lists some tags, exploration has shown that many more are active. It's recommended to test tags relevant to your needs. Use `%20` to encode spaces in multi-word tags.

**Known/Documented Tags (partial list):**
`'balance sheet', 'capital employed', 'class action', 'company announcement', 'consensus eps estimate', 'consensus estimate', 'credit rating', 'discounted cash flow', 'dividend payments', 'earnings estimate', 'earnings growth', 'earnings per share', 'earnings release', 'earnings report', 'earnings results', 'earnings surprise', 'estimate revisions', 'european regulatory news', 'financial results', 'fourth quarter', 'free cash flow', 'future cash flows', 'growth rate', 'initial public offering', 'insider ownership', 'insider transactions', 'institutional investors', 'institutional ownership', 'intrinsic value', 'market research reports', 'net income', 'operating income', 'present value', 'press releases', 'price target', 'quarterly earnings', 'quarterly results', 'ratings', 'research analysis and reports', 'return on equity', 'revenue estimates', 'revenue growth', 'roce', 'roe', 'share price', 'shareholder rights', 'shareholder', 'shares outstanding', 'split', 'strong buy', 'total revenue', 'zacks investment research', 'zacks rank'`

**Note:** The `tags` field in the API response is limited to the first 20 tags, sorted alphabetically. The documented list of supported topic tags appears to be a small subset and some may be outdated.

---

## Request & Response Examples

**Fetching news for a specific ticker:**
```http
GET https://eodhd.com/api/news?s=AAPL.US&limit=3&api_token=YOUR_API_TOKEN&fmt=json
```

**Important Note on Symbol Queries:**
Despite what the official EODHD documentation suggests, our testing has revealed that:
1. Only a single symbol is accepted in the `s` parameter
2. Using comma-separated symbols (e.g., `s=AAPL.US,MSFT.US`) results in empty responses
3. If you need to fetch news for multiple symbols, you must make separate API calls for each symbol

**Fetching news for a topic tag with date range:**
```http
GET https://eodhd.com/api/news?t=earnings&from=2025-05-01&to=2025-05-18&limit=10&api_token=YOUR_API_TOKEN&fmt=json
```

---

## Error Codes & Handling
Official documentation on EODHD API error codes is sparse. Based on common API practices and observations:
- Expect standard HTTP status codes (e.g., 400 for bad requests, 401 for authentication issues, 403 for permission issues, 429 for rate limits, 5xx for server errors).
- Implement robust exception handling for all API calls.
- A retry mechanism with exponential backoff is recommended for transient errors (e.g., network issues, temporary server errors).

```python
import requests
import time

def fetch_eodhd_with_retry(url, params, max_retries=3, backoff_factor=1.5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=25) # Increased timeout due to potential delays
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            print(f"Attempt {attempt + 1} failed: {e}") # Basic logging
            if attempt == max_retries - 1:
                # Consider more detailed logging here
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
            time.sleep(backoff_factor ** attempt)
    return None # Should not be reached if max_retries > 0
```
- Response time for requests can sometimes take up to 60 seconds, so ensure your HTTP client timeout is configured appropriately.

---

## Rate Limiting & Usage Limits

- **API Call Cost**: Each request to the `/api/news` endpoint consumes **5 API calls** from your EODHD plan allowance.
- **Free Plan Limits (as of exploration):**
    - **Daily API Calls**: 20 API calls. This effectively means **4 news requests per day** (20 calls / 5 calls per request).
    - **Minute Request Limit**: The API is generally restricted to 1000 requests per minute (across all users/plans, not specific to the free tier's daily limit). This is indicated by headers like `X-RateLimit-Limit` and `X-RateLimit-Remaining`.
- **Best Practices:**
    - Given the low daily limit on the free plan, space out your 4 news requests. For example, one request every 15 seconds within a minute, once per day.
    - Monitor API call usage if you upgrade to a paid plan.
    - Cache results where appropriate to avoid redundant calls, although freshness is key for our RAG.
    - The `limit` parameter can request up to 1000 articles, but be mindful of processing time.

---

## Free Plan Capabilities
- **API Calls per Day:** 20 (translates to 4 news requests)
- **Data Range:** Typically the past year of news.
- **Content:** Full article content.
- **Usage:** Intended for personal use.
- **Sources:** EODHD claims to aggregate news from 10+ sources and 40+ media outlets (unverified).
- Articles are available in multiple languages, but most are in English.

---

## Integration Patterns

**Typical Workflow:**
1.  Construct your API request to `https://eodhd.com/api/news` using desired ticker symbols (`s`) or topic tags (`t`), date filters (`from`, `to`), and pagination (`limit`, `offset`).
2.  Fetch articles from EODHD.
3.  Before processing, check against a local database (see [Database for Tracking API Usage](#database-for-tracking-api-usage)) of article URLs to prevent duplicate processing and embedding.
4.  Clean and process the `content` of new articles.
5.  Generate embeddings for the processed content.
6.  Store the article metadata (title, link, date, symbols, tags, sentiment) and embeddings in your vector database (e.g., ChromaDB).
7.  Update your local tracking database with the URLs of newly processed articles and any relevant metadata for API call strategy.
8.  Adhere to rate limits.

For more details on the `EODHDClient` class and its methods, refer to the `src/financial_news_rag/eodhd.py` file.

**Python Examples using `EODHDClient`:**

The following examples demonstrate how to use the `EODHDClient` from our project to fetch news.

```python
# Imports
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from financial_news_rag.eodhd import EODHDClient # Assuming EODHDClient is in this path

# Load environment variables
load_dotenv() # Make sure your .env file has EODHD_API_KEY

# Create an EODHD client
# The client will automatically pick up the API key from the environment variable EODHD_API_KEY
client = EODHDClient()

# Define date range for the last 7 days
today = datetime.now()
week_ago = today - timedelta(days=7)
today_str = today.strftime("%Y-%m-%d")
week_ago_str = week_ago.strftime("%Y-%m-%d")

# Example 1: Fetch news by tag (Mergers and Acquisitions)
# This will fetch up to 10 articles tagged with "MERGERS AND ACQUISITIONS"
# published between week_ago_str and today_str.
merger_news = client.fetch_news(
    tag="MERGERS AND ACQUISITIONS",
    from_date=week_ago_str,
    to_date=today_str,
    limit=10
)

# Example merger_news output (structure):
# [
#     {
#         'url_hash': '458b3e5a901a99886feef68be6ef3f3e55381c1cdf6a71e8984ec3e8949275a3',
#         'title': 'Article title',
#         'content': 'Full article content here...',
#         'url': 'https://example.com/news/article.html',
#         'published_at': '2025-05-19T13:37:00+00:00', # Normalized date
#         'fetched_at': '2025-05-19T17:09:17.540581+00:00', # Custom field
#         'source_api': 'EODHD', # Custom field
#         'symbols': [],
#         'tags': ['EUROPEAN REGULATORY NEWS', 'MERGERS AND ACQUISITIONS'],
#         'sentiment': {'polarity': 0.997, 'neg': 0.005, 'neu': 0.887, 'pos': 0.107}
#     }
#     # ... more articles ...
# ]

if merger_news:
    print(f"Fetched {len(merger_news)} articles for 'MERGERS AND ACQUISITIONS'")
    # print(merger_news[0]) # To inspect the first article
else:
    print("No articles found for 'MERGERS AND ACQUISITIONS' in the last 7 days.")

# Example 2: Fetch news for a specific symbol (Apple)
# This will fetch up to 10 articles related to "AAPL.US"
# published between week_ago_str and today_str.
apple_news = client.fetch_news(
    symbols="AAPL.US", 
    from_date=week_ago_str,
    to_date=today_str,
    limit=10
)

# Example apple_news output (structure):
# [
#     {
#         'url_hash': '458b3e5a901a99886feef68be6ef3f3e55381c1cdf6a71e8984ec3e8949275a3',
#         'title': 'Article title',
#         'content': "Full article content here...",
#         'url': 'https://example.com/news/article.html',
#         'published_at': '2025-05-19T12:46:30+00:00', # Normalized date
#         'fetched_at': '2025-05-19T17:30:10.329923+00:00', # Custom field
#         'source_api': 'EODHD', # Custom field
#         'symbols': ['AAPL.US', 'BABA.US', 'BABAF.US'],
#         'tags': [],
#         'sentiment': {'polarity': -0.881, 'neg': 0.097, 'neu': 0.863, 'pos': 0.04}
#     }
#     # ... more articles ...
# ]

if apple_news:
    print(f"Fetched {len(apple_news)} articles for 'AAPL.US'")
    # print(apple_news[0]) # To inspect the first article
else:
    print("No articles found for 'AAPL.US' in the last 7 days.")

```

**Explanation of Custom Fields in `EODHDClient` Response:**
When using our `EODHDClient`, the response for each article includes a few additional fields not present in the raw EODHD API response:
-   `source_api`: A string field (e.g., "EODHD") to identify the origin of the data, useful if integrating multiple news sources.
-   `fetched_at`: An ISO 8601 timestamp indicating when our system retrieved the article.
-   `url_hash`: A SHA-256 hash of the article's `url`. Since the `url` is the only guaranteed unique identifier from the EODHD API, this hash can serve as a reliable primary key for database storage or for quick comparisons to avoid processing duplicate articles.

These additions aid in data management, tracking, and ensuring data integrity within our RAG system.

### API Call Strategy
(Adapted from exploration notes)

Balancing information breadth and depth with API call limitations is key. The goal is fresh, forward-looking, high-signal, low-overlap news for investment opportunities.

**How far back to fetch news?**
*   **Core relevance (for RAG):** 1-3 months. This captures recent earnings, product launches, market shifts, and deal announcements that are highly actionable.
*   **Contextual relevance:** Up to 6-12 months for understanding ongoing M&A sagas, venture capital funding rounds progression, or the development of major trends.

Given the free plan's limitation of 4 news requests per day, a highly focused strategy is necessary.

**Initial Tag Strategy (Example for very limited calls):**
Focus on the most impactful tags, rotating them daily or every few days.
*   **Priority 1 (High Signal):**
    *   `MERGERS AND ACQUISITIONS`
    *   `VENTURE CAPITAL`
    *   `EARNINGS RELEASES AND OPERATING RESULTS` (or `EARNINGS REPORT`)
    *   `PRODUCT / SERVICES ANNOUNCEMENT` (or `COMPANY ANNOUNCEMENT` and filter)
*   **Priority 2 (Broader Tech/Innovation - rotate these in):**
    *   `AI` (or `ARTIFICIAL INTELLIGENCE`)
    *   `FINTECH`
    *   `BIOTECH`
    *   `IPO`

With only 4 requests, you might pick one tag from Priority 1 and fetch news for the last 1-7 days, aiming for a `limit` that maximizes fresh articles without excessive overlap if run daily for the same tag. The `from` and `to` parameters will be critical.

**Optimal Rotation Strategy (Post-Initial Phase / Paid Plan):**
If on a paid plan with more calls, the previously outlined 7-day strategy and subsequent weekly rotations become more feasible.
*   **Anchor Tags (search these frequently):** `MERGERS AND ACQUISITIONS`, `EARNINGS RELEASES AND OPERATING RESULTS`. Alternate with `VENTURE CAPITAL` or `PRODUCT / SERVICES ANNOUNCEMENT`.
*   **Thematic Rotation (rotate these weekly/bi-weekly):** Cover Tech, Funding, Sector/Macro, Company Specifics as detailed in the original exploration notes.

### Database for Tracking API Usage
To optimize API calls and avoid redundant data processing, a separate database (SQLite) is recommended to track:
*   **`articles` table:**
    *   `url` (PRIMARY KEY, TEXT): The unique URL of the article. Ensures each article is processed once.
    *   `title` (TEXT)
    *   `published_date` (TIMESTAMP)
    *   `fetched_date` (TIMESTAMP): When we fetched it.
    *   `source_tag` (TEXT): The tag used to find this article.
    *   `processed_for_rag` (BOOLEAN, default FALSE)
*   **`api_tags_log` table:**
    *   `tag_name` (TEXT)
    *   `last_fetched_date` (TIMESTAMP)
    *   `oldest_article_date_retrieved` (DATE): To understand coverage for a tag.
    *   `newest_article_date_retrieved` (DATE)
    *   `articles_retrieved_count` (INTEGER)
    *   Purpose: Track information about mean time between articles for specific tags to find out if specific tags require different `offset` or date range strategies to fill the minimum requirement of core relevance (1 month) for RAG.
*   **`api_errors_log` table:**
    *   `timestamp` (TIMESTAMP)
    *   `request_url` (TEXT)
    *   `status_code` (INTEGER, nullable)
    *   `error_message` (TEXT, nullable)
    *   `response_text` (TEXT, nullable)
    *   Purpose: To build knowledge about possible errors and their codes due to non-existent official documentation.

This database will help in:
1.  Ensuring article uniqueness before RAG processing.
2.  Refining the API call strategy by understanding which tags yield fresh content and how far back one needs to query.
3.  Logging and understanding API errors.

---

## References
- [EODHD Financial News API Documentation](https://eodhd.com/financial-apis/stock-market-financial-news-api)
- [Project Technical Design: Configuration Management](./technical_design.md#configuration-management) (for API key handling)
- [EODHDClient source code](../src/financial_news_rag/eodhd.py)

---
