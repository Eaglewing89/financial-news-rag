# EODHD Financial News API Documentation

## Overview
The EODHD Financial News API provides access to the latest financial news headlines and full articles for a given ticker symbol or topic tag. It supports filtering by date, pagination, and returns results in JSON or XML format.

- **Endpoint:**
  - `GET https://eodhd.com/api/news`


## Parameters
| Parameter  | Required           | Type    | Description                                                                 |
|------------|--------------------|---------|-----------------------------------------------------------------------------|
| s          | Yes (if t not set) | string  | The ticker code to retrieve news for, e.g., `AAPL.US`                       |
| t          | Yes (if s not set) | string  | The topic tag to retrieve news for, e.g., `technology`                      |
| from       | No                 | string  | Start date for filtering news (YYYY-MM-DD)                                  |
| to         | No                 | string  | End date for filtering news (YYYY-MM-DD)                                    |
| limit      | No                 | integer | Number of results to return (default: 50, min: 1, max: 1000)                |
| offset     | No                 | integer | Offset for pagination (default: 0)                                          |
| fmt        | No                 | string  | Response format: `json` or `xml` (default: `json`)                          |
| api_token  | Yes                | string  | Your unique API access token                                                |

## Example Usage
```python
import requests

url = f'https://eodhd.com/api/news?s=AAPL.US&offset=0&limit=10&api_token=your_api_token&fmt=json'
data = requests.get(url).json()
print(data)
```

## Example Request and Response
**Request:**
```
https://eodhd.com/api/news?t=earnings&offset=0&limit=200&api_token={eodhd_api_key}&fmt=json
```

**Sample Response:**
```python
[
    {
        'date': '2025-04-15T12:28:10+00:00',
        'title': "Cboe's volatility expert sees more market turmoil ahead as tariff uncertainty not fully priced in",
        'content': 'Full article text here...',
        'link': 'https://www.cnbc.com/2025/04/15/tariff-uncertainty-not-fully-priced-into-stocks-cboes-mandy-xu.html',
        'symbols': ['CBOE.US', 'DXY.US', 'SPX.US', 'TLT.US', 'US10Y.US', 'VIX.US'],
        'tags': [
            'ANNA GLEASON', 'BREAKING NEWS E', 'BREAKING NEWS ECONOMY', ... , 'EARNINGS', 'ECONOMY'
        ],
        'sentiment': {'polarity': 0.324, 'neg': 0.065, 'neu': 0.862, 'pos': 0.073}
    },
    ...
]
```

## Output Format (JSON)
Each article includes:

| Field     | Type                | Description                                      |
|-----------|---------------------|--------------------------------------------------|
| date      | string (ISO 8601)   | Publication date and time of the article          |
| title     | string              | Headline of the news article                     |
| content   | string              | Full article body                                |
| link      | string              | Direct URL to the article                        |
| symbols   | array               | List of ticker symbols mentioned                  |
| tags      | array               | Article topic tags (may be empty, max 20 shown)   |
| sentiment | object              | Sentiment scores: polarity, neg, neu, pos         |

## Supported Topic Tags
You can use the following tags (parameter `t`) to get news for a given topic (50+ available):

`'balance sheet', 'capital employed', 'class action', 'company announcement', 'consensus eps estimate', 'consensus estimate', 'credit rating', 'discounted cash flow', 'dividend payments', 'earnings estimate', 'earnings growth', 'earnings per share', 'earnings release', 'earnings report', 'earnings results', 'earnings surprise', 'estimate revisions', 'european regulatory news', 'financial results', 'fourth quarter', 'free cash flow', 'future cash flows', 'growth rate', 'initial public offering', 'insider ownership', 'insider transactions', 'institutional investors', 'institutional ownership', 'intrinsic value', 'market research reports', 'net income', 'operating income', 'present value', 'press releases', 'price target', 'quarterly earnings', 'quarterly results', 'ratings', 'research analysis and reports', 'return on equity', 'revenue estimates', 'revenue growth', 'roce', 'roe', 'share price', 'shareholder rights', 'shareholder', 'shares outstanding', 'split', 'strong buy', 'total revenue', 'zacks investment research', 'zacks rank'`

## Notes from Exploration
- The API may not always return the exact number of articles specified by the `limit` parameter.
- The `tags` field in the response is limited to the first 20 tags in alphabetical order.
- Articles are available in multiple languages, but most are in English.
- EODHD claims to aggregate news from 10+ sources and 40+ media outlets (unverified).
- Each API request to this endpoint consumes **5 API calls**.
- Use %20 for white spaces in the request. For example 'mergers and acquisistions' would be mergers%20and%20acquisitions
- The supported topic tags from documentation is a very small subset and some appear to be outdated.
- Response time for request can take up to 20 seconds. Usually faster. 


### api call strategy suggestion from exploration
Balancing information breadth and depth with API call limitations. The goal is fresh, forward-looking, high-signal, low-overlap news for investment opportunities.

**How far back to fetch news?**

For our purpose of identifying current investment opportunities, news relevance decays quickly.
*   **Core relevance (for RAG):** 1-3 months. This captures recent earnings, product launches, market shifts, and deal announcements that are highly actionable.
*   **Contextual relevance:** Up to 6-12 months for understanding ongoing M&A sagas, venture capital funding rounds progression, or the development of major trends (e.g., AI adoption in a sector).


**Initial 7-Day Tag Strategy (4 tags per day):**

The aim here is to get a diverse initial set of high-signal, forward-looking articles, minimizing overlap between the *primary intent* of the daily searches.

*   **Day 1: Capital & Growth Focus**
    1.  `VENTURE CAPITAL`
    2.  `IPO`
    3.  `FINANCING AGREEMENTS`
    4.  `PRIVATE EQUITY`
*   **Day 2: Corporate Development & Future Products**
    1.  `MERGERS AND ACQUISITIONS`
    2.  `PRODUCT / SERVICES ANNOUNCEMENT`
    3.  `PARTNERSHIPS`
    4.  `COMPANY RESTRUCTURING` (If not a common tag, use `COMPANY ANNOUNCEMENT` and filter, or `LEADERSHIP` changes which often accompany restructuring)
*   **Day 3: Performance & Shareholder Value**
    1.  `EARNINGS RELEASES AND OPERATING RESULTS`
    2.  `CONFERENCE CALLS/ WEBCASTS`
    3.  `STOCK BUYBACKS` (If available and distinct, otherwise `DIVIDENDS`)
    4.  `MAJOR SHAREHOLDER ANNOUNCEMENTS`
*   **Day 4: Technology & Innovation**
    1.  `AI` (or `ARTIFICIAL INTELLIGENCE` - pick one consistently)
    2.  `INNOVATION`
    3.  `PATENTS`
    4.  `CYBERSECURITY`
*   **Day 5: Key Sectors & Emerging Trends**
    1.  `FINTECH`
    2.  `BIOTECH`
    3.  `SUSTAINABILITY` (or `ESG` if that's a tag)
    4.  `BLOCKCHAIN`
*   **Day 6: Market Dynamics & Governance**
    1.  `MARKET INSIDER` (If it refers to significant transactions/filings) or `REGULATORY NEWS`
    2.  `ECONOMIC RESEARCH AND REPORTS`
    3.  `LEADERSHIP` (e.g., CEO/CFO changes)
    4.  `NEW CONTRACTS` (If available, otherwise `BUSINESS` and hope for good secondary tags)
*   **Day 7: New Ventures & Analysis**
    1.  `STARTUPS`
    2.  `SEED FUNDING` (More specific than just funding)
    3.  `RESEARCH ANALYSIS AND REPORTS` (e.g., analyst upgrades/downgrades, market outlooks)
    4.  `TECHNOLOGY` (As a broader catch-all for tech news not covered by AI/Cybersecurity)

**Optimal Rotation Strategy (Post-Initial Week):**

After the initial week, you'll have a foundational dataset. The rotation should maintain freshness in key areas and explore others.

1.  **Anchor Tags (2 per day - search these almost daily or every other day):**
    *   `MERGERS AND ACQUISITIONS`
    *   `EARNINGS RELEASES AND OPERATING RESULTS`
    *   Alternate one of these with `VENTURE CAPITAL` or `PRODUCT / SERVICES ANNOUNCEMENT` to ensure they are hit 3-4 times a week.

2.  **Thematic Rotation (2 per day - rotate these weekly/bi-weekly):**
    *   **Week A: Tech Deep Dive & Future**
        *   `AI`
        *   `INNOVATION` / `PATENTS` (alternate)
        *   `CYBERSECURITY`
        *   `FINTECH`
    *   **Week B: Funding & Growth Stages**
        *   `IPO`
        *   `STARTUPS` / `SEED FUNDING` (alternate)
        *   `PRIVATE EQUITY`
        *   `PARTNERSHIPS`
    *   **Week C: Sector & Macro Focus**
        *   `BIOTECH` (or another sector like `ENERGY`, `RETAIL` depending on your interest)
        *   `SUSTAINABILITY`
        *   `REGULATORY NEWS` (or `EUROPEAN REGULATORY NEWS`)
        *   `ECONOMIC RESEARCH AND REPORTS`
    *   **Week D: Company Specifics & Market Signals**
        *   `CONFERENCE CALLS/ WEBCASTS`
        *   `LEADERSHIP`
        *   `MAJOR SHAREHOLDER ANNOUNCEMENTS`
        *   `DIVIDENDS`

**Followup plan**
We need a database separate from our vector database to keep track of the following information:  
- urls table to keep track of unique articles so we do not run duplicate articles through our text processing into embedding pipeline. Make sure this search is performant. 
- tags table to track information about mean time between articles to find out if specific tags require offset parameters to fill the minimum requirement of core relevance for RAG which is 1 month. 
- errors table to build knowledge about possible errors and their codes due to non-existant documentation about them. 
- maybe something more? 


## Free Plan Information
- **API Calls per Day:** 20/day
- **API Requests per Minute:** 20/day
- **Data Range:** Past year
- **Type of Usage:** Personal use


## API error codes
There appears to be absolutely no information about error codes in the documentation. 

## Rate limiting

Minute Limit (requests)
Minute request limit (not API calls) means that API is restricted to no more than 1000 requests per minute. It’s easy to check this limit with headers you get with every request:

X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 998
It is advisable to spread out the requests more or less evenly throughout the minute, without making them go off all at once, to avoid getting a “Too Many Requests” error.

Note that in our case we are restricted to only 20 requests per day, but as each news requests consumes 5 api calls means that we actually only make a total of 4 requests each day. We will simply space these 4 requests out over a single minute and we'll be good. 

## Integration Patterns

**Typical Workflow:**
1. Fetch articles from EODHD using `https://eodhd.com/api/news` with appropriate date filters and search by specified tag.
2. Search our EODHD database to filter out already stored articles based on urls from the response. 
3. Clean and process article text.
4. Store articles and metadata in our vector database (e.g., ChromaDB).
5. Update our EODHD database as described above. 
6. Use basic rate limit handling as described above.

---

## References
- [EODHD API Documentation](https://eodhd.com/financial-apis/stock-market-financial-news-api)

---
