# ReRanker (`reranker.py`)

The `reranker.py` module provides the `ReRanker` class, which is designed to re-rank a list of financial news articles based on their relevance to a given user query. It uses a Gemini Large Language Model (LLM) to assess the relevance of each article.

## `ReRanker` Class

This class encapsulates the logic for interacting with the Gemini API to get relevance scores and then re-sorts the articles.

### Initialization

```python
class ReRanker:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the ReRanker with API key and model name.
        ...
        """
        # ...
```

The constructor sets up the `ReRanker`.

**Parameters:**

*   `api_key` (str): Your Gemini API key. This is **required**.
*   `model_name` (str, optional): The specific Gemini model to be used for relevance assessment. Defaults to `"gemini-2.0-flash"`.

**Raises:**

*   `ValueError`: If the `api_key` is not provided.

### Methods

#### `rerank_articles(query: str, articles: List[Dict]) -> List[Dict]`

```python
def rerank_articles(self, query: str, articles: List[Dict]) -> List[Dict]:
    """
    Re-rank a list of articles based on their relevance to a query.
    ...
    Returns:
        A list of article dictionaries, sorted by relevance score in descending order,
        with an additional 'rerank_score' field
    """
    # ...
```
Takes a user query and a list of articles (retrieved from a primary search, e.g., vector search) and re-orders them based on relevance scores obtained from the Gemini LLM.

**Parameters:**

*   `query` (str): The original user query string.
*   `articles` (List[Dict]): A list of article dictionaries. Each dictionary is expected to have at least the following keys:
    *   `"url_hash"` (str): A unique identifier for the article.
    *   `"processed_content"` (str): The text content of the article that will be assessed for relevance.
    *   It can contain other fields, which will be preserved in the output.

**Returns:**

*   `List[Dict]`: A new list of article dictionaries, sorted in descending order based on the `"rerank_score"`. Each article dictionary in the returned list will have an additional key `"rerank_score"` (a float between 0.0 and 10.0) representing its assessed relevance to the query. If an error occurs during the re-ranking process for any article or globally, the original list of articles might be returned, or articles might have a default score of 0.0.

**Internal Relevance Assessment (`_assess_article_relevance`):**

This private method is called by `rerank_articles` for each article.

*   **Prompting:** It constructs a prompt for the Gemini model, providing system instructions to act as a financial analyst assistant. The prompt asks the model to rate the article's relevance to the user query on a scale of 0-10 and return the result as a JSON object `{"id": "article-id", "score": relevance_score}`.
*   **Content Truncation:** If an article's `processed_content` is longer than 10,000 characters, it is truncated to prevent overly long inputs to the LLM.
*   **API Call:** It calls the Gemini API (`generate_content`) with the constructed prompt and specific configuration (e.g., `max_output_tokens`, `temperature`).
*   **Response Parsing:** It attempts to parse the LLM's response as JSON. If that fails, it tries to extract the score using a regular expression as a fallback.
*   **Error Handling & Retries:** The `_assess_article_relevance` method is decorated with `@tenacity.retry`:
    *   Retries on `GoogleAPIError`, `ServiceUnavailable`, `ConnectionError`.
    *   Stops after 3 attempts.
    *   Uses exponential backoff (multiplier 1, min 2s, max 30s).
    *   If an article's content is empty or an unrecoverable error occurs during its assessment, it defaults to a score of 0.0 for that article.

If the `articles` list is empty or the `query` is empty, the method returns the original articles or an empty list appropriately, with a warning log.
