# Document Text Processing Pipeline

## Description
This document details the text processing pipeline for the Financial News RAG system. It covers text cleaning, chunking, preprocessing for embedding, and tokenization strategies. The pipeline is implemented in the `TextProcessingPipeline` class found in `src/financial_news_rag/text_processor.py`. An example of its usage can be found in `examples/text_processing_example.py`.

The primary source of text for this pipeline is the **full article content** obtained from APIs like EODHD and initially stored in our **SQLite database** as `raw_content`. The processed text is also stored back in SQLite as `processed_content` before being used for embedding.

---

## 1. Text Cleaning Approaches

### 1.1 Cleaning and Normalization
- **Input:** The `raw_content` field from the `articles` table in the SQLite database.
- **Implementation:** The `TextProcessingPipeline.clean_article_text()` method in `src/financial_news_rag/text_processor.py` handles this.
- **Steps:**
    - HTML Removal: All article `raw_content` is stripped of HTML tags.
    - Whitespace Normalization: Consecutive whitespace is collapsed to a single space and leading/trailing whitespace is trimmed.
    - Boilerplate Removal: Common phrases (e.g., “Click here to read more.”) are removed using regex.
    - Encoding Fixes: Known encoding issues (e.g., smart quotes, dashes) are replaced with standard ASCII equivalents.
    - Unicode Normalization: All text is normalized to NFC form to ensure consistent representation.
- **Output:** Cleaned text, which will be stored in the `processed_content` field in the SQLite `articles` table.

**Reference Implementation Snippet (from `src/financial_news_rag/text_processor.py`):**
```python
# ... (within TextProcessingPipeline class)
def clean_article_text(self, raw_text: str) -> str:
    # ...
    # Remove HTML tags
    text = re.sub(r\'<.*?>\', \'\', text)
    # Normalize whitespace
    text = re.sub(r\'\\s+\', \' \', text).strip()
    # Remove common boilerplate phrases
    # ... (boilerplate_patterns list and loop)
    # Fix encoding issues
    # ...
    # Normalize unicode to NFC form
    text = unicodedata.normalize(\'NFC\', text)
    # ...
    return text
```

---

## 2. Chunking Strategies for RAG

### 2.1 Model Constraints
- **Embedding Model:** Google’s `text-embedding-004` (see `model_details.md`)
- **Token Limit:** Configurable, defaults to 2,048 tokens per chunk (see `TextProcessingPipeline.__init__` and `split_into_chunks` in `src/financial_news_rag/text_processor.py`). (1 token ≈ 4 characters)

### 2.2 Chunking Method
- **Input:** The `processed_content` from the SQLite `articles` table.
- **Implementation:** The `TextProcessingPipeline.split_into_chunks()` method in `src/financial_news_rag/text_processor.py`.
- **Sentence-Based Splitting:**
  - `processed_content` is split into sentences using NLTK’s `sent_tokenize` (with a regex-based fallback).
  - Sentences are grouped into chunks such that the total estimated tokens per chunk does not exceed `self.max_tokens_per_chunk`.
  - Token estimation: `len(text) // 4`.
  - Handles cases where single sentences might exceed the token limit by splitting them further.
- **No Overlap (Current):** Chunks are non-overlapping.
- **Chunk Metadata:** Each chunk, when retrieved via `TextProcessingPipeline.get_chunks_for_article()`, includes metadata from the parent article (e.g., `url_hash`, `title`, `published_at`, `symbols`, `tags`).
- **Output:** A list of text chunks. These chunks are what will be sent to the embedding model. The relationship between chunks and the parent article (identified by `url_hash` in SQLite) is maintained.

**Reference Implementation Snippet (from `src/financial_news_rag/text_processor.py`):**
```python
# ... (within TextProcessingPipeline class)
def split_into_chunks(self, processed_text: str) -> List[str]:
    # ...
    try:
        sentences = sent_tokenize(processed_text)
    except Exception as e:
        # Fallback to simple regex-based sentence splitting
        sentences = re.split(r\'(?<=[.!?])\\s+\', processed_text)
    # ...
    # (Loop through sentences, estimate tokens, and group into chunks)
    # ...
    return chunks
```

---

## 3. Preprocessing for Embedding Generation

- **Pipeline Order (as demonstrated in `examples/text_processing_example.py` and `src/financial_news_rag/text_processor.py`):**
  1. Fetch `raw_content` from an external source (e.g., EODHD API).
  2. Store articles (including `raw_content`) in SQLite using `TextProcessingPipeline.store_articles()`. Duplicates (based on `url_hash`) are typically ignored.
  3. Process pending articles using `TextProcessingPipeline.process_articles()`:
     a. Retrieve articles with `status_text_processing = 'PENDING'`.
     b. Clean `raw_content` using `clean_article_text()` to produce `processed_content`.
     c. Store `processed_content` in SQLite and update `status_text_processing` (e.g., to 'SUCCESS' or 'FAILED').
  4. Retrieve `processed_content` for embedding (e.g., using `TextProcessingPipeline.get_processed_articles_for_embedding()` or `get_chunks_for_article()`).
  5. Split `processed_content` into chunks using `split_into_chunks()`.
  6. (Future Step) Generate embeddings for each valid chunk and store them in a vector database (e.g., ChromaDB), linking back to the SQLite `articles` table via `url_hash`. Update `status_embedding` in SQLite.
- **Entity Extraction:** Entities are primarily identified from the EODHD API response fields (e.g., `symbols`, `tags`), which are stored alongside the article in SQLite.
- **Metadata Association:** Each chunk is associated with its parent article's metadata stored in SQLite (title, `url_hash`, published_at, EODHD symbols/tags, etc.). This is evident in the output of `TextProcessingPipeline.get_chunks_for_article()`.

---

## 4. Tokenization Approach

- **Token Estimation:**
  - For chunking `processed_content` from SQLite, tokens are estimated as `len(text) // 4` (Gemini models: 1 token ≈ 4 characters). This is used in `TextProcessingPipeline.split_into_chunks()`.
  - For embedding, the Google API handles tokenization internally, but pre-chunking ensures compliance with its limits.
- **Text Elements:**
  - The `raw_content` field from the SQLite `articles` table is the initial input for cleaning.
  - The `processed_content` field in SQLite is the direct input for chunking.
  - Article `title` from SQLite is preserved as metadata.
  - EODHD `symbols` and `tags` are preserved as metadata in SQLite.
- **Special Handling:**
  - Financial symbols, tickers, and numbers are preserved in the text during cleaning.
  - Non-ASCII characters are normalized or handled during the cleaning process.

---

## 5. Database Interaction
The `TextProcessingPipeline` class heavily relies on an SQLite database (default: `financial_news.db`) for:
- Storing raw and processed article content.
- Tracking the status of text processing (`status_text_processing`) and embedding (`status_embedding`) for each article.
- Logging API calls (`api_call_log` table) as demonstrated in `examples/text_processing_example.py` using `pipeline.log_api_call()`.
- Storing article metadata (URL, title, published date, tags, symbols, etc.).

Refer to `TextProcessingPipeline._init_database()` and other database-related methods in `src/financial_news_rag/text_processor.py` for schema details and interaction logic.

---

## 6. Example Workflow
The script `examples/text_processing_example.py` provides a practical demonstration of using the `TextProcessingPipeline`:
1. Initializes `EODHDClient` and `TextProcessingPipeline`.
2. Fetches news articles.
3. Logs the API call details.
4. Stores articles in the SQLite database via `pipeline.store_articles()`.
5. Processes these articles (cleaning) using `pipeline.process_articles()`.
6. Retrieves and displays chunks for a sample article using `pipeline.get_chunks_for_article()`.

---

## 7. References
- **Source Code:** `src/financial_news_rag/text_processor.py`
- **Example Usage:** `examples/text_processing_example.py`
- [Project Specification](./project_spec.md)
- [Model Details](./model_details.md)
- [Technical Design](./technical_design.md)
- [EODHD API Guide](./eodhd_api.md)

---

**This document is the authoritative reference for the text processing pipeline in the Financial News RAG system. All implementation and future design decisions should align with this pipeline, recognizing SQLite as the primary store for text at various stages.**
