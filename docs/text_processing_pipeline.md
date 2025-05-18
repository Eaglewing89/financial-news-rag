# Document Text Processing Pipeline

## Description
This document details the text processing pipeline for the Financial News RAG system. It covers text cleaning, chunking, preprocessing for embedding, and tokenization strategies, referencing architectural and implementation decisions from the project’s technical documentation. The primary source of text for this pipeline is the **full article content** obtained from the EODHD API.

---

## 1. Text Cleaning Approaches

### 1.1 Cleaning and Normalization
- **HTML Removal:** All article `content` (from EODHD API) is stripped of HTML tags using regex or BeautifulSoup.
- **Whitespace Normalization:** Consecutive whitespace is collapsed to a single space and leading/trailing whitespace is trimmed.
- **Boilerplate Removal:** Common phrases (e.g., “Click here to read more.”) are removed using regex.
- **Encoding Fixes:** Known encoding issues (e.g., smart quotes, dashes) are replaced with standard ASCII equivalents.
- **Unicode Normalization:** All text is normalized to NFC form to ensure consistent representation.
- **Special Characters:** Financial symbols and special characters are preserved if relevant, but extraneous non-informative characters are removed.
- **Deduplication:** Duplicate content is detected and removed at the article and chunk level.

**Reference Implementation:**
```python
def clean_article_text(text):
    """Clean and normalize article text content (primarily from EODHD API)."""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove boilerplate
    text = re.sub(r'Click here to read more\.?', '', text)
    # Fix encoding issues
    text = text.replace('\u00e2\u20ac\u2122', "'").replace('\u00e2\u20ac\u0153', '"').replace('\u00e2\u20ac', '"')
    return text
```

---

## 2. Chunking Strategies for RAG

### 2.1 Model Constraints
- **Embedding Model:** Google’s `text-embedding-004` (see `model_details.md`)
- **Token Limit:** 2,048 tokens per chunk (~1,500–1,600 words; 1 token ≈ 4 characters)

### 2.2 Chunking Method
- **Sentence-Based Splitting:**
  - Full article `content` from EODHD is split into sentences using NLTK’s `sent_tokenize`.
  - Sentences are grouped into chunks such that the total estimated tokens per chunk does not exceed 2,048.
  - Token estimation: `len(text) // 4`.
- **No Overlap (MVP):** Chunks are non-overlapping for simplicity. Overlapping windows may be considered in future enhancements.
- **Chunk Metadata:** Each chunk retains reference to its parent article and position for traceability.

**Reference Implementation:**
```python
def split_into_chunks(text, max_tokens=2048):
    """Splits full article text into manageable chunks for embedding."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    token_estimator = lambda s: len(s) // 4
    for sentence in sentences:
        sentence_tokens = token_estimator(sentence)
        if current_length + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
```

---

## 3. Preprocessing for Embedding Generation

- **Pipeline Order:**
  1. Clean and normalize full article `content` from EODHD (see above).
  2. Split into chunks (see above).
  3. (Optional) Summarize if article is extremely long
  4. Remove duplicate or near-duplicate chunks
  5. Validate chunk size (≤2,048 tokens)
- **Entity Extraction:** Entities are primarily identified from the EODHD API response fields (e.g., `symbols`, `tags`). If Marketaux API is used for supplementary data, its entity metadata can also be incorporated.
- **Metadata Association:** Each chunk is associated with article metadata (title, uuid, published_at, EODHD symbols/tags, etc.) for downstream storage and retrieval.
- **Unicode and Encoding:** All text is normalized to ensure compatibility with embedding API.

---

## 4. Tokenization Approach

- **Token Estimation:**
  - For chunking, tokens are estimated as `len(text) // 4` (Gemini models: 1 token ≈ 4 characters).
  - For embedding, the Google API handles tokenization internally, but pre-chunking ensures compliance with limits.
- **Text Elements:**
  - The `content` field from the EODHD API response (full article text) is the primary input for cleaning, chunking, and embedding.
  - Article `title` from EODHD may also be considered for embedding or as metadata.
  - EODHD `symbols` and `tags` are preserved in metadata.
  - If Marketaux is used, its `snippet` and entity highlights are secondary and handled as metadata.
- **Special Handling:**
  - Financial symbols, tickers, and numbers are preserved in the text.
  - Non-ASCII characters are normalized or removed if not relevant.

---

## 5. References
- [Project Specification](./project_spec.md)
- [Model Details](./model_details.md)
- [Technical Design](./technical_design.md)
- [EODHD API Guide](./eodhd_api.md) (Primary source for article content)
- [Marketaux API Guide](./marketaux_api.md) (Secondary source for snippets/metadata)

---

**This document is the authoritative reference for the text processing pipeline in the Financial News RAG system. All implementation and future design decisions should align with this pipeline.**
