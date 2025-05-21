# Technical Design Document

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture Overview](#system-architecture-overview)
3. [Component Architecture Diagram](#component-architecture-diagram)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [ChromaDB Schema Definitions](#chromadb-schema-definitions)
6. [Component Interactions](#component-interactions)
7. [Error Handling Strategies](#error-handling-strategies)
8. [Configuration Management](#configuration-management)
9. [Security Considerations](#security-considerations)
10. [Extensibility and Future Enhancements](#extensibility-and-future-enhancements)
11. [Testing](#testing)

---

## Introduction

This document provides a comprehensive technical design for the Financial News Retrieval Augmented Generation (RAG) module. It formalizes the system architecture, data flow, component interactions, schema definitions, error handling, and configuration management. This blueprint ensures robust, maintainable, and extensible implementation aligned with the [Project Specification](./project_spec.md).

## System Architecture Overview

The system is a modular Python package that implements a RAG pipeline for financial news. It fetches **full news articles primarily from the EODHD API**. The fetched articles, along with their metadata and processing status, are stored in a **local SQLite database**. The text content is then processed and embedded. These **embeddings, along with a reference to the SQLite database, are stored in ChromaDB**. The system enables semantic search (querying ChromaDB for relevant embedding IDs and then retrieving full content from SQLite) with Gemini-based re-ranking. The Marketaux API may be used as a secondary source for snippets or specific metadata if needed for tasks outside the primary RAG pipeline. The architecture is designed for future API wrapping and multi-agent integration.

**Key Components:**
- News Fetcher (primarily EODHD API integration)
- **SQLite Database (Primary store for article metadata, raw content, processed content, and pipeline status)**
- Text Processing Pipeline
- Embedding Generator (Google text-embedding-004)
- **Vector Store (ChromaDB - Stores embeddings and references to SQLite)**
- Semantic Search Engine
- Gemini Reranker (Gemini 2.0 Flash)
- Configuration & Error Management

## Component Architecture Diagram

```
+-------------------+
|  User/Client Code |
+--------+----------+
         |
         v
+--------+----------+
|  Core Module API  |
| (search_news, ...)|
+--------+----------+
         |
         v
+-------------------+
|  News Fetcher     |<--- EODHD API (Primary - Full Articles)
+-------------------+
         |
         v
+-------------------+
|  SQLite Database  |
| (Articles Table:  |
| metadata, raw &   |
| processed content,|
| status)           |
+-------------------+
         |         ^
         |         |
         v         |
+-------------------+
| Text Processing   |
+-------------------+
         |
         v
+-------------------+
| Embedding Model   |<--- Google text-embedding-004
+-------------------+
         |
         v
+-------------------+
|   ChromaDB        |
| (Embeddings +     |
|  SQLite Ref ID)   |
+-------------------+
         |         ^
         |         |
         v         |
+-------------------+
| Semantic Search   |--- (Retrieves content from SQLite via Ref ID)
+-------------------+
         |
         v
+-------------------+
| Gemini Reranker   |<--- Gemini 2.0 Flash
+-------------------+
```

## Data Flow Diagrams

### News Ingestion & Storage

1. Fetch full articles from EODHD API.
2. Store raw article content and metadata in the **SQLite database** (see `articles` table in [eodhd_api.md](./eodhd_api.md#articles-table)). Update status.
3. Clean and normalize text (from raw content in SQLite). Store processed content in SQLite. Update status.
4. Extract entities and metadata (from EODHD response, already in SQLite).
5. Generate embeddings from processed content (Google API).
6. Store embeddings in **ChromaDB** with a reference ID (e.g., `url_hash`) linking back to the article's record in SQLite. Update status in SQLite.

### Search & Reranking

1. User issues search query.
2. Query embedded articles in **ChromaDB** (semantic similarity) to get relevant reference IDs (e.g., `url_hash`) and scores.
3. (Optional) Filter by entity/date (can be done via ChromaDB metadata if limited metadata is stored there, or by querying SQLite after initial ID retrieval).
4. Retrieve full processed content and other necessary metadata from **SQLite** using the reference IDs.
5. Top results (content from SQLite) sent to Gemini for re-ranking.
6. Return sorted, relevant articles to user.

## ChromaDB Schema Definitions

**Collection Name:** `financial_news_embeddings` (Default collection name in `ChromaDBManager`)

**ChromaDB Entry Structure:** 
Implemented in [`src/financial_news_rag/chroma_manager.py`](../src/financial_news_rag/chroma_manager.py) as `ChromaDBManager.add_embeddings()`:

- **`ids`**: A unique identifier for each chunk embedding, formatted as `"{article_url_hash}_{chunk_index}"`. This ensures a direct link back to the article in SQLite and maintains the sequence of chunks.
- **`embeddings`**: Vector (dimension 768) generated from chunks of `processed_content` (stored in SQLite) using Google's text-embedding-004 model.
- **`documents`**: The actual text content of each chunk, stored to enable direct document retrieval during querying.
- **`metadatas`**: A set of metadata for filtering directly in ChromaDB:
  ```json
  {
    "article_url_hash": "hash_value",       // Links directly to SQLite article
    "chunk_index": 0,                       // Position of chunk in article
    "published_at_timestamp": 1678886400,   // Optional: timestamp for date filtering
    "source_query_tag": "technology"        // Optional: filtering by original query tag
  }
  ```

The implementation maintains the primary source of detailed metadata in the SQLite database, with ChromaDB storing only the minimal necessary metadata for efficient filtering. The schema for the SQLite `articles` table is detailed in [eodhd_api.md](./eodhd_api.md#articles-table).

For a working example of the full end-to-end process (from SQLite to embeddings to ChromaDB), see [`examples/generate_and_store_embeddings_example.py`](../examples/generate_and_store_embeddings_example.py).

**SQLite Database Schema:**
The definitive schema for the SQLite database, particularly the `articles` table which stores all metadata, raw content, processed content, and pipeline statuses, is maintained in the [EODHD API Integration Guide](./eodhd_api.md#database-for-tracking-api-usage). ChromaDB entries link to this table via the `article_url_hash` field in each chunk's metadata.

## Component Interactions

- **News Fetcher**: Primarily calls EODHD API for full articles, handles retries/rate limits.
- **SQLite Database**: Stores fetched articles (raw content, metadata), processed content, and pipeline statuses. Acts as the source of truth for article data.
- **Text Processing**: Reads raw content from SQLite, cleans it, and writes processed content back to SQLite. Implemented in [`src/financial_news_rag/text_processor.py`](../src/financial_news_rag/text_processor.py) as `TextProcessingPipeline`.
- **Embedding Generator**: Reads processed content from SQLite, sends text to Google API, receives embedding vectors. Implemented in [`src/financial_news_rag/embeddings.py`](../src/financial_news_rag/embeddings.py) as `EmbeddingsGenerator`.
- **ChromaDB Manager**: Handles all ChromaDB operations, including storing embeddings with references to SQLite and querying similar vectors. Implemented in [`src/financial_news_rag/chroma_manager.py`](../src/financial_news_rag/chroma_manager.py) as `ChromaDBManager`.
- **Semantic Search**: Queries ChromaDB for similar embedding IDs, then retrieves full article details from SQLite using these IDs.
- **Gemini Reranker**: Receives top results, re-ranks using Gemini 2.0 Flash, returns sorted list.
- **Config/Error Management**: Loads .env, validates keys, logs errors, manages retries.

## Error Handling Strategies

- **API Failures**: Retry with exponential backoff (see [`fetch_eodhd_with_retry` pattern in eodhd_api.md](./eodhd_api.md#error-codes--handling) for EODHD. Similar patterns apply if Marketaux is used, see [marketaux_api.md#error-codes--handling](./marketaux_api.md#error-codes--handling)).
- **Rate Limiting**: Adhere to API-specific rate limits. For EODHD, this includes managing the 5 calls per news request and daily limits (see [eodhd_api.md](./eodhd_api.md#rate-limiting--usage-limits)). For Marketaux, refer to its documentation.
- **ChromaDB Errors**: Catch and log, fail gracefully with appropriate status returns. Implemented in [`src/financial_news_rag/chroma_manager.py`](../src/financial_news_rag/chroma_manager.py) with comprehensive exception handling.
- **Embedding/Reranking Failures**: Fallback to original ranking or skip embedding. Embedding API errors and retries are handled in [`src/financial_news_rag/embeddings.py`](../src/financial_news_rag/embeddings.py).
- **Configuration Errors**: Validate on startup, raise clear exceptions if missing
- **Logging**: All errors logged with context; user-facing errors are friendly

> **Reference Patterns:**
> The canonical implementations of retry logic and rate limit considerations are maintained in the respective API documentation files: [eodhd_api.md](./eodhd_api.md#error-codes--handling) for EODHD and [marketaux_api.md](./marketaux_api.md#error-codes--handling) for Marketaux. This technical design document references those as the source of truth for these error handling patterns.

## Configuration Management

- **API Keys**: Managed via `.env` file, loaded with `python-dotenv` (see [`src/financial_news_rag/config.py`](../src/financial_news_rag/config.py) and [`src/financial_news_rag/embeddings.py`](../src/financial_news_rag/embeddings.py)).
- **Required Variables**:
  - `GEMINI_API_KEY`
  - `EODHD_API_KEY`
- **Optional Variables**:
  - `MARKETAUX_API_KEY` (if Marketaux is used for secondary tasks)
- **Loading Pattern**:
  ```python
  from dotenv import load_dotenv
  import os
  load_dotenv()
  gemini_api_key = os.getenv('GEMINI_API_KEY')
  eodhd_api_key = os.getenv('EODHD_API_KEY')
  marketaux_api_key = os.getenv('MARKETAUX_API_KEY') # Load if used
  ```
- **Validation**: On module init, check for required keys
- **Environment Separation**: Support for dev/test/prod .env files

## Security Considerations

- API keys never hardcoded; always loaded from environment
- Sensitive data not logged
- Future: Use secrets manager for production

## Extensibility and Future Enhancements

- Multi-source news ingestion (Finnhub, Newsfilter, etc.)
- Advanced filtering, analytics, and sentiment analysis
- API layer (FastAPI) for external access
- Automated refresh and monitoring
- Schema versioning and migrations

## Testing

All testing strategy, implementation details, and quality metrics are now documented in [testing.md](testing.md). This includes:
- Unit, integration, and end-to-end test plans
- Coverage and performance requirements
- Test data and mocking strategies
- How to run and interpret tests

All contributors must follow the [testing.md](testing.md) plan for any new code or changes.

---

**This document is the authoritative technical reference for the Financial News RAG module. All implementation and future design decisions should align with this blueprint.**
