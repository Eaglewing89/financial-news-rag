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

The system is a modular Python package that implements a RAG pipeline for financial news. It fetches **full news articles primarily from the EODHD API**, processes and embeds them, stores them in ChromaDB, and enables semantic search with Gemini-based re-ranking. The Marketaux API may be used as a secondary source for snippets or specific metadata if needed for tasks outside the primary RAG pipeline. The architecture is designed for future API wrapping and multi-agent integration.

**Key Components:**
- News Fetcher (primarily EODHD API integration, potentially Marketaux for secondary tasks)
- Text Processing Pipeline
- Embedding Generator (Google text-embedding-004)
- Vector Store (ChromaDB)
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
|                   |<--- Marketaux API (Secondary - Snippets, Optional)
+-------------------+
         |
         v
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
+-------------------+
         |
         v
+-------------------+
| Semantic Search   |
+-------------------+
         |
         v
+-------------------+
| Gemini Reranker   |<--- Gemini 2.0 Flash
+-------------------+
```

## Data Flow Diagrams

### News Ingestion & Storage

1. Fetch full articles from EODHD API
2. Clean and normalize text (full content)
3. Extract entities and metadata (from EODHD response)
4. Generate embeddings (Google API)
5. Store articles + embeddings in ChromaDB

### Search & Reranking

1. User issues search query
2. Query embedded articles in ChromaDB (semantic similarity)
3. (Optional) Filter by entity/date
4. Top results sent to Gemini for re-ranking
5. Return sorted, relevant articles to user

## ChromaDB Schema Definitions

**Collection Name:** `financial_news`

**Document Schema (based on EODHD API - see [eodhd_api.md](./eodhd_api.md#response-fields-json) for full details):**
```json
{
  "uuid": "unique-article-id", // Generated locally or use EODHD link/guid if unique
  "title": "Article title from EODHD",
  "url": "https://eodhd.com/news-link or original source link",
  "source_api": "EODHD", // To distinguish from other potential sources
  "published_at": "2025-05-18T10:00:00Z", // From EODHD 'date' field
  "content": "Full article content from EODHD",
  "symbols": ["AAPL.US", "MSFT.US"], // From EODHD 'symbols' field
  "tags": ["earnings", "technology"], // From EODHD 'tags' field
  "sentiment": {"polarity": 0.324, "neg": 0.065, "neu": 0.862, "pos": 0.073}, // From EODHD 'sentiment' field
  "embedding_model": "text-embedding-004",
  "embedding_timestamp": "2025-05-18T10:05:00Z",
  "fetched_timestamp": "2025-05-18T10:00:30Z"
}
```

**ChromaDB Storage:**
- `documents`: Full article `content` (chunked if necessary for embedding model context limits)
- `metadatas`: All fields above (or a relevant subset) except the embedding vector itself. UUID can be part of metadata if not the primary ID.
- `ids`: A unique identifier for each document/chunk (e.g., generated UUID, or a hash of the URL).
- `embeddings`: Vector generated from the `content` using the Google API.

## Component Interactions

- **News Fetcher**: Primarily calls EODHD API for full articles, handles retries/rate limits, returns processed articles. May interact with Marketaux API for supplementary data if configured.
- **Text Processing**: Cleans full article text, extracts entities/tags from EODHD data, normalizes data.
- **Embedding Generator**: Sends text to Google API, receives embedding vector.
- **ChromaDB**: Stores/retrieves articles and embeddings, supports metadata filtering.
- **Semantic Search**: Queries ChromaDB for similar articles.
- **Gemini Reranker**: Receives top results, re-ranks using Gemini 2.0 Flash, returns sorted list.
- **Config/Error Management**: Loads .env, validates keys, logs errors, manages retries.

## Error Handling Strategies

- **API Failures**: Retry with exponential backoff (see [`fetch_eodhd_with_retry` pattern in eodhd_api.md](./eodhd_api.md#error-codes--handling) for EODHD. Similar patterns apply if Marketaux is used, see [marketaux_api.md#error-codes--handling](./marketaux_api.md#error-codes--handling)).
- **Rate Limiting**: Adhere to API-specific rate limits. For EODHD, this includes managing the 5 calls per news request and daily limits (see [eodhd_api.md](./eodhd_api.md#rate-limiting--usage-limits)). For Marketaux, refer to its documentation.
- **ChromaDB Errors**: Catch and log, fail gracefully, optionally retry
- **Embedding/Reranking Failures**: Fallback to original ranking or skip embedding
- **Configuration Errors**: Validate on startup, raise clear exceptions if missing
- **Logging**: All errors logged with context; user-facing errors are friendly

> **Reference Patterns:**
> The canonical implementations of retry logic and rate limit considerations are maintained in the respective API documentation files: [eodhd_api.md](./eodhd_api.md#error-codes--handling) for EODHD and [marketaux_api.md](./marketaux_api.md#error-codes--handling) for Marketaux. This technical design document references those as the source of truth for these error handling patterns.

## Configuration Management

- **API Keys**: Managed via `.env` file, loaded with `python-dotenv`
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
