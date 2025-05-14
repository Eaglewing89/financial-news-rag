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

---

## 1. Introduction

This document provides a comprehensive technical design for the Financial News Retrieval Augmented Generation (RAG) module. It formalizes the system architecture, data flow, component interactions, schema definitions, error handling, and configuration management. This blueprint ensures robust, maintainable, and extensible implementation aligned with the [Project Specification](./project_spec.md).

## 2. System Architecture Overview

The system is a modular Python package that implements a RAG pipeline for financial news. It fetches news from Marketaux, processes and embeds articles, stores them in ChromaDB, and enables semantic search with Gemini-based re-ranking. The architecture is designed for future API wrapping and multi-agent integration.

**Key Components:**
- News Fetcher (Marketaux API integration)
- Text Processing Pipeline
- Embedding Generator (Google text-embedding-004)
- Vector Store (ChromaDB)
- Semantic Search Engine
- Gemini Reranker (Gemini 2.0 Flash)
- Configuration & Error Management

## 3. Component Architecture Diagram

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
|  News Fetcher     |<--- Marketaux API
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

## 4. Data Flow Diagrams

### 4.1 News Ingestion & Storage

1. Fetch articles from Marketaux API
2. Clean and normalize text
3. Extract entities and metadata
4. Generate embeddings (Google API)
5. Store articles + embeddings in ChromaDB

### 4.2 Search & Reranking

1. User issues search query
2. Query embedded articles in ChromaDB (semantic similarity)
3. (Optional) Filter by entity/date
4. Top results sent to Gemini for re-ranking
5. Return sorted, relevant articles to user

## 5. ChromaDB Schema Definitions

**Collection Name:** `financial_news`

**Document Schema:**
```json
{
  "uuid": "unique-article-id",
  "title": "Article title",
  "url": "https://source.com/article",
  "source": "Source name",
  "published_at": "2025-05-14T10:00:00Z",
  "content": "Full article content",
  "entities": [
    {
      "symbol": "TSLA",
      "name": "Tesla Inc.",
      "type": "equity",
      "sentiment_score": 0.85
    }
  ],
  "embedding_model": "text-embedding-004",
  "embedding_timestamp": "2025-05-14T10:05:00Z"
}
```

**ChromaDB Storage:**
- `documents`: Article content (chunked if needed)
- `metadatas`: All fields above except embedding vector
- `ids`: UUIDs
- `embeddings`: Vector from Google API

## 6. Component Interactions

- **News Fetcher**: Calls Marketaux API, handles retries/rate limits, returns raw articles.
- **Text Processing**: Cleans text, extracts entities, normalizes data.
- **Embedding Generator**: Sends text to Google API, receives embedding vector.
- **ChromaDB**: Stores/retrieves articles and embeddings, supports metadata filtering.
- **Semantic Search**: Queries ChromaDB for similar articles.
- **Gemini Reranker**: Receives top results, re-ranks using Gemini 2.0 Flash, returns sorted list.
- **Config/Error Management**: Loads .env, validates keys, logs errors, manages retries.

## 7. Error Handling Strategies

- **API Failures**: Retry with exponential backoff (see [`fetch_with_retry` pattern](./project_spec.md#error-handling-strategy))
- **Rate Limiting**: Simple in-memory rate limiter (see [`RateLimiter` class](./project_spec.md#error-handling-strategy))
- **ChromaDB Errors**: Catch and log, fail gracefully, optionally retry
- **Embedding/Reranking Failures**: Fallback to original ranking or skip embedding
- **Configuration Errors**: Validate on startup, raise clear exceptions if missing
- **Logging**: All errors logged with context; user-facing errors are friendly

> **Reference Patterns:**
> The canonical implementations of `fetch_with_retry` and `RateLimiter` are maintained in the [Project Specification: Error Handling Strategy](./project_spec.md#error-handling-strategy). This technical design document references those as the source of truth for error handling patterns.

## 8. Configuration Management

- **API Keys**: Managed via `.env` file, loaded with `python-dotenv`
- **Required Variables**:
  - `GEMINI_API_KEY`
  - `MARKETAUX_API_KEY`
- **Loading Pattern**:
  ```python
  from dotenv import load_dotenv
  import os
  load_dotenv()
  gemini_api_key = os.getenv('GEMINI_API_KEY')
  marketaux_api_key = os.getenv('MARKETAUX_API_KEY')
  ```
- **Validation**: On module init, check for required keys
- **Environment Separation**: Support for dev/test/prod .env files

## 9. Security Considerations

- API keys never hardcoded; always loaded from environment
- Sensitive data not logged
- Future: Use secrets manager for production

## 10. Extensibility and Future Enhancements

- Multi-source news ingestion (Finnhub, Newsfilter, etc.)
- Advanced filtering, analytics, and sentiment analysis
- API layer (FastAPI) for external access
- Automated refresh and monitoring
- Schema versioning and migrations

---

**This document is the authoritative technical reference for the Financial News RAG module. All implementation and future design decisions should align with this blueprint.**
