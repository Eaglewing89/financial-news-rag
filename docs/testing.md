# Testing and Quality Assurance for Financial News RAG

## Purpose
This document defines the testing strategy, implementation plan, and quality metrics for the Financial News RAG module. It is the authoritative reference for all testing-related practices in this project.

## 1. Testing Philosophy
- Ensure correctness, reliability, and maintainability of the core RAG module.
- Validate all critical workflows: news ingestion, text processing, embedding, storage, and search.
- Provide confidence for future refactoring and extension.

## 2. Test Types and Scope
### 2.1 Unit Tests
- **Goal:** Verify correctness of individual functions/classes in isolation.
- **Scope:**
  - Text cleaning and chunking (see [text_processing_pipeline.md](text_processing_pipeline.md))
  - Embedding generation (mocked API calls)
  - ChromaDB storage/retrieval logic
  - Utility functions (date parsing, config loading, etc.)

### 2.2 Integration Tests
- **Goal:** Validate interactions between components.
- **Scope:**
  - End-to-end news ingestion (from API fetch to ChromaDB storage)
  - Semantic search and Gemini re-ranking (mocked Gemini API)
  - Data refresh and update workflows

### 2.3 End-to-End (E2E) Tests
- **Goal:** Simulate real user workflows and ensure the system works as a whole.
- **Scope:**
  - Search for news and validate results
  - Add new articles and verify retrievability
  - Refresh data and check for correct updates

### 2.4 Performance and Coverage
- **Coverage:**
  - Use `pytest-cov` to ensure at least 80% code coverage.
- **Performance:**
  - Benchmark search latency and throughput for typical queries.

## 3. Implementation Plan
- All tests are implemented using `pytest`.
- Test files are located in the `tests/` directory:
  - `test_search.py`: Search and re-ranking logic
  - `test_embeddings.py`: Embedding and chunking logic
  - `test_data.py`: Data ingestion, cleaning, and storage
- Use mocking for all external API calls (EODHD, Gemini, Google Embeddings) to ensure tests are deterministic and do not require network access.
- Use temporary or in-memory ChromaDB instances for tests to avoid polluting production data.
- Add fixtures for common test data (sample articles, embeddings, etc.).
- Run all tests and coverage checks before each release or major change.

## 4. Quality Metrics
- **Precision:** Accuracy of retrieved articles for a query
- **Recall:** Percentage of relevant articles retrieved
- **F1 Score:** Harmonic mean of precision and recall
- **Latency:** Response time for search queries
- **Throughput:** Number of queries handled per unit time

## 5. How to Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/financial_news_rag --cov-report=term-missing
```

## 6. References
- [project_spec.md](project_spec.md#testing-strategy)
- [technical_design.md](technical_design.md#testing)
- [text_processing_pipeline.md](text_processing_pipeline.md)
- [model_details.md](model_details.md)

---

**This document is the single source of truth for testing in the Financial News RAG project. All contributors must follow this plan.**
