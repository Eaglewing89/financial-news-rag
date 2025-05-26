# Testing Guide

This document describes the testing strategy, structure, and usage for the Financial News RAG project. It is intended for contributors and maintainers to ensure code quality, reliability, and maintainability.

---

## Table of Contents

- [Overview](#overview)
- [Test Suite Structure](#test-suite-structure)
- [How to Run Tests](#how-to-run-tests)
- [Test Coverage](#test-coverage)
- [Fixtures, Factories, and Helpers](#fixtures-factories-and-helpers)
- [Test Types](#test-types)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

All tests use [pytest](https://docs.pytest.org/) and are located in the `tests/` directory. The suite includes both unit and integration tests, with extensive use of fixtures, factories, and custom assertion helpers to ensure isolation and reproducibility.

---

## Test Suite Structure

```
tests/
    __init__.py
    conftest.py
    fixtures/
        factories.py
        sample_data.py
    helpers/
        assertions.py
        mock_helpers.py
    integration/
        test_orchestrator_integration.py
    unit/
        test_article_manager.py
        test_chroma_manager.py
        test_config.py
        test_embeddings.py
        test_eodhd.py
        test_reranker.py
        test_text_processor.py
        test_utils.py
```

- **unit/**: Unit tests for individual modules and classes.
- **integration/**: Integration tests for orchestrator and end-to-end workflows.
- **fixtures/**: Data and object factories for generating test data.
- **helpers/**: Custom assertion functions and mock utilities.
- **conftest.py**: Shared pytest fixtures and configuration.

---

## How to Run Tests

From the project root, run:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/unit/test_article_manager.py
```

To run only integration tests:

```bash
pytest tests/integration/
```

To see verbose output:

```bash
pytest -v
```

---

## Test Coverage

To check code coverage (requires `pytest-cov`):

```bash
pytest --cov=financial_news_rag
```

A summary will be printed in the terminal. Aim for at least 80% coverage. For a detailed HTML report:

```bash
pytest --cov=financial_news_rag --cov-report=html
```

Open `htmlcov/index.html` in your browser to view the report.

---

## Fixtures, Factories, and Helpers

### Fixtures (`conftest.py`)

- **Configuration and Environment**: `test_config`, `temp_directory`, `temp_db_path`
- **Component Instances**: `article_manager`, `chroma_manager`
- **Mocked Components**: `mock_eodhd_client`, `mock_text_processor`, `mock_embeddings_generator`, `mock_chroma_manager`, `mock_reranker`, `mock_article_manager`
- **Orchestrator Mocks**: `mock_orchestrator_components`, `orchestrator_with_mocks`
- **Test Data**: `sample_article`, `sample_articles_list`, `sample_processed_chunks`, `sample_embeddings`, `sample_chroma_results`, `sample_html_content`, etc.
- **Parameterized Fixtures**: `tag_parameter`, `symbol_parameter`, `status_parameter`
- **Utility Fixtures**: `assert_article_stored`, `assert_chunks_in_chroma`

Fixtures are automatically injected into tests as needed, providing isolation and reproducibility.

### Factories (`fixtures/factories.py`, `fixtures/sample_data.py`)

- **ArticleFactory**: Generates realistic article dictionaries with random or specified fields.
- **ChunkFactory**: Generates text chunks and embeddings.
- **TestScenarioFactory**: Builds complex, multi-component test scenarios.
- **ConfigDataFactory**: Produces environment variable sets for config testing.

Factories allow for flexible, repeatable test data generation.

### Helpers (`helpers/assertions.py`, `helpers/mock_helpers.py`)

- **assertions.py**: Custom assertion functions for validating article structure, processing status, embedding status, database statistics, and ChromaDB results.
- **mock_helpers.py**: Utilities for setting up test environments, mocks, and common operations.

---

## Test Types

- **Unit Tests**: Located in `tests/unit/`. Test individual classes and functions in isolation, using mocks and fixtures.
- **Integration Tests**: Located in `tests/integration/`. Test the orchestrator and full pipeline, using real implementations and mocked external APIs.
- **End-to-End Scenarios**: Some integration tests simulate realistic workflows, including fetching, processing, embedding, storing, and searching articles.

---

## Best Practices

- Use fixtures and factories for all test data and component setup.
- Mock external APIs (EODHD, Gemini) to avoid network calls and costs.
- Use custom assertion helpers for clear, domain-specific test validation.
- Keep tests isolated and idempotent—no test should depend on the outcome of another.
- Maintain high coverage and add tests for new features and bug fixes.

---

## Troubleshooting

- If tests fail due to missing dependencies, install them with:
  ```bash
  pip install -r requirements.txt
  ```
- For database or file-related errors, ensure tests are using temporary paths and not writing to production data.
- Use `pytest -s` to see print/debug output during test runs.

---
