# Testing Strategy

This document outlines the testing strategy for the Financial News RAG project. The goal is to ensure the reliability, correctness, and robustness of each component and the overall system.

## 1. Levels of Testing

### 1.1. Unit Tests

*   **Focus**: Individual classes and their methods in isolation.
*   **Scope**: Each low-level component will have dedicated unit tests.
    *   **`EODHDClient`**: Mock EODHD API responses to test news fetching logic, parameter handling, and error conditions (e.g., API key errors, network issues).
    *   **`ArticleManager`**: Test database interactions (CRUD operations for articles and chunks), status updates, and querying logic. Use an in-memory SQLite database for test isolation and speed.
    *   **`TextProcessor`**: Test text cleaning functions (e.g., HTML tag removal, whitespace normalization) and various text chunking strategies with diverse inputs (short text, long text, text with unusual formatting).
    *   **`EmbeddingsGenerator`**: Test embedding generation for sample texts. Mock the sentence transformer model if necessary for speed, or use a very small model for actual embedding tests. Verify embedding dimensions and types.
    *   **`ChromaDBManager`**: Test adding embeddings, querying for similar items, and managing ChromaDB collections. Use an in-memory or temporary ChromaDB instance.
*   **Tools**: `pytest` framework.
*   **Location**: `tests/` directory, with filenames like `test_article_manager.py`, `test_text_processor.py`, etc.

### 1.2. Integration Tests

*   **Focus**: Interactions between two or more components.
*   **Scope**:
    *   Test the flow from `ArticleManager` -> `TextProcessor` -> `ArticleManager` (storing raw articles, processing them, storing chunks, updating statuses).
    *   Test the flow from `ArticleManager` -> `EmbeddingsGenerator` -> `ChromaDBManager` -> `ArticleManager` (retrieving chunks, generating embeddings, storing in ChromaDB, updating statuses).
    *   Test the interaction between `EODHDClient` and `ArticleManager` (fetching news and storing it).
*   **Tools**: `pytest` framework.
*   **Location**: Within the `tests/` directory, potentially in files like `test_pipeline_integration.py`.

### 1.3. End-to-End (E2E) Tests

*   **Focus**: Testing the entire pipeline workflow, from data ingestion to retrieval (or as much as is currently implemented).
*   **Scope**: The `examples/end_to_end_pipeline_example.py` script serves as a primary E2E test. It verifies that all components (`EODHDClient`, `ArticleManager`, `TextProcessor`, `EmbeddingsGenerator`, `ChromaDBManager`) work together correctly to achieve the goal of fetching, processing, and storing news articles and their embeddings.
*   **Verification**: Successful execution of the script without errors, and potentially checking the state of the database and vector store afterwards for expected outcomes (e.g., number of articles processed, embeddings stored).
*   **Tools**: Python execution of the example script, potentially with assertions or checks on output files/databases.

## 2. Test Coverage

Strive for high test coverage for critical components and logic. While 100% coverage is not always practical, key functionalities, edge cases, and error handling paths should be thoroughly tested.

## 3. Mocking and Test Doubles

*   Use mocking libraries (e.g., `unittest.mock`) to isolate units under test and to simulate external dependencies like APIs (EODHD) or computationally intensive processes (embedding model loading) for unit tests.
*   For `ArticleManager` tests, an in-memory SQLite database will be used.
*   For `ChromaDBManager` tests, a transient/in-memory ChromaDB instance will be used.

## 4. Test Data

*   Create representative sample data for testing:
    *   Sample HTML content for `TextProcessor`.
    *   Sample article structures for `ArticleManager`.
    *   Mocked API responses for `EODHDClient`.
*   Include edge cases: empty inputs, very long inputs, malformed data (where appropriate to test error handling).

## 5. Running Tests

*   Tests can be run using the `pytest` command from the root of the project directory:
    ```bash
    python -m pytest
    ```
    or for specific files:
    ```bash
    python -m pytest tests/test_article_manager.py -v
    ```
*   Ensure necessary dependencies like NLTK's `punkt` are available in the test environment.

## 6. Continuous Integration (CI) - Future

*   Set up a CI pipeline (e.g., using GitHub Actions) to automatically run all tests on every push or pull request to the main branches. This helps catch regressions early.

## 7. Current Test Status

*   Unit tests for `ArticleManager` (`tests/test_article_manager.py`) are in place and passing.
*   Unit tests for `TextProcessor` (`tests/test_text_processor.py`) are in place and passing, including checks for text cleaning and chunking logic.
*   The `examples/end_to_end_pipeline_example.py` script acts as a comprehensive E2E test for the current set of low-level components.

This testing strategy will evolve as the project grows and new features (like the re-ranking class or high-level orchestrator) are added.
