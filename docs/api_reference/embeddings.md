[← Back to Main Documentation Index](../index.md)  

[← API reference Index](./index.md)

# Embeddings Generator (`embeddings.py`)

The `embeddings.py` module provides the `EmbeddingsGenerator` class, which is responsible for generating vector embeddings for text chunks. It utilizes Google's `text-embedding-004` model through the Gemini API.

## `EmbeddingsGenerator` Class

This class handles the initialization of the Gemini API client, the generation of embeddings, and includes error handling with retry logic for API calls.

### Initialization

```python
class EmbeddingsGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        model_dimensions: Dict[str, int],
        task_type: str = "SEMANTIC_SIMILARITY",
    ):
        """
        Initialize the EmbeddingsGenerator with API key and model settings.
        ...
        """
        # ...
```

The constructor sets up the `EmbeddingsGenerator`.

**Parameters:**

*   `api_key` (str): Your Gemini API key. This is **required**.
*   `model_name` (str): The name of the embedding model to be used (e.g., `"text-embedding-004"`).
*   `model_dimensions` (Dict[str, int]): A dictionary mapping model names to their output embedding dimensions. This is used to validate the `model_name` and to know the expected dimension of the output vectors (e.g., `{"text-embedding-004": 768}`).
*   `task_type` (str, optional): The default task type for which the embeddings will be generated. This influences how the model optimizes the embeddings. Defaults to `"SEMANTIC_SIMILARITY"`. Other common values might include `"RETRIEVAL_DOCUMENT"`, `"RETRIEVAL_QUERY"`, etc., depending on the model's capabilities.

**Raises:**

*   `ValueError`: If the `api_key` is not provided or if the specified `model_name` is not found in the `model_dimensions` dictionary.

### Methods

#### `generate_embeddings(text_chunks: List[str], task_type: Optional[str] = None) -> List[List[float]]`

```python
def generate_embeddings(
    self, text_chunks: List[str], task_type: Optional[str] = None
) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks.
    ...
    Returns:
        A list of embedding vectors, where each vector corresponds to a text chunk
    """
    # ...
```
Generates an embedding vector for each text chunk in the input list.

**Parameters:**

*   `text_chunks` (List[str]): A list of strings, where each string is a text chunk to be embedded.
*   `task_type` (Optional[str], optional): The task type for these specific embeddings. If not provided, the `default_task_type` set during initialization is used.

**Returns:**

*   `List[List[float]]`: A list of embedding vectors. Each inner list contains floating-point numbers representing the embedding for the corresponding input text chunk. If a chunk fails to embed after retries, a zero vector (a list of `0.0`s with the length of `self.embedding_dim`) is returned in its place to maintain the list's structure.

**Error Handling & Retries:**

The internal `_embed_single_text` method, which `generate_embeddings` calls for each chunk, is decorated with `@tenacity.retry`. This provides resilience against transient API errors:

*   It retries on `GoogleAPIError`, `ServiceUnavailable`, and `ConnectionError` exceptions.
*   It stops after a maximum of 5 attempts.
*   It uses exponential backoff for waiting between retries (multiplier of 1, min 2s, max 60s).
*   If all retries fail, the original exception is re-raised by `_embed_single_text`, which is then caught by `generate_embeddings`.

`generate_embeddings` itself will log an error for any chunk that ultimately fails and will insert a zero vector for that chunk in the results. It logs the total number of successful and failed embeddings at the end.

#### `generate_and_verify_embeddings(text_chunks: List[str], task_type: Optional[str] = None) -> Dict[str, Union[List[List[float]], bool]]`

```python
def generate_and_verify_embeddings(
    self, text_chunks: List[str], task_type: Optional[str] = None
) -> Dict[str, Union[List[List[float]], bool]]:
    """
    Generate embeddings for a list of text chunks and verify their validity.
    ...
    Returns:
        A dictionary containing:
            - "embeddings": List of embedding vectors (including any zero vectors for failures)
            - "all_valid": Boolean indicating if all embeddings are valid (not zero vectors)
    """
    # ...
```
This method first calls `generate_embeddings` to get the embedding vectors for the provided text chunks. It then checks if any of the returned embeddings are zero vectors (which indicates a failure for that specific chunk).

**Parameters:**

*   `text_chunks` (List[str]): A list of strings, where each string is a text chunk to be embedded.
*   `task_type` (Optional[str], optional): The task type for these specific embeddings. If not provided, the `default_task_type` set during initialization is used.

**Returns:**

*   `Dict[str, Union[List[List[float]], bool]]`: A dictionary with two keys:
    *   `"embeddings"` (List[List[float]]): The list of generated embedding vectors (same as the output of `generate_embeddings`).
    *   `"all_valid"` (bool): `True` if all embeddings in the list are non-zero vectors (i.e., no failures occurred or the input `text_chunks` was empty). `False` if at least one embedding is a zero vector.
