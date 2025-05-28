"""
Embeddings Generator for Financial News RAG.

This module provides a class for generating embeddings from text chunks
using Google's text-embedding-004 model via the Gemini API.
"""

import logging
import time
from typing import Dict, List, Optional, Union

from google import genai
from google.api_core.exceptions import GoogleAPIError, ServiceUnavailable
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configure module logger
logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """
    A class for generating embedding vectors for text chunks using Google's text-embedding-004 model.

    This class is responsible for:
    - Initializing a Gemini API client
    - Generating embeddings for text chunks
    - Handling API errors gracefully with retry logic
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        model_dimensions: Dict[str, int],
        task_type: str = "SEMANTIC_SIMILARITY",
        rate_limit_delay: float = 0.5
    ):
        """
        Initialize the EmbeddingsGenerator with API key and model settings.

        Args:
            api_key: The Gemini API key.
            model_name: The name of the embedding model to use.
            model_dimensions: Dictionary mapping model names to their output dimensions.
            task_type: The default task type for embeddings.

        Raises:
            ValueError: If the API key is not provided or the model is not in model_dimensions.
        """
        if not api_key:
            raise ValueError("Gemini API key is required.")

        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.default_task_type = task_type
        self.rate_limit_delay = rate_limit_delay

        # Set embedding dimension based on model, fail if unknown
        if model_name not in model_dimensions:
            raise ValueError(
                f"Embedding dimension for model '{model_name}' is not defined. "
                f"Please add it to the model_dimensions configuration."
            )
        self.embedding_dim = model_dimensions[model_name]
        logger.info(
            f"EmbeddingsGenerator initialized with model: {model_name} (dim={self.embedding_dim})"
        )

    @retry(
        retry=retry_if_exception_type(
            (GoogleAPIError, ServiceUnavailable, ConnectionError)
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def _embed_single_text(
        self, text: str, task_type: Optional[str] = None
    ) -> List[float]:
        """
        Generate an embedding for a single text chunk.

        Args:
            text: The text to embed
            task_type: The type of task for which the embedding will be used

        Returns:
            A list of floating-point values representing the embedding vector

        Raises:
            GoogleAPIError: If there's an issue with the API call
            ValueError: If the text is empty or the embedding fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        # Use provided task_type or default
        task_type = task_type or self.default_task_type

        # Configure embedding request
        config = types.EmbedContentConfig(task_type=task_type)

        start_time = time.time()
        try:
            # Call the API to generate embedding
            result = self.client.models.embed_content(
                model=self.model_name, contents=text, config=config
            )

            # Extract and return the embedding values
            embedding_vector = result.embeddings[0].values

            # Log successful embedding
            duration = time.time() - start_time
            logger.debug(
                f"Successfully generated embedding in {duration:.2f}s. "
                f"Vector dimension: {len(embedding_vector)}"
            )

            return embedding_vector

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error generating embedding after {duration:.2f}s: {str(e)}")

            # Re-raise appropriate exceptions for retry logic
            if isinstance(e, (GoogleAPIError, ServiceUnavailable, ConnectionError)):
                logger.info(f"Retrying due to {type(e).__name__}: {str(e)}")
                raise
            else:
                # For other exceptions, wrap in ValueError
                raise ValueError(f"Failed to generate embedding: {str(e)}") from e

    def generate_embeddings(
        self, text_chunks: List[str], task_type: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            text_chunks: List of text chunks to embed
            task_type: The type of task for which the embeddings will be used

        Returns:
            A list of embedding vectors, where each vector corresponds to a text chunk

        Note:
            If `text_chunks` is empty, a warning is logged, and an empty list is returned.
        """
        if not text_chunks:
            logger.warning("No text chunks provided for embedding generation")
            return []

        embeddings = []
        errors = 0
        logger.info(f"Generating embeddings for {len(text_chunks)} text chunks")
        # Process each chunk and generate its embedding
        for i, chunk in enumerate(text_chunks):
            try:
                # Add slight delay between requests to avoid rate limiting
                time.sleep(self.rate_limit_delay)
                embedding = self._embed_single_text(chunk, task_type)
                embeddings.append(embedding)
                # Log progress periodically
                if (i + 1) % 10 == 0 or (i + 1) == len(text_chunks):
                    logger.info(f"Processed {i+1}/{len(text_chunks)} chunks")
            except Exception as e:
                logger.error(
                    f"Failed to generate embedding for chunk {i+1}/{len(text_chunks)}: {str(e)}"
                )
                errors += 1
                # Append a zero vector as a placeholder for failed embeddings
                # This ensures the output list matches the input list in length
                embeddings.append([0.0] * self.embedding_dim)
        # Report completion stats
        success_count = len(text_chunks) - errors
        logger.info(
            f"Embedding generation complete. Success: {success_count}/{len(text_chunks)}"
        )
        if errors > 0:
            logger.warning(f"Failed to generate embeddings for {errors} chunks")
        return embeddings

    def generate_and_verify_embeddings(
        self, text_chunks: List[str], task_type: Optional[str] = None
    ) -> Dict[str, Union[List[List[float]], bool]]:
        """
        Generate embeddings for a list of text chunks and verify their validity.

        Args:
            text_chunks: List of text chunks to embed
            task_type: The type of task for which the embeddings will be used

        Returns:
            A dictionary containing:
                - "embeddings": List of embedding vectors (including any zero vectors for failures)
                - "all_valid": Boolean indicating if all embeddings are valid (not zero vectors)
                  True if no zero vectors are present or if chunk_embeddings is empty
        """
        # Generate embeddings using the existing method
        chunk_embeddings = self.generate_embeddings(text_chunks, task_type)

        # Check if any embedding is a zero vector (all elements are 0.0)
        all_valid = True
        if chunk_embeddings:  # Only check if we have embeddings
            all_valid = not any(
                all(val == 0.0 for val in emb) for emb in chunk_embeddings
            )

            if not all_valid:
                logger.warning(
                    "One or more embeddings are zero vectors, indicating failures"
                )

        return {"embeddings": chunk_embeddings, "all_valid": all_valid}
