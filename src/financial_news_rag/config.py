"""
Configuration Module

This module provides a centralized configuration management system for the Financial News RAG application.
It loads configuration from environment variables and provides default values where appropriate.
"""

import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv


class Config:
    """
    Configuration manager for the Financial News RAG application.

    This class loads environment variables from a .env file and provides
    methods to access configuration values with appropriate defaults.
    """

    def __init__(self):
        """
        Initialize the configuration manager.

        Loads environment variables from .env file if it exists.
        """
        # Load environment variables from .env file
        load_dotenv()

        # EODHD API configuration
        self._eodhd_api_key = self._get_required_env("EODHD_API_KEY")
        self._eodhd_api_url = self._get_env(
            "EODHD_API_URL_OVERRIDE", "https://eodhd.com/api/news"
        )
        self._eodhd_default_timeout = int(
            self._get_env("EODHD_DEFAULT_TIMEOUT_OVERRIDE", "100")
        )
        self._eodhd_default_max_retries = int(
            self._get_env("EODHD_DEFAULT_MAX_RETRIES_OVERRIDE", "3")
        )
        self._eodhd_default_backoff_factor = float(
            self._get_env("EODHD_DEFAULT_BACKOFF_FACTOR_OVERRIDE", "1.5")
        )
        self._eodhd_default_limit = int(
            self._get_env("EODHD_DEFAULT_LIMIT_OVERRIDE", "50")
        )

        # Gemini API configuration
        self._gemini_api_key = self._get_required_env("GEMINI_API_KEY")

        # Embeddings configuration
        self._embeddings_default_model = self._get_env(
            "EMBEDDINGS_DEFAULT_MODEL", "text-embedding-004"
        )
        self._embeddings_default_task_type = self._get_env(
            "EMBEDDINGS_DEFAULT_TASK_TYPE", "SEMANTIC_SIMILARITY"
        )
        self._embeddings_default_rate_limit_delay = float(self._get_env(
            "EMBEDDINGS_DEFAULT_RATE_LIMIT_DELAY", "0.5")
        )

        # ReRanker configuration
        self._reranker_default_model = self._get_env(
            "RERANKER_DEFAULT_MODEL", "gemini-2.0-flash"
        )

        # TextProcessor configuration
        self._textprocessor_max_tokens_per_chunk = int(
            self._get_env("TEXTPROCESSOR_MAX_TOKENS_PER_CHUNK", "2048")
        )
        self._textprocessor_use_nltk = bool(
            self._get_env("TEXTPROCESSOR_USE_NLTK", False)
        )
        self._textprocessor_nltk_auto_download = bool(
            self._get_env("TEXTPROCESSOR_NLTK_AUTO_DOWNLOAD", False)
        )

        # Database configuration
        self._database_path = self._get_env(
            "DATABASE_PATH_OVERRIDE", os.path.join(os.getcwd(), "financial_news.db")
        )

        # ChromaDB configuration
        self._chroma_default_collection_name = self._get_env(
            "CHROMA_DEFAULT_COLLECTION_NAME", "financial_news_embeddings"
        )
        self._chroma_default_persist_directory = self._get_env(
            "CHROMA_DEFAULT_PERSIST_DIRECTORY", os.path.join(os.getcwd(), "chroma_db")
        )

        # Model dimensions as a dictionary with model names as keys and dimensions as values
        default_dimensions = {"text-embedding-004": 768}
        try:
            # Try to load model dimensions from environment variable as JSON string
            env_dimensions = self._get_env("EMBEDDINGS_MODEL_DIMENSIONS", "")
            if env_dimensions:
                custom_dimensions = json.loads(env_dimensions)
                # Merge with default dimensions, custom values override defaults
                self._embeddings_model_dimensions = {
                    **default_dimensions,
                    **custom_dimensions,
                }
            else:
                self._embeddings_model_dimensions = default_dimensions
        except json.JSONDecodeError:
            # If JSON parsing fails, log a warning and use defaults
            print(
                f"Warning: Failed to parse EMBEDDINGS_MODEL_DIMENSIONS as JSON. Using defaults."
            )
            self._embeddings_model_dimensions = default_dimensions

    def _get_required_env(self, key: str) -> str:
        """
        Get a required environment variable.

        Args:
            key: The name of the environment variable.

        Returns:
            The value of the environment variable.

        Raises:
            ValueError: If the environment variable is not set.
        """
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' is not set.")
        return value

    def _get_env(self, key: str, default: str) -> str:
        """
        Get an environment variable with a default value.

        Args:
            key: The name of the environment variable.
            default: The default value to use if the environment variable is not set.

        Returns:
            The value of the environment variable or the default value.
        """
        return os.getenv(key, default)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: The configuration key.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value or the default value.
        """
        # Convert the key to the attribute name format
        attr_name = f"_{key.lower()}"

        # Return the attribute if it exists, otherwise return the default
        return getattr(self, attr_name, default)

    @property
    def eodhd_api_key(self) -> str:
        """Get the EODHD API key."""
        return self._eodhd_api_key

    @property
    def eodhd_api_url(self) -> str:
        """Get the EODHD API URL."""
        return self._eodhd_api_url

    @property
    def eodhd_default_timeout(self) -> int:
        """Get the default timeout for EODHD API requests."""
        return self._eodhd_default_timeout

    @property
    def eodhd_default_max_retries(self) -> int:
        """Get the default maximum number of retries for EODHD API requests."""
        return self._eodhd_default_max_retries

    @property
    def eodhd_default_backoff_factor(self) -> float:
        """Get the default backoff factor for EODHD API requests."""
        return self._eodhd_default_backoff_factor

    @property
    def eodhd_default_limit(self) -> int:
        """Get the default limit for EODHD API requests."""
        return self._eodhd_default_limit

    @property
    def gemini_api_key(self) -> str:
        """Get the Gemini API key."""
        return self._gemini_api_key

    @property
    def embeddings_default_model(self) -> str:
        """Get the default model for text embeddings."""
        return self._embeddings_default_model

    @property
    def embeddings_default_task_type(self) -> str:
        """Get the default task type for text embeddings."""
        return self._embeddings_default_task_type

    @property
    def embeddings_model_dimensions(self) -> Dict[str, int]:
        """Get the dimensions for each embedding model."""
        return self._embeddings_model_dimensions
    
    @property
    def embeddings_default_rate_limit_delay(self) -> float:
        """Get the default rate limit delay for text embeddings."""
        return self._embeddings_default_rate_limit_delay

    @property
    def reranker_default_model(self) -> str:
        """Get the default model for the ReRanker."""
        return self._reranker_default_model

    @property
    def textprocessor_max_tokens_per_chunk(self) -> int:
        """Get the maximum tokens per chunk for the TextProcessor."""
        return self._textprocessor_max_tokens_per_chunk
    
    @property
    def textprocessor_use_nltk(self) -> bool:
        """Get boolean use nltk for the TextProcessor."""
        return self._textprocessor_use_nltk
    
    @property
    def textprocessor_nltk_auto_download(self) -> bool:
        """Get boolean auto download nltk data for the TextProcessor."""
        return self._textprocessor_nltk_auto_download

    @property
    def database_path(self) -> str:
        """Get the path to the SQLite database file."""
        return self._database_path

    @property
    def chroma_default_collection_name(self) -> str:
        """Get the default collection name for ChromaDB."""
        return self._chroma_default_collection_name

    @property
    def chroma_default_persist_directory(self) -> str:
        """Get the default persistence directory for ChromaDB."""
        return self._chroma_default_persist_directory

    @property
    def chroma_default_embedding_dimension(self) -> int:
        """Get the default embedding dimension for ChromaDB based on the default embedding model."""
        return self._embeddings_model_dimensions.get(self.embeddings_default_model, 768)


# Create a global config instance for easy import
config = Config()
