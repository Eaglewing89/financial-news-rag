"""
Unit tests for the EmbeddingsGenerator class.

These tests validate the functionality of the EmbeddingsGenerator class,
including initialization, embedding generation, error handling, and API interactions.
All API calls are mocked for isolated testing.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from financial_news_rag.embeddings import EmbeddingsGenerator
from tests.fixtures.sample_data import ChunkFactory


class TestEmbeddingsGeneratorInitialization:
    """Test suite for EmbeddingsGenerator initialization."""

    def test_init_with_valid_parameters(self, test_config):
        """Test initialization with valid parameters."""
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=test_config.embeddings_default_task_type,
        )

        assert generator.model_name == test_config.embeddings_default_model
        assert generator.default_task_type == test_config.embeddings_default_task_type
        assert generator.embedding_dim == 768
        assert hasattr(generator, "client")

    def test_init_with_default_task_type(self, test_config):
        """Test initialization with default task type."""
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
        )

        # Should use default task type
        assert generator.default_task_type is not None
        assert generator.embedding_dim == 768

    def test_init_missing_api_key_raises_error(self, test_config):
        """Test that initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="API key is required"):
            EmbeddingsGenerator(
                api_key="",
                model_name=test_config.embeddings_default_model,
                model_dimensions=test_config.embeddings_model_dimensions,
            )

        with pytest.raises(ValueError, match="API key is required"):
            EmbeddingsGenerator(
                api_key=None,
                model_name=test_config.embeddings_default_model,
                model_dimensions=test_config.embeddings_model_dimensions,
            )

    def test_init_unknown_model_raises_error(self, test_config):
        """Test that initialization with unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Embedding dimension.*is not defined"):
            EmbeddingsGenerator(
                api_key=test_config.gemini_api_key,
                model_name="unknown-model-name",
                model_dimensions=test_config.embeddings_model_dimensions,
            )

    def test_init_missing_model_dimensions_raises_error(self, test_config):
        """Test that missing model dimensions in config raises ValueError."""
        incomplete_dimensions = {"different-model": 512}

        with pytest.raises(ValueError, match="Embedding dimension.*is not defined"):
            EmbeddingsGenerator(
                api_key=test_config.gemini_api_key,
                model_name=test_config.embeddings_default_model,
                model_dimensions=incomplete_dimensions,
            )


class TestEmbeddingsGeneratorEmbeddingGeneration:
    """Test suite for embedding generation functionality."""

    @pytest.fixture
    def generator(self, test_config):
        """Create a test EmbeddingsGenerator instance."""
        return EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=test_config.embeddings_default_task_type,
        )

    @pytest.fixture
    def mock_embedding_vector(self):
        """Create a mock embedding vector."""
        return [0.1] * 768  # 768-dimensional vector

    @pytest.fixture
    def test_chunks(self):
        """Create test text chunks."""
        return ChunkFactory.create_chunks(count=3)

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_generate_embeddings_success(
        self, mock_client_cls, test_config, mock_embedding_vector, test_chunks
    ):
        """Test successful embedding generation for text chunks."""
        # Setup mock client and response
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create generator after mocking
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=test_config.embeddings_default_task_type,
        )

        # Mock the embedding response
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = mock_embedding_vector
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding_obj] * len(test_chunks)

        mock_client.models.embed_content.return_value = mock_result

        # Generate embeddings
        embeddings = generator.generate_embeddings(test_chunks)

        # Verify results
        assert len(embeddings) == len(test_chunks)
        assert all(len(emb) == 768 for emb in embeddings)
        assert all(isinstance(emb, list) for emb in embeddings)

        # Verify API was called for each chunk
        assert mock_client.models.embed_content.call_count == len(test_chunks)
        # Verify the last call used the correct model
        call_args = mock_client.models.embed_content.call_args
        assert call_args[1]["model"] == test_config.embeddings_default_model
        # Since chunks are processed individually, verify the last chunk was processed
        assert call_args[1]["contents"] == test_chunks[-1]
        # Verify task type is passed correctly in config
        assert (
            call_args[1]["config"].task_type == test_config.embeddings_default_task_type
        )

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_generate_embeddings_single_chunk(
        self, mock_client_cls, generator, mock_embedding_vector
    ):
        """Test embedding generation for a single text chunk."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock the _embed_single_text method directly to return a list
        # This ensures we're not relying on the mock's behavior for values
        with patch.object(
            generator, "_embed_single_text", return_value=mock_embedding_vector
        ):
            # Test with single chunk
            single_chunk = ["This is a single test chunk for embedding."]
            embeddings = generator.generate_embeddings(single_chunk)

            assert len(embeddings) == 1
            assert len(embeddings[0]) == 768
            assert isinstance(embeddings[0], list)

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_generate_embeddings_custom_task_type(
        self, mock_client_cls, test_config, mock_embedding_vector, test_chunks
    ):
        """Test embedding generation with custom task type."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create generator after mocking
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=test_config.embeddings_default_task_type,
        )

        # Mock the _embed_single_text method directly to return a list
        with patch.object(
            generator, "_embed_single_text", return_value=mock_embedding_vector
        ):
            # Generate embeddings with custom task type
            custom_task_type = "RETRIEVAL_DOCUMENT"
            embeddings = generator.generate_embeddings(
                test_chunks, task_type=custom_task_type
            )

            # Verify embeddings were generated
            assert len(embeddings) == len(test_chunks)
            assert all(len(emb) == 768 for emb in embeddings)

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_generate_embeddings_empty_input(self, mock_client_cls, generator):
        """Test embedding generation with empty input."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Test with empty list
        embeddings = generator.generate_embeddings([])
        assert embeddings == []

        # Verify no API call was made
        mock_client.models.embed_content.assert_not_called()

    def test_generate_embeddings_none_input(self, generator):
        """Test embedding generation with None input."""
        # The method handles None gracefully by returning empty list
        result = generator.generate_embeddings(None)
        assert result == []


class TestEmbeddingsGeneratorErrorHandling:
    """Test suite for error handling in embedding generation."""

    @pytest.fixture
    def generator(self, test_config):
        """Create a test EmbeddingsGenerator instance."""
        return EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
        )

    @pytest.fixture
    def test_chunks(self):
        """Create test text chunks."""
        return ["Test chunk 1", "Test chunk 2"]

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_api_error_handling(self, mock_client_cls, test_config, test_chunks):
        """Test handling of Google API errors."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create generator after mocking
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=test_config.embeddings_default_task_type,
        )

        # Mock the _embed_single_text method to raise an error and immediately fail
        from google.api_core.exceptions import GoogleAPIError

        def mock_embed_single_text(text, task_type=None):
            raise GoogleAPIError("API quota exceeded")

        # Patch the method to avoid retry delays
        with patch.object(
            generator, "_embed_single_text", side_effect=mock_embed_single_text
        ):
            # Should handle the error gracefully and return zero vectors
            embeddings = generator.generate_embeddings(test_chunks)

        assert len(embeddings) == len(test_chunks)
        # Check that zero vectors are returned for failed embeddings
        assert all(emb == [0.0] * 768 for emb in embeddings)

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_service_unavailable_handling(
        self, mock_client_cls, test_config, test_chunks
    ):
        """Test handling of service unavailable errors."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create generator after mocking
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=test_config.embeddings_default_task_type,
        )

        # Mock the _embed_single_text method to raise an error and immediately fail
        from google.api_core.exceptions import ServiceUnavailable

        def mock_embed_single_text(text, task_type=None):
            raise ServiceUnavailable("Service temporarily unavailable")

        # Patch the method to avoid retry delays
        with patch.object(
            generator, "_embed_single_text", side_effect=mock_embed_single_text
        ):
            # Should handle the error gracefully and return zero vectors
            embeddings = generator.generate_embeddings(test_chunks)

        assert len(embeddings) == len(test_chunks)
        # Check that zero vectors are returned for failed embeddings
        assert all(emb == [0.0] * 768 for emb in embeddings)

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_invalid_response_handling(self, mock_client_cls, test_config, test_chunks):
        """Test handling of invalid API responses."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create generator after mocking
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=test_config.embeddings_default_task_type,
        )

        # Mock invalid response (missing embeddings)
        mock_result = MagicMock()
        mock_result.embeddings = None
        mock_client.models.embed_content.return_value = mock_result

        # Should handle the error gracefully and return zero vectors
        embeddings = generator.generate_embeddings(test_chunks)
        assert len(embeddings) == len(test_chunks)
        # Check that zero vectors are returned for failed embeddings
        assert all(emb == [0.0] * 768 for emb in embeddings)

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_mismatched_embedding_count(
        self, mock_client_cls, test_config, test_chunks
    ):
        """Test handling when API returns wrong number of embeddings."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create generator after mocking
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=test_config.embeddings_default_task_type,
        )

        # Mock response with wrong number of embeddings
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = [0.1] * 768
        mock_result = MagicMock()
        mock_result.embeddings = [
            mock_embedding_obj
        ]  # Only one embedding for multiple chunks

        mock_client.models.embed_content.return_value = mock_result

        # Should handle the error gracefully and return zero vectors for failed chunks
        embeddings = generator.generate_embeddings(test_chunks)
        assert len(embeddings) == len(test_chunks)
        # First chunk might succeed, others will get zero vectors
        assert all(len(emb) == 768 for emb in embeddings)

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_invalid_embedding_dimensions(
        self, mock_client_cls, test_config, test_chunks
    ):
        """Test handling of embeddings with wrong dimensions."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create generator after mocking
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=test_config.embeddings_default_task_type,
        )

        # Mock embedding with wrong dimensions
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = [0.1] * 512  # Wrong dimension (should be 768)
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding_obj] * len(test_chunks)

        mock_client.models.embed_content.return_value = mock_result

        # Should still return embeddings (dimensions might vary by model)
        embeddings = generator.generate_embeddings(test_chunks)
        assert len(embeddings) == len(test_chunks)
        # The actual implementation returns whatever the API provides
        assert all(len(emb) == 512 for emb in embeddings)  # Returns actual dimensions


class TestEmbeddingsGeneratorBatchProcessing:
    """Test suite for batch processing capabilities."""

    @pytest.fixture
    def generator(self, test_config):
        """Create a test EmbeddingsGenerator instance."""
        return EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
        )

    @patch("financial_news_rag.embeddings.genai.Client")
    @patch("financial_news_rag.embeddings.time.sleep")
    def test_large_batch_processing(self, mock_sleep, mock_client_cls, test_config):
        """Test processing of large batches of text chunks."""
        # Mock time.sleep to eliminate waiting time
        mock_sleep.return_value = None

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create generator after mocking
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
        )

        # Create large batch - reduced from 50 to 20 chunks
        # Still tests batch processing but reduces execution time
        large_batch = [
            f"Test chunk number {i} with various content." for i in range(20)
        ]

        # Mock response for large batch
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = [0.1] * 768
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding_obj] * len(large_batch)

        mock_client.models.embed_content.return_value = mock_result

        embeddings = generator.generate_embeddings(large_batch)

        # Ensure sleep was called the right number of times
        assert mock_sleep.call_count == len(large_batch) - 1

        # Verify test still validates proper batch processing
        assert len(embeddings) == 20
        assert all(len(emb) == 768 for emb in embeddings)

    @patch("financial_news_rag.embeddings.genai.Client")
    def test_financial_content_embedding(self, mock_client_cls, test_config):
        """Test embedding generation for financial content."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Create generator after mocking
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
        )

        # Financial content chunks
        financial_chunks = [
            "Apple Inc. (AAPL) reported Q4 earnings of $1.50 per share.",
            "The Federal Reserve raised interest rates by 0.25% to combat inflation.",
            "Tesla's stock price surged 15% following strong delivery numbers.",
            "Goldman Sachs upgraded Microsoft to 'Buy' with a $350 price target.",
        ]

        # Mock response
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = [0.1] * 768
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding_obj] * len(financial_chunks)

        mock_client.models.embed_content.return_value = mock_result

        embeddings = generator.generate_embeddings(financial_chunks)

        assert len(embeddings) == len(financial_chunks)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 768 for emb in embeddings)


class TestEmbeddingsGeneratorModelConfiguration:
    """Test suite for different model configurations."""

    def test_different_model_dimensions(self, test_config):
        """Test initialization with different model dimensions."""
        # Test with different model
        custom_dimensions = {"text-embedding-004": 768, "custom-model": 1024}

        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name="custom-model",
            model_dimensions=custom_dimensions,
        )

        assert generator.embedding_dim == 1024
        assert generator.model_name == "custom-model"

    def test_model_name_formatting(self, test_config):
        """Test that model names are properly formatted for API calls."""
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
        )

        # The implementation should handle model name formatting
        assert generator.model_name == test_config.embeddings_default_model

    @pytest.mark.parametrize(
        "task_type",
        [
            "SEMANTIC_SIMILARITY",
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "CLASSIFICATION",
            "CLUSTERING",
        ],
    )
    def test_different_task_types(self, test_config, task_type):
        """Test initialization with different task types."""
        generator = EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
            task_type=task_type,
        )

        assert generator.default_task_type == task_type


class TestEmbeddingsGeneratorUtilityMethods:
    """Test suite for utility methods."""

    @pytest.fixture
    def generator(self, test_config):
        """Create a test EmbeddingsGenerator instance."""
        return EmbeddingsGenerator(
            api_key=test_config.gemini_api_key,
            model_name=test_config.embeddings_default_model,
            model_dimensions=test_config.embeddings_model_dimensions,
        )

    def test_embedding_dimensions_property(self, generator):
        """Test that embedding dimensions are correctly exposed."""
        assert generator.embedding_dim == 768
        assert isinstance(generator.embedding_dim, int)

    def test_model_name_property(self, generator, test_config):
        """Test that model name is correctly exposed."""
        assert generator.model_name == test_config.embeddings_default_model
        assert isinstance(generator.model_name, str)
