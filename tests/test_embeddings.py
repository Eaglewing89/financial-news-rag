"""
Tests for the embeddings generator module.

These tests validate the functionality of the EmbeddingsGenerator class,
including error handling and embedding generation.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from google.api_core.exceptions import GoogleAPIError, ServiceUnavailable

from financial_news_rag.embeddings import EmbeddingsGenerator


class TestEmbeddingsGenerator(unittest.TestCase):
    """Test the EmbeddingsGenerator class functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Sample test data
        self.test_chunks = [
            "This is the first test chunk for embedding generation.",
            "This is the second test chunk for embedding testing. It contains more text.",
            "And this is the third chunk with even more content to test embeddings generation."
        ]
        
        # Test API key
        self.test_api_key = "test_api_key_123"
        
        # Test model name
        self.test_model_name = "text-embedding-004"
        
        # Test task type
        self.test_task_type = "SEMANTIC_SIMILARITY"
        
        # Test model dimensions
        self.test_model_dimensions = {"text-embedding-004": 768}
        
        # Mock embedding vector (dimensions for text-embedding-004 = 768)
        self.mock_embedding = [0.01] * 768
    
    def test_initialization_with_params(self):
        """Test initialization with API key and model parameters."""
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions,
            task_type=self.test_task_type
        )
        self.assertEqual(generator.model_name, self.test_model_name)
        self.assertEqual(generator.default_task_type, self.test_task_type)
        self.assertEqual(generator.embedding_dim, 768)
        # Can't directly check client config because it's private, but initialization
        # should complete without errors
    
    def test_missing_api_key(self):
        """Test that ValueError is raised when API key is empty."""
        # Attempt to initialize with empty API key
        with self.assertRaises(ValueError):
            EmbeddingsGenerator(
                api_key="", 
                model_name=self.test_model_name, 
                model_dimensions=self.test_model_dimensions
            )
    
    def test_unknown_model(self):
        """Test that ValueError is raised when model is not in model_dimensions."""
        # Attempt to initialize with unknown model
        with self.assertRaises(ValueError):
            EmbeddingsGenerator(
                api_key=self.test_api_key, 
                model_name="unknown-model", 
                model_dimensions=self.test_model_dimensions
            )
    
    @patch('financial_news_rag.embeddings.genai.Client')
    def test_generate_embeddings_success(self, mock_client_cls):
        """Test successful embedding generation for text chunks."""
        # Set up the mock response
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Mock embeddings response
        mock_result = MagicMock()
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = self.mock_embedding
        mock_result.embeddings = [mock_embedding_obj]
        
        # Configure mock client to return the mock response
        mock_client.models.embed_content.return_value = mock_result
        
        # Create generator and generate embeddings
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions
        )
        embeddings = generator.generate_embeddings(self.test_chunks)
        
        # Verify results
        self.assertEqual(len(embeddings), len(self.test_chunks))
        for embedding in embeddings:
            self.assertEqual(len(embedding), 768)  # text-embedding-004 dimension
            self.assertEqual(embedding, self.mock_embedding)
        
        # Verify API was called correctly for each chunk
        self.assertEqual(mock_client.models.embed_content.call_count, len(self.test_chunks))
    
    @patch('financial_news_rag.embeddings.genai.Client')
    def test_retry_on_api_error(self, mock_client_cls):
        """Test retry logic when API call fails with GoogleAPIError."""
        # Set up the mock client
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Configure mock to raise exception then succeed
        mock_client.models.embed_content.side_effect = [
            GoogleAPIError("Test API error"),  # First attempt fails
            MagicMock(embeddings=[MagicMock(values=self.mock_embedding)])  # Second succeeds
        ]
        
        # Create generator and call the method that should trigger retry
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions
        )
        
        # Patch the retry decorator to only make 2 attempts (faster test)
        with patch('financial_news_rag.embeddings.stop_after_attempt', return_value=lambda f: f):
            with patch('financial_news_rag.embeddings.wait_exponential', return_value=lambda f: f):
                result = generator._embed_single_text("Test text")
        
        # Verify results
        self.assertEqual(result, self.mock_embedding)
        self.assertEqual(mock_client.models.embed_content.call_count, 2)
    
    @patch('financial_news_rag.embeddings.genai.Client')
    def test_empty_text_chunks(self, mock_client_cls):
        """Test handling of empty text chunks list."""
        # Create generator and call with empty list
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions
        )
        result = generator.generate_embeddings([])
        
        # Verify results
        self.assertEqual(result, [])
        
        # API should not be called
        mock_client = mock_client_cls.return_value
        mock_client.models.embed_content.assert_not_called()
    
    @patch('financial_news_rag.embeddings.genai.Client')
    def test_empty_text_handling(self, mock_client_cls):
        """Test handling of empty text."""
        # Create generator
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions
        )
        
        # API should not be called for empty string
        with self.assertRaises(ValueError):
            generator._embed_single_text("")
        
        mock_client = mock_client_cls.return_value
        mock_client.models.embed_content.assert_not_called()
    
    @patch('financial_news_rag.embeddings.genai.Client')
    def test_partial_failure_handling(self, mock_client_cls):
        """Test handling when some chunks fail to embed."""
        # Set up the mock client
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Configure mock to succeed for first chunk, fail for second, succeed for third
        success_response = MagicMock(embeddings=[MagicMock(values=self.mock_embedding)])
        
        mock_client.models.embed_content.side_effect = [
            success_response,  # First chunk succeeds
            ValueError("Test embedding failure"),  # Second chunk fails
            success_response   # Third chunk succeeds
        ]
        
        # Create generator and call with test chunks
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions
        )
        
        # Patch the retry decorator to avoid actual retries in test
        with patch('financial_news_rag.embeddings.retry', lambda **kwargs: lambda f: f):
            embeddings = generator.generate_embeddings(self.test_chunks)
        
        # Verify results
        self.assertEqual(len(embeddings), 3)  # Still should have 3 results
        
        # First and third should be normal, second should be zeros
        self.assertEqual(embeddings[0], self.mock_embedding)
        self.assertEqual(embeddings[1], [0.0] * 768)  # Zero vector for failed embedding
        self.assertEqual(embeddings[2], self.mock_embedding)
        
        # API should be called for all chunks
        self.assertEqual(mock_client.models.embed_content.call_count, 3)
    
    @patch('financial_news_rag.embeddings.EmbeddingsGenerator.generate_embeddings')
    def test_generate_and_verify_embeddings_success(self, mock_generate_embeddings):
        """Test successful verification of embeddings (no zero vectors)."""
        # Create a list of valid embeddings (non-zero vectors)
        valid_embeddings = [
            [0.1, 0.2] * 384,  # Non-zero vector
            [0.3, 0.4] * 384,  # Non-zero vector
            [0.5, 0.6] * 384   # Non-zero vector
        ]
        
        # Configure mock to return valid embeddings
        mock_generate_embeddings.return_value = valid_embeddings
        
        # Create generator and call with test chunks
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions
        )
        result = generator.generate_and_verify_embeddings(self.test_chunks)
        
        # Verify results
        self.assertEqual(result["embeddings"], valid_embeddings)
        self.assertTrue(result["all_valid"])
        
        # Verify generate_embeddings was called correctly
        mock_generate_embeddings.assert_called_once_with(self.test_chunks, None)
    
    @patch('financial_news_rag.embeddings.EmbeddingsGenerator.generate_embeddings')
    def test_generate_and_verify_embeddings_partial_failure(self, mock_generate_embeddings):
        """Test verification of embeddings with some zero vectors."""
        # Create a list of embeddings with one zero vector
        mixed_embeddings = [
            [0.1, 0.2] * 384,          # Non-zero vector
            [0.0] * 768,               # Zero vector (failure)
            [0.5, 0.6] * 384           # Non-zero vector
        ]
        
        # Configure mock to return mixed embeddings
        mock_generate_embeddings.return_value = mixed_embeddings
        
        # Create generator and call with test chunks
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions
        )
        result = generator.generate_and_verify_embeddings(self.test_chunks)
        
        # Verify results
        self.assertEqual(result["embeddings"], mixed_embeddings)
        self.assertFalse(result["all_valid"])
        
        # Verify generate_embeddings was called correctly
        mock_generate_embeddings.assert_called_once_with(self.test_chunks, None)
    
    @patch('financial_news_rag.embeddings.EmbeddingsGenerator.generate_embeddings')
    def test_generate_and_verify_embeddings_all_failed(self, mock_generate_embeddings):
        """Test verification of embeddings with all zero vectors."""
        # Create a list of all zero vectors
        failed_embeddings = [
            [0.0] * 768,  # Zero vector
            [0.0] * 768   # Zero vector
        ]
        
        # Configure mock to return zero vectors
        mock_generate_embeddings.return_value = failed_embeddings
        
        # Create generator and call with test chunks
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions
        )
        result = generator.generate_and_verify_embeddings(self.test_chunks[:2])  # Use only 2 chunks
        
        # Verify results
        self.assertEqual(result["embeddings"], failed_embeddings)
        self.assertFalse(result["all_valid"])
        
        # Verify generate_embeddings was called correctly
        mock_generate_embeddings.assert_called_once_with(self.test_chunks[:2], None)
    
    @patch('financial_news_rag.embeddings.EmbeddingsGenerator.generate_embeddings')
    def test_generate_and_verify_embeddings_empty_input(self, mock_generate_embeddings):
        """Test verification of embeddings with empty input."""
        # Configure mock to return empty list
        mock_generate_embeddings.return_value = []
        
        # Create generator and call with empty list
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions
        )
        result = generator.generate_and_verify_embeddings([])
        
        # Verify results
        self.assertEqual(result["embeddings"], [])
        self.assertTrue(result["all_valid"])
        
        # Verify generate_embeddings was called correctly
        mock_generate_embeddings.assert_called_once_with([], None)
    
    def test_custom_task_type(self):
        """Test initialization with custom task type."""
        custom_task = "CUSTOM_TASK_TYPE"
        generator = EmbeddingsGenerator(
            api_key=self.test_api_key, 
            model_name=self.test_model_name, 
            model_dimensions=self.test_model_dimensions,
            task_type=custom_task
        )
        
        self.assertEqual(generator.default_task_type, custom_task)


if __name__ == '__main__':
    unittest.main()
