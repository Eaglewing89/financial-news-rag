"""
Tests for the embeddings generator module.

These tests validate the functionality of the EmbeddingsGenerator class,
including API key loading, error handling, and embedding generation.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from google.api_core.exceptions import GoogleAPIError, ServiceUnavailable

from financial_news_rag.embeddings import EmbeddingsGenerator


class TestEmbeddingsGenerator(unittest.TestCase):
    """Test the EmbeddingsGenerator class functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.orig_env = os.environ.copy()
        
        # Set mock API key for testing
        os.environ['GEMINI_API_KEY'] = 'test_api_key_123'
        
        # Sample test data
        self.test_chunks = [
            "This is the first test chunk for embedding generation.",
            "This is the second test chunk for embedding testing. It contains more text.",
            "And this is the third chunk with even more content to test embeddings generation."
        ]
        
        # Mock embedding vector (dimensions for text-embedding-004 = 768)
        self.mock_embedding = [0.01] * 768
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.orig_env)
    
    def test_initialization_with_env_var(self):
        """Test initialization with API key from environment variable."""
        generator = EmbeddingsGenerator()
        self.assertEqual(generator.model_name, "text-embedding-004")
        # Can't directly check client config because it's private, but initialization
        # should complete without errors
    
    def test_initialization_with_explicit_key(self):
        """Test initialization with explicitly provided API key."""
        generator = EmbeddingsGenerator(api_key="explicit_test_key")
        self.assertEqual(generator.model_name, "text-embedding-004")
        # Initialization should complete without errors
    
    @patch('financial_news_rag.embeddings.load_dotenv')
    def test_missing_api_key(self, mock_load_dotenv):
        """Test that ValueError is raised when API key is missing."""
        # Mock load_dotenv to do nothing (not load from .env file)
        mock_load_dotenv.return_value = None
        
        # Remove API key from environment
        os.environ.pop('GEMINI_API_KEY', None)
        
        # Attempt to initialize without providing key
        with self.assertRaises(ValueError):
            EmbeddingsGenerator()
    
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
        generator = EmbeddingsGenerator()
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
        generator = EmbeddingsGenerator()
        
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
        generator = EmbeddingsGenerator()
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
        generator = EmbeddingsGenerator()
        
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
        generator = EmbeddingsGenerator()
        
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


if __name__ == '__main__':
    unittest.main()
