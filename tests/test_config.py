"""
Tests for the Configuration module.
"""

import os
import pytest
from unittest.mock import patch

from financial_news_rag.config import Config


class TestConfig:
    """Tests for the Config class."""

    def test_init_loads_environment_variables(self):
        """Test that initialization loads environment variables."""
        with patch('financial_news_rag.config.load_dotenv') as mock_load_dotenv:
            with patch('financial_news_rag.config.Config._get_required_env', return_value="test_api_key"):
                Config()
            mock_load_dotenv.assert_called_once()
    
    def test_get_required_env_success(self):
        """Test that _get_required_env returns the value when the environment variable is set."""
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            config = Config()
            value = config._get_required_env("TEST_KEY")
            assert value == "test_value"
    
    def test_get_required_env_failure(self):
        """Test that _get_required_env raises ValueError when the environment variable is not set."""
        with patch.dict(os.environ, clear=True):
            config = Config()
            with pytest.raises(ValueError, match="Required environment variable 'TEST_KEY' is not set."):
                config._get_required_env("TEST_KEY")
    
    def test_get_env_with_default(self):
        """Test that _get_env returns the default value when the environment variable is not set."""
        with patch.dict(os.environ, clear=True):
            config = Config()
            value = config._get_env("TEST_KEY", "default_value")
            assert value == "default_value"
    
    def test_get_env_with_env_var(self):
        """Test that _get_env returns the environment variable value when it is set."""
        with patch.dict(os.environ, {"TEST_KEY": "env_value"}):
            config = Config()
            value = config._get_env("TEST_KEY", "default_value")
            assert value == "env_value"
    
    def test_get_method(self):
        """Test that the get method returns the config value by key."""
        # Create a config instance and manually set a test attribute
        with patch('financial_news_rag.config.Config._get_required_env', return_value="test_api_key"):
            config = Config()
            # Add a test attribute
            config._test_key = "test_value"
            assert config.get("test_key") == "test_value"
            assert config.get("nonexistent_key", "default") == "default"
    
    @patch.dict(os.environ, {
        "EODHD_API_KEY": "test_api_key",
        "EODHD_API_URL_OVERRIDE": "https://test.api.url",
        "EODHD_DEFAULT_TIMEOUT_OVERRIDE": "200",
        "EODHD_DEFAULT_MAX_RETRIES_OVERRIDE": "5",
        "EODHD_DEFAULT_BACKOFF_FACTOR_OVERRIDE": "2.0",
        "EODHD_DEFAULT_LIMIT_OVERRIDE": "100"
    })
    def test_eodhd_config_properties(self):
        """Test that the EODHD config properties return the correct values."""
        config = Config()
        assert config.eodhd_api_key == "test_api_key"
        assert config.eodhd_api_url == "https://test.api.url"
        assert config.eodhd_default_timeout == 200
        assert config.eodhd_default_max_retries == 5
        assert config.eodhd_default_backoff_factor == 2.0
        assert config.eodhd_default_limit == 100
    
    @patch('financial_news_rag.config.Config._get_required_env', return_value="test_api_key")
    def test_eodhd_config_default_values(self, mock_get_required_env):
        """Test that the EODHD config properties use default values when not overridden."""
        with patch.dict(os.environ, clear=True):
            config = Config()
            assert config.eodhd_api_key == "test_api_key"
            assert config.eodhd_api_url == "https://eodhd.com/api/news"
            assert config.eodhd_default_timeout == 100
            assert config.eodhd_default_max_retries == 3
            assert config.eodhd_default_backoff_factor == 1.5
            assert config.eodhd_default_limit == 50
    
    @patch.dict(os.environ, {
        "EODHD_API_KEY": "test_api_key",
        "EODHD_API_URL_OVERRIDE": "https://test.api.url",
        "EODHD_DEFAULT_TIMEOUT_OVERRIDE": "200",
        "EODHD_DEFAULT_MAX_RETRIES_OVERRIDE": "5",
        "EODHD_DEFAULT_BACKOFF_FACTOR_OVERRIDE": "2.0",
        "EODHD_DEFAULT_LIMIT_OVERRIDE": "100",
        "GEMINI_API_KEY": "test_gemini_api_key",
        "EMBEDDINGS_DEFAULT_MODEL": "custom-embedding-model",
        "EMBEDDINGS_DEFAULT_TASK_TYPE": "CUSTOM_TASK",
        "EMBEDDINGS_MODEL_DIMENSIONS": '{"custom-embedding-model": 1024, "text-embedding-004": 999}'
    })
    def test_all_config_properties(self):
        """Test that all config properties return the correct values when environment variables are set."""
        config = Config()
        # EODHD properties
        assert config.eodhd_api_key == "test_api_key"
        assert config.eodhd_api_url == "https://test.api.url"
        assert config.eodhd_default_timeout == 200
        assert config.eodhd_default_max_retries == 5
        assert config.eodhd_default_backoff_factor == 2.0
        assert config.eodhd_default_limit == 100
        
        # Gemini and embeddings properties
        assert config.gemini_api_key == "test_gemini_api_key"
        assert config.embeddings_default_model == "custom-embedding-model"
        assert config.embeddings_default_task_type == "CUSTOM_TASK"
        assert config.embeddings_model_dimensions == {
            "custom-embedding-model": 1024,
            "text-embedding-004": 999
        }
    
    @patch('financial_news_rag.config.Config._get_required_env')
    def test_gemini_config_default_values(self, mock_get_required_env):
        """Test that the Gemini and embeddings config properties use default values when not overridden."""
        # Configure the mock to return different values based on the key
        mock_get_required_env.side_effect = lambda key: {
            'EODHD_API_KEY': 'test_eodhd_api_key',
            'GEMINI_API_KEY': 'test_gemini_api_key'
        }[key]
        
        with patch.dict(os.environ, clear=True):
            config = Config()
            # Check Gemini API key
            assert config.gemini_api_key == "test_gemini_api_key"
            
            # Check embeddings default values
            assert config.embeddings_default_model == "text-embedding-004"
            assert config.embeddings_default_task_type == "SEMANTIC_SIMILARITY"
            assert config.embeddings_model_dimensions == {"text-embedding-004": 768}
    
    def test_embeddings_model_dimensions_json_error(self):
        """Test handling of invalid JSON in EMBEDDINGS_MODEL_DIMENSIONS."""
        with patch('financial_news_rag.config.Config._get_required_env') as mock_get_required_env:
            mock_get_required_env.side_effect = lambda key: {
                'EODHD_API_KEY': 'test_eodhd_api_key',
                'GEMINI_API_KEY': 'test_gemini_api_key'
            }[key]
            
            with patch('financial_news_rag.config.Config._get_env') as mock_get_env:
                mock_get_env.side_effect = lambda key, default: {
                    'EMBEDDINGS_MODEL_DIMENSIONS': 'invalid json'
                }.get(key, default)
                
                with patch('builtins.print') as mock_print:
                    config = Config()
                    
                    # Verify warning was printed
                    mock_print.assert_called_with(
                        "Warning: Failed to parse EMBEDDINGS_MODEL_DIMENSIONS as JSON. Using defaults."
                    )
                    
                    # Verify default values are used
                    assert config.embeddings_model_dimensions == {"text-embedding-004": 768}
