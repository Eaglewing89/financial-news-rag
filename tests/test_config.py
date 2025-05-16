"""
Tests for the configuration module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from financial_news_rag.config import Config, get_config


class TestConfig:
    """Tests for the Config class."""
    
    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        with patch.dict(os.environ, {"MARKETAUX_API_KEY": "test_key", "GEMINI_API_KEY": "test_key"}):
            config = Config()
            
            # Check that required keys are loaded
            assert config.get("MARKETAUX_API_KEY") == "test_key"
            assert config.get("GEMINI_API_KEY") == "test_key"
            
            # Check that default values are set
            assert config.get("MARKETAUX_RATE_LIMIT") == 60
            assert config.get("REFRESH_DAYS_BACK") == 3
            assert config.get("RETRY_MAX_ATTEMPTS") == 3
            assert config.get("RETRY_BACKOFF_FACTOR") == 1.5
    
    def test_env_file_loading(self):
        """Test loading from a specific env file."""
        # Mock load_dotenv to track if it's called with the right file
        with patch('financial_news_rag.config.load_dotenv') as mock_load_dotenv:
            Config(env_file="test.env")
            mock_load_dotenv.assert_called_once_with("test.env")
    
    def test_environment_override(self):
        """Test that environment variables override defaults."""
        env_vars = {
            "MARKETAUX_API_KEY": "test_key",
            "GEMINI_API_KEY": "test_key",
            "MARKETAUX_RATE_LIMIT": "30",
            "REFRESH_DAYS_BACK": "5",
            "RETRY_MAX_ATTEMPTS": "5"
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config()
            
            assert config.get("MARKETAUX_RATE_LIMIT") == 30
            assert config.get("REFRESH_DAYS_BACK") == 5
            assert config.get("RETRY_MAX_ATTEMPTS") == 5
            # Default value for non-overridden setting
            assert config.get("RETRY_BACKOFF_FACTOR") == 1.5
    
    def test_validate_success(self):
        """Test validation with all required keys present."""
        with patch.dict(os.environ, {"MARKETAUX_API_KEY": "test_key", "GEMINI_API_KEY": "test_key"}):
            config = Config()
            assert config.validate(raise_error=False) is True
    
    def test_validate_missing_key(self):
        """Test validation with missing required key."""
        # Missing GEMINI_API_KEY
        with patch.dict(os.environ, {"MARKETAUX_API_KEY": "test_key"}):
            # Create a config with a deliberately empty GEMINI_API_KEY
            config = Config()
            config.set("GEMINI_API_KEY", "")  # Explicitly set empty value
            
            # Test with raise_error=False
            assert config.validate(raise_error=False) is False
            
            # Test with raise_error=True
            with pytest.raises(ValueError) as excinfo:
                config.validate(raise_error=True)
            assert "GEMINI_API_KEY" in str(excinfo.value)
    
    def test_get_with_default(self):
        """Test get method with a default value for missing key."""
        config = Config()
        assert config.get("NON_EXISTENT_KEY", "default_value") == "default_value"
    
    def test_get_all(self):
        """Test get_all method returns all config values."""
        with patch.dict(os.environ, {"MARKETAUX_API_KEY": "test_key", "GEMINI_API_KEY": "test_key"}):
            config = Config()
            all_config = config.get_all()
            
            assert isinstance(all_config, dict)
            assert "MARKETAUX_API_KEY" in all_config
            assert "GEMINI_API_KEY" in all_config
            assert "RETRY_MAX_ATTEMPTS" in all_config
    
    def test_set(self):
        """Test set method to change a config value."""
        config = Config()
        config.set("CUSTOM_KEY", "custom_value")
        
        assert config.get("CUSTOM_KEY") == "custom_value"


def test_get_config():
    """Test get_config returns the global config instance."""
    config = get_config()
    assert config is not None
    assert isinstance(config, Config)
