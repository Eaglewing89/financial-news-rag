"""
Configuration management for the Financial News RAG module.

This module handles loading environment variables, validating API keys,
and providing access to configuration parameters.
"""

import os
from typing import Dict, Optional, Any

from dotenv import load_dotenv


class Config:
    """
    Configuration manager for the Financial News RAG module.
    
    Handles loading API keys from environment variables and validation.
    """
    
    # Required API keys
    REQUIRED_KEYS = ["MARKETAUX_API_KEY", "GEMINI_API_KEY"]
    
    # Default configuration values
    DEFAULT_VALUES = {
        "MARKETAUX_RATE_LIMIT": 60,  # calls per minute
        "REFRESH_DAYS_BACK": 3,       # default days to look back when refreshing
        "RETRY_MAX_ATTEMPTS": 3,      # max retry attempts for API calls
        "RETRY_BACKOFF_FACTOR": 1.5,  # exponential backoff factor
    }
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            env_file: Optional path to .env file. If None, looks for .env in current directory.
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Initialize configuration dictionary
        self._config = {}
        
        # Load values from environment
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration values from environment variables."""
        # Load required API keys
        for key in self.REQUIRED_KEYS:
            self._config[key] = os.getenv(key)
        
        # Load default values, override with environment variables if provided
        for key, default_value in self.DEFAULT_VALUES.items():
            env_value = os.getenv(key)
            if env_value is not None:
                # Convert to appropriate type based on default value
                if isinstance(default_value, int):
                    self._config[key] = int(env_value)
                elif isinstance(default_value, float):
                    self._config[key] = float(env_value)
                elif isinstance(default_value, bool):
                    self._config[key] = env_value.lower() in ("true", "yes", "1")
                else:
                    self._config[key] = env_value
            else:
                self._config[key] = default_value
    
    def validate(self, raise_error: bool = True) -> bool:
        """
        Validate configuration, checking for required API keys.
        
        Args:
            raise_error: If True, raises ValueError for missing values.
                         If False, returns validation result as boolean.
        
        Returns:
            True if configuration is valid, False otherwise (if raise_error=False).
            
        Raises:
            ValueError: If raise_error=True and missing required API keys.
        """
        missing_keys = []
        
        # Check for missing required keys
        for key in self.REQUIRED_KEYS:
            if not self._config.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            if raise_error:
                raise ValueError(
                    f"Missing required API key(s): {', '.join(missing_keys)}. "
                    f"Please set these in your environment or .env file."
                )
            return False
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key to retrieve.
            default: Default value if key not found.
            
        Returns:
            Configuration value or default if not found.
        """
        return self._config.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary containing all configuration values.
        """
        return self._config.copy()
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value (for testing or runtime changes).
        
        Args:
            key: Configuration key to set.
            value: Value to set.
        """
        self._config[key] = value


# Create global config instance for module-level access
config = Config()

# Validate configuration on module import
try:
    config.validate()
except ValueError:
    # Do not raise here, allow app to continue but log warning
    # Users will get more specific errors when they try to use functionality
    # requiring the missing keys
    import warnings
    warnings.warn("Missing required API keys in configuration.")


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Global Config instance.
    """
    return config
