"""
Utility functions for the Financial News RAG module.

Provides common utilities used across the module.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union

# Configure module-level logger
logger = logging.getLogger("financial_news_rag")


def format_date(date_value: Union[str, datetime], output_format: str = "iso") -> str:
    """
    Format a date string or datetime object to a consistent format.
    
    Args:
        date_value: Date as string or datetime object.
        output_format: Output format, one of: "iso", "date_only", "ymd", or custom strftime format.
        
    Returns:
        Formatted date string.
    """
    if isinstance(date_value, str):
        try:
            # Try ISO format first
            date_obj = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
        except ValueError:
            try:
                # Try other common formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
                    try:
                        date_obj = datetime.strptime(date_value, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Could not parse date: {date_value}")
            except Exception as e:
                raise ValueError(f"Could not parse date: {date_value}. Error: {str(e)}")
    else:
        date_obj = date_value
    
    # Format according to requested output format
    if output_format == "iso":
        return date_obj.isoformat()
    elif output_format == "date_only":
        return date_obj.strftime("%Y-%m-%d")
    elif output_format == "ymd":
        return date_obj.strftime("%Y%m%d")
    else:
        # Assume output_format is a strftime format string
        return date_obj.strftime(output_format)


def safe_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely get a value from a nested dictionary using dot notation.
    
    Args:
        data: Dictionary to extract value from.
        path: Path to value using dot notation (e.g., "meta.found").
        default: Default value if path not found.
        
    Returns:
        Value at path or default if not found.
    """
    parts = path.split(".")
    result = data
    
    for part in parts:
        if isinstance(result, dict) and part in result:
            result = result[part]
        else:
            return default
    
    return result


def validate_query_params(
    params: Dict[str, Any], 
    required: Optional[list] = None, 
    allowed: Optional[list] = None
) -> Dict[str, str]:
    """
    Validate query parameters against required and allowed parameters.
    
    Args:
        params: Dictionary of parameters to validate.
        required: List of required parameter names.
        allowed: List of allowed parameter names.
        
    Returns:
        Dictionary of validation errors, empty if no errors.
    """
    errors = {}
    
    # Check required parameters
    if required:
        for param in required:
            if param not in params or params[param] is None:
                errors[param] = f"Parameter '{param}' is required."
    
    # Check if parameters are in allowed list
    if allowed:
        for param in params:
            if param not in allowed:
                errors[param] = f"Parameter '{param}' is not allowed."
    
    return errors
