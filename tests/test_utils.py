"""
Tests for the utility functions.
"""

import pytest
from datetime import datetime

from financial_news_rag.utils import format_date, safe_get, validate_query_params


class TestFormatDate:
    """Tests for the format_date function."""
    
    def test_datetime_to_iso(self):
        """Test formatting a datetime object to ISO format."""
        date = datetime(2025, 5, 15, 10, 30, 45)
        result = format_date(date, "iso")
        assert result == "2025-05-15T10:30:45"
    
    def test_datetime_to_date_only(self):
        """Test formatting a datetime object to date_only format."""
        date = datetime(2025, 5, 15, 10, 30, 45)
        result = format_date(date, "date_only")
        assert result == "2025-05-15"
    
    def test_datetime_to_ymd(self):
        """Test formatting a datetime object to ymd format."""
        date = datetime(2025, 5, 15, 10, 30, 45)
        result = format_date(date, "ymd")
        assert result == "20250515"
    
    def test_datetime_to_custom_format(self):
        """Test formatting a datetime object to a custom format."""
        date = datetime(2025, 5, 15, 10, 30, 45)
        result = format_date(date, "%Y/%m/%d %H:%M")
        assert result == "2025/05/15 10:30"
    
    def test_iso_string_to_iso(self):
        """Test formatting an ISO string to ISO format."""
        date_str = "2025-05-15T10:30:45Z"
        result = format_date(date_str, "iso")
        assert result == "2025-05-15T10:30:45+00:00"
    
    def test_date_string_to_iso(self):
        """Test formatting a date string to ISO format."""
        date_str = "2025-05-15"
        result = format_date(date_str, "iso")
        assert "2025-05-15" in result
    
    def test_alternative_date_format(self):
        """Test formatting a date string in different format."""
        date_str = "15/05/2025"  # This format is actually supported in the implementation
        result = format_date(date_str, "iso")
        assert "2025-05-15" in result
    
    def test_invalid_date_string(self):
        """Test formatting an invalid date string."""
        date_str = "not-a-date"
        with pytest.raises(ValueError):
            format_date(date_str, "iso")


class TestSafeGet:
    """Tests for the safe_get function."""
    
    def test_get_top_level_key(self):
        """Test getting a top-level key from a dictionary."""
        data = {"name": "John", "age": 30}
        result = safe_get(data, "name")
        assert result == "John"
    
    def test_get_nested_key(self):
        """Test getting a nested key from a dictionary."""
        data = {"user": {"name": "John", "age": 30}}
        result = safe_get(data, "user.name")
        assert result == "John"
    
    def test_get_missing_key(self):
        """Test getting a missing key with default value."""
        data = {"name": "John", "age": 30}
        result = safe_get(data, "email", "default@example.com")
        assert result == "default@example.com"
    
    def test_get_missing_nested_key(self):
        """Test getting a missing nested key with default value."""
        data = {"user": {"name": "John"}}
        result = safe_get(data, "user.email", "default@example.com")
        assert result == "default@example.com"
    
    def test_get_from_non_dict(self):
        """Test getting a key from a non-dictionary value."""
        data = {"users": ["John", "Jane"]}
        result = safe_get(data, "users.0", "default")
        assert result == "default"  # Should return default as users is a list, not a dict


class TestValidateQueryParams:
    """Tests for the validate_query_params function."""
    
    def test_valid_params(self):
        """Test validating parameters with all valid."""
        params = {"name": "John", "age": 30}
        required = ["name"]
        allowed = ["name", "age", "email"]
        errors = validate_query_params(params, required, allowed)
        assert errors == {}
    
    def test_missing_required_param(self):
        """Test validating parameters with a missing required parameter."""
        params = {"age": 30}
        required = ["name", "age"]
        errors = validate_query_params(params, required)
        assert "name" in errors
    
    def test_disallowed_param(self):
        """Test validating parameters with a disallowed parameter."""
        params = {"name": "John", "password": "secret"}
        allowed = ["name", "age", "email"]
        errors = validate_query_params(params, None, allowed)
        assert "password" in errors
    
    def test_none_value_required_param(self):
        """Test validating parameters with a required parameter having None value."""
        params = {"name": None, "age": 30}
        required = ["name", "age"]
        errors = validate_query_params(params, required)
        assert "name" in errors
    
    def test_no_validation_rules(self):
        """Test validating parameters with no validation rules."""
        params = {"name": "John", "age": 30}
        errors = validate_query_params(params)
        assert errors == {}
