"""
Unit tests for utility functions.

Tests the utility functions for URL hashing, date parsing, 
timestamp conversion, and datetime operations.
"""

import pytest
from datetime import datetime, timezone, timedelta

from financial_news_rag.utils import (
    generate_url_hash,
    parse_iso_date_string,
    convert_iso_to_timestamp,
    get_utc_now,
    get_cutoff_datetime
)


class TestGenerateUrlHash:
    """Tests for generate_url_hash function."""

    def test_generate_url_hash_valid_url(self):
        """Test generate_url_hash with a valid URL."""
        url = "http://example.com/article1"
        hash_output = generate_url_hash(url)
        
        # Verify output properties
        assert isinstance(hash_output, str)
        assert len(hash_output) == 64  # SHA-256 produces a 64-character hex string
        assert generate_url_hash(url) == generate_url_hash(url)  # Deterministic

    def test_generate_url_hash_empty_url(self):
        """Test generate_url_hash with an empty URL string."""
        assert generate_url_hash("") == ""

    def test_generate_url_hash_none_url(self):
        """Test generate_url_hash with a None URL input."""
        assert generate_url_hash(None) == ""


class TestParseIsoDateString:
    """Tests for parse_iso_date_string function."""

    def test_parse_iso_date_string_valid_with_z(self):
        """Test parse_iso_date_string with a valid ISO string ending in Z."""
        date_str = "2023-10-26T10:00:00Z"
        expected_dt = datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc)
        assert parse_iso_date_string(date_str) == expected_dt

    def test_parse_iso_date_string_valid_with_offset(self):
        """Test parse_iso_date_string with a valid ISO string including a timezone offset."""
        date_str = "2023-10-26T12:00:00+02:00"
        expected_dt = datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc)
        assert parse_iso_date_string(date_str) == expected_dt

    def test_parse_iso_date_string_valid_naive_assumes_utc(self):
        """Test parse_iso_date_string with a naive ISO string, expecting UTC assumption."""
        # This case depends on the implementation detail: "If datetime object is naive, assume UTC"
        # For fromisoformat, it's better to ensure the string implies timezone or the function handles it.
        # The current utils.py function's fromisoformat will parse it as naive, then .replace(tzinfo=timezone.utc)
        date_str = "2023-10-26T10:00:00"
        expected_dt = datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc)
        assert parse_iso_date_string(date_str) == expected_dt

    def test_parse_iso_date_string_invalid_format(self):
        """Test parse_iso_date_string with an invalid date format."""
        assert parse_iso_date_string("invalid-date-string") is None

    def test_parse_iso_date_string_empty(self):
        """Test parse_iso_date_string with an empty string."""
        assert parse_iso_date_string("") is None

    def test_parse_iso_date_string_none(self):
        """Test parse_iso_date_string with None input."""
        assert parse_iso_date_string(None) is None


class TestConvertIsoToTimestamp:
    """Tests for convert_iso_to_timestamp function."""

    def test_convert_iso_to_timestamp_valid(self):
        """Test convert_iso_to_timestamp with a valid ISO string."""
        date_str = "2023-10-26T10:00:00Z"
        expected_timestamp = int(datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc).timestamp())
        assert convert_iso_to_timestamp(date_str) == expected_timestamp

    def test_convert_iso_to_timestamp_invalid(self):
        """Test convert_iso_to_timestamp with an invalid ISO string."""
        assert convert_iso_to_timestamp("invalid-date-string") is None

    def test_convert_iso_to_timestamp_none(self):
        """Test convert_iso_to_timestamp with None input."""
        assert convert_iso_to_timestamp(None) is None


class TestGetUtcNow:
    """Tests for get_utc_now function."""

    def test_get_utc_now(self):
        """Test get_utc_now returns a UTC-aware datetime close to the current time."""
        dt_now = get_utc_now()
        assert isinstance(dt_now, datetime)
        assert dt_now.tzinfo == timezone.utc
        # Check if it's close to the actual current time
        assert abs((datetime.now(timezone.utc) - dt_now).total_seconds()) < 1


class TestGetCutoffDatetime:
    """Tests for get_cutoff_datetime function."""

    def test_get_cutoff_datetime_positive_days(self):
        """Test get_cutoff_datetime with a positive number of days."""
        days = 30
        cutoff_dt = get_cutoff_datetime(days)
        expected_dt = datetime.now(timezone.utc) - timedelta(days=days)
        assert isinstance(cutoff_dt, datetime)
        assert cutoff_dt.tzinfo == timezone.utc
        # Allow for a small difference due to execution time
        assert abs((expected_dt - cutoff_dt).total_seconds()) < 1

    def test_get_cutoff_datetime_zero_days(self):
        """Test get_cutoff_datetime with zero days."""
        days = 0
        cutoff_dt = get_cutoff_datetime(days)
        expected_dt = datetime.now(timezone.utc)
        assert isinstance(cutoff_dt, datetime)
        assert cutoff_dt.tzinfo == timezone.utc
        assert abs((expected_dt - cutoff_dt).total_seconds()) < 1

    def test_get_cutoff_datetime_negative_days(self):
        """Test get_cutoff_datetime with a negative number of days (expecting a future date)."""
        # Behavior for negative days might be to go into the future
        days = -5
        cutoff_dt = get_cutoff_datetime(days)
        expected_dt = datetime.now(timezone.utc) - timedelta(days=days)  # which means adding 5 days
        assert isinstance(cutoff_dt, datetime)
        assert cutoff_dt.tzinfo == timezone.utc
        assert abs((expected_dt - cutoff_dt).total_seconds()) < 1
