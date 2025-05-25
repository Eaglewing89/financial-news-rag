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
from tests.fixtures.sample_data import ArticleFactory


@pytest.fixture
def sample_urls():
    """Fixture providing a list of sample URLs for testing."""
    return [
        "https://example.com/article/finance-news-2023",
        "https://finance.example.org/market-update?id=12345",
        "https://news.example.net/technology/ai-developments",
        ""  # Empty URL for edge case testing
    ]


@pytest.fixture
def sample_iso_dates():
    """Fixture providing various ISO date string formats for testing."""
    return {
        "with_z": "2023-10-26T10:00:00Z",
        "with_offset": "2023-10-26T12:00:00+02:00",
        "naive": "2023-10-26T10:00:00",
        "invalid": "invalid-date-string",
        "empty": ""
    }


class TestGenerateUrlHash:
    """Tests for generate_url_hash function."""

    def test_generate_url_hash_valid_url(self, sample_urls):
        """Test generate_url_hash with a valid URL."""
        url = sample_urls[0]  # Use the first sample URL
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
        
    def test_hash_consistency_across_articles(self):
        """Test that URL hashing is consistent for the same URL in different articles."""
        # Generate two test articles with identical URLs
        article1 = ArticleFactory.create_article(url="https://example.com/special-test-article")
        article2 = ArticleFactory.create_article(url="https://example.com/special-test-article")
        
        # Generate hashes directly
        hash1 = generate_url_hash(article1["url"])
        hash2 = generate_url_hash(article2["url"])
        
        # Verify hashes are identical
        assert hash1 == hash2
        assert len(hash1) == 64


class TestParseIsoDateString:
    """Tests for parse_iso_date_string function."""

    def test_parse_iso_date_string_valid_with_z(self, sample_iso_dates):
        """Test parse_iso_date_string with a valid ISO string ending in Z."""
        date_str = sample_iso_dates["with_z"]
        expected_dt = datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc)
        assert parse_iso_date_string(date_str) == expected_dt

    def test_parse_iso_date_string_valid_with_offset(self, sample_iso_dates):
        """Test parse_iso_date_string with a valid ISO string including a timezone offset."""
        date_str = sample_iso_dates["with_offset"]
        expected_dt = datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc)
        assert parse_iso_date_string(date_str) == expected_dt

    def test_parse_iso_date_string_valid_naive_assumes_utc(self, sample_iso_dates):
        """Test parse_iso_date_string with a naive ISO string, expecting UTC assumption."""
        # This case depends on the implementation detail: "If datetime object is naive, assume UTC"
        # For fromisoformat, it's better to ensure the string implies timezone or the function handles it.
        # The current utils.py function's fromisoformat will parse it as naive, then .replace(tzinfo=timezone.utc)
        date_str = sample_iso_dates["naive"]
        expected_dt = datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc)
        assert parse_iso_date_string(date_str) == expected_dt

    def test_parse_iso_date_string_invalid_format(self, sample_iso_dates):
        """Test parse_iso_date_string with an invalid date format."""
        assert parse_iso_date_string(sample_iso_dates["invalid"]) is None

    def test_parse_iso_date_string_empty(self, sample_iso_dates):
        """Test parse_iso_date_string with an empty string."""
        assert parse_iso_date_string(sample_iso_dates["empty"]) is None

    def test_parse_iso_date_string_none(self):
        """Test parse_iso_date_string with None input."""
        assert parse_iso_date_string(None) is None
        
    @pytest.mark.parametrize("date_str,expected_year,expected_month,expected_day", [
        ("2022-01-15T08:30:00Z", 2022, 1, 15),
        ("2021-12-31T23:59:59Z", 2021, 12, 31),
        ("2023-02-28T12:00:00Z", 2023, 2, 28),
    ])
    def test_parse_iso_date_string_various_dates(self, date_str, expected_year, expected_month, expected_day):
        """Test parse_iso_date_string with various valid dates using parametrization."""
        result = parse_iso_date_string(date_str)
        assert result is not None
        assert result.year == expected_year
        assert result.month == expected_month
        assert result.day == expected_day
        assert result.tzinfo == timezone.utc


class TestConvertIsoToTimestamp:
    """Tests for convert_iso_to_timestamp function."""

    def test_convert_iso_to_timestamp_valid(self, sample_iso_dates):
        """Test convert_iso_to_timestamp with a valid ISO string."""
        date_str = sample_iso_dates["with_z"]
        expected_timestamp = int(datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc).timestamp())
        assert convert_iso_to_timestamp(date_str) == expected_timestamp

    def test_convert_iso_to_timestamp_invalid(self, sample_iso_dates):
        """Test convert_iso_to_timestamp with an invalid ISO string."""
        assert convert_iso_to_timestamp(sample_iso_dates["invalid"]) is None

    def test_convert_iso_to_timestamp_none(self):
        """Test convert_iso_to_timestamp with None input."""
        assert convert_iso_to_timestamp(None) is None
        
    @pytest.mark.parametrize("date_str,expected_timestamp", [
        ("2020-01-01T00:00:00Z", 1577836800),  # 2020-01-01 00:00:00 UTC
        ("2020-01-01T00:00:00+00:00", 1577836800),
        ("2020-01-01T01:00:00+01:00", 1577836800),  # Same UTC time, different representation
    ])
    def test_convert_iso_to_timestamp_equivalence(self, date_str, expected_timestamp):
        """Test that different timezone representations of the same time produce the same timestamp."""
        assert convert_iso_to_timestamp(date_str) == expected_timestamp


class TestGetUtcNow:
    """Tests for get_utc_now function."""

    def test_get_utc_now(self):
        """Test get_utc_now returns a UTC-aware datetime close to the current time."""
        dt_now = get_utc_now()
        assert isinstance(dt_now, datetime)
        assert dt_now.tzinfo == timezone.utc
        # Check if it's close to the actual current time
        assert abs((datetime.now(timezone.utc) - dt_now).total_seconds()) < 1
        
    def test_get_utc_now_timezone_consistency(self):
        """Test that get_utc_now consistently returns UTC timezone."""
        # Call function multiple times to verify consistency
        results = [get_utc_now() for _ in range(5)]
        
        # Verify all results have UTC timezone
        for result in results:
            assert result.tzinfo == timezone.utc
            assert result.tzinfo.utcoffset(result) == timedelta(0)


class TestGetCutoffDatetime:
    """Tests for get_cutoff_datetime function."""

    @pytest.mark.parametrize("days", [0, 1, 7, 30, 90, 365])
    def test_get_cutoff_datetime_various_days(self, days):
        """Test get_cutoff_datetime with various day values."""
        cutoff_dt = get_cutoff_datetime(days)
        expected_dt = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Validate the result
        assert isinstance(cutoff_dt, datetime)
        assert cutoff_dt.tzinfo == timezone.utc
        
        # Allow for a small difference due to execution time
        # The difference should be less than 1 second
        time_difference = abs((expected_dt - cutoff_dt).total_seconds())
        assert time_difference < 1, f"Time difference too large: {time_difference} seconds"

    def test_get_cutoff_datetime_zero_days(self):
        """Test get_cutoff_datetime with zero days."""
        days = 0
        cutoff_dt = get_cutoff_datetime(days)
        expected_dt = datetime.now(timezone.utc)
        assert isinstance(cutoff_dt, datetime)
        assert cutoff_dt.tzinfo == timezone.utc
        assert abs((expected_dt - cutoff_dt).total_seconds()) < 1

    def test_get_cutoff_datetime_negative_days(self):
        """Test get_cutoff_datetime with a negative number of days (future date)."""
        days = -5
        cutoff_dt = get_cutoff_datetime(days)
        expected_dt = datetime.now(timezone.utc) - timedelta(days=days)  # adding 5 days
        assert isinstance(cutoff_dt, datetime)
        assert cutoff_dt.tzinfo == timezone.utc
        assert abs((expected_dt - cutoff_dt).total_seconds()) < 1
        
    def test_get_cutoff_datetime_timezone_preservation(self):
        """Test that get_cutoff_datetime preserves the UTC timezone."""
        # Test with different day values
        for days in [10, 20, 30]:
            result = get_cutoff_datetime(days)
            # Verify timezone is UTC
            assert result.tzinfo == timezone.utc
            assert result.tzinfo.utcoffset(result) == timedelta(0)
