# src/financial_news_rag/utils.py
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional


def generate_url_hash(url: str) -> str:
    """
    Generate a SHA-256 hash from a URL for use as a unique identifier.

    Args:
        url: Article URL

    Returns:
        str: SHA-256 hash of the URL, or an empty string if URL is empty.
    """
    if not url:
        return ""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def parse_iso_date_string(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO 8601 date string into a timezone-aware datetime object (UTC).
    Handles strings with or without 'Z'.

    Args:
        date_str: The ISO 8601 date string.

    Returns:
        A datetime object or None if parsing fails or input is None.
    """
    if not date_str:
        return None
    try:
        # Ensure 'Z' is replaced with '+00:00' for consistent parsing
        if date_str.endswith("Z"):
            date_str = date_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(date_str)
        # If datetime object is naive, assume UTC
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        # Consider logging a warning here if robust error tracking is needed
        # For now, returning None as per function signature
        return None


def convert_iso_to_timestamp(date_str: Optional[str]) -> Optional[int]:
    """
    Convert an ISO 8601 date string to a UNIX timestamp (seconds since epoch).

    Args:
        date_str: The ISO 8601 date string.

    Returns:
        An integer timestamp or None if parsing fails or input is None.
    """
    dt = parse_iso_date_string(date_str)
    if dt:
        return int(dt.timestamp())
    return None


def get_utc_now() -> datetime:
    """Returns the current time in UTC."""
    return datetime.now(timezone.utc)


def get_cutoff_datetime(days: int) -> datetime:
    """
    Calculates a cutoff datetime by subtracting a number of days from the current UTC time.

    Args:
        days: The number of days to subtract.

    Returns:
        A datetime object representing the cutoff time in UTC.
    """
    return get_utc_now() - timedelta(days=days)
