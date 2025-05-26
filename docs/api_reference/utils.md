# `utils` Module API Reference

The `utils` module provides a collection of helper functions used throughout the Financial News RAG application. These functions handle common tasks such as data type conversion, unique ID generation, and date/time manipulations.

## Functions

Below is a detailed description of each utility function available in this module.

### `generate_url_hash(url: str) -> str`

Generates a SHA-256 hash from a given URL string. This hash can be used as a unique identifier for articles or other resources based on their URL.

-   **Parameters:**
    -   `url` (`str`): The URL string to be hashed.
-   **Returns:** (`str`)
    -   A hexadecimal string representing the SHA-256 hash of the URL.
    -   Returns an empty string if the input `url` is empty or `None`.

**Example:**
```python
from financial_news_rag.utils import generate_url_hash

article_url = "https://example.com/news/article123"
url_hash = generate_url_hash(article_url)
print(f"Hash for {article_url}: {url_hash}")
```

### `parse_iso_date_string(date_str: Optional[str]) -> Optional[datetime]`

Parses an ISO 8601 formatted date string and converts it into a timezone-aware `datetime` object, normalized to UTC. It is designed to handle common variations of the ISO 8601 format, including those ending with 'Z' (Zulu time) or with explicit timezone offsets.

-   **Parameters:**
    -   `date_str` (`Optional[str]`): The ISO 8601 date string to parse. Can be `None`.
-   **Returns:** (`Optional[datetime]`)
    -   A `datetime` object representing the parsed date and time in UTC.
    -   Returns `None` if the input `date_str` is `None` or if parsing fails due to an invalid format.

**Example:**
```python
from financial_news_rag.utils import parse_iso_date_string

date1_str = "2023-10-26T10:30:00Z"
date2_str = "2023-10-26T12:00:00+02:00"

dt1 = parse_iso_date_string(date1_str)
dt2 = parse_iso_date_string(date2_str)

if dt1:
    print(f"Parsed UTC datetime for '{date1_str}': {dt1}")
if dt2:
    print(f"Parsed UTC datetime for '{date2_str}': {dt2}")
```

### `convert_iso_to_timestamp(date_str: Optional[str]) -> Optional[int]`

Converts an ISO 8601 formatted date string into a UNIX timestamp (an integer representing the number of seconds since the epoch, January 1, 1970, 00:00:00 UTC).
This function internally uses `parse_iso_date_string` to handle the date parsing.

-   **Parameters:**
    -   `date_str` (`Optional[str]`): The ISO 8601 date string to convert. Can be `None`.
-   **Returns:** (`Optional[int]`)
    -   An integer representing the UNIX timestamp.
    -   Returns `None` if the input `date_str` is `None` or if parsing fails.

**Example:**
```python
from financial_news_rag.utils import convert_iso_to_timestamp

date_string = "2023-01-01T00:00:00Z"
timestamp = convert_iso_to_timestamp(date_string)

if timestamp:
    print(f"Timestamp for '{date_string}': {timestamp}")
```

### `get_utc_now() -> datetime`

Returns the current date and time as a timezone-aware `datetime` object, specifically in UTC.

-   **Returns:** (`datetime`)
    -   A `datetime` object representing the current moment in UTC.

**Example:**
```python
from financial_news_rag.utils import get_utc_now

current_utc_time = get_utc_now()
print(f"Current UTC time: {current_utc_time}")
```

### `get_cutoff_datetime(days: int) -> datetime`

Calculates a cutoff `datetime` by subtracting a specified number of days from the current UTC time. This is useful for filtering data based on age (e.g., fetching articles published within the last `N` days).

-   **Parameters:**
    -   `days` (`int`): The number of days to subtract from the current UTC time.
-   **Returns:** (`datetime`)
    -   A `datetime` object representing the cutoff date and time in UTC.

**Example:**
```python
from financial_news_rag.utils import get_cutoff_datetime

# Get the datetime for 7 days ago
cutoff_date = get_cutoff_datetime(7)
print(f"Cutoff datetime (7 days ago): {cutoff_date}")
```
