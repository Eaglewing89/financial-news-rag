[← Back to Main Documentation Index](../index.md)  

[← API reference Index](./index.md)


# `text_processor` Module API Reference

The `text_processor` module is designed to handle the cleaning, normalization, validation, and chunking of raw text content, specifically from financial news articles. It ensures that text is in a suitable format for further processing, such as embedding generation.

## `TextProcessor` Class


This is the primary class in the module, providing all text processing functionalities.

### Initialization

```python
from financial_news_rag.text_processor import TextProcessor

# Example: Initialize with a max token count per chunk (NLTK not used)
processor = TextProcessor(max_tokens_per_chunk=512)

# Example: Initialize with NLTK sentence tokenization (requires NLTK and punkt data)
processor = TextProcessor(max_tokens_per_chunk=512, use_nltk=True, nltk_auto_download=True)
```

-   **`__init__(self, max_tokens_per_chunk: int, use_nltk: bool = False, nltk_auto_download: bool = False)`**
    -   Initializes the `TextProcessor`.
    -   **Parameters:**
        -   `max_tokens_per_chunk` (`int`): The maximum number of tokens allowed in a single text chunk. This is used by the `split_into_chunks` method to ensure that text segments are appropriately sized for embedding models.
        -   `use_nltk` (`bool`, optional): Whether to use NLTK for sentence tokenization. Defaults to `False`. If `False`, a regex-based sentence splitter is used.
        -   `nltk_auto_download` (`bool`, optional): If `True` and `use_nltk` is enabled, will attempt to auto-download the required NLTK data ("punkt") if not already present. Defaults to `False`.

### Methods

#### `process_and_validate_content(self, raw_text: Optional[str]) -> Dict[str, str]`

This method takes raw text, cleans it, and validates whether it's usable. It's a higher-level function that combines cleaning and basic validation.

-   **Parameters:**
    -   `raw_text` (`Optional[str]`): The raw article content as a string. Can be `None` or empty.
-   **Returns:** (`Dict[str, str]`)
    -   A dictionary with the following keys:
        -   `"status"` (`str`): Either `"SUCCESS"` or `"FAILED"`.
        -   `"reason"` (`str`): If status is `"FAILED"`, this provides a brief explanation (e.g., "Empty raw content", "No content after cleaning"). Empty if `"SUCCESS"`.
        -   `"content"` (`str`): The cleaned text content if successful and valid. An empty string if validation failed.

**Example:**
```python
raw_html_content = "<p>Some news content here. Read more: example.com</p>"
result = processor.process_and_validate_content(raw_html_content)

if result["status"] == "SUCCESS":
    cleaned_text = result["content"]
    print(f"Cleaned content: {cleaned_text}")
else:
    print(f"Processing failed: {result['reason']}")
```

#### `clean_article_text(self, raw_text: str) -> str`

Performs several cleaning and normalization steps on the input text:
1.  Removes HTML tags.
2.  Normalizes whitespace (multiple spaces, tabs, newlines become a single space).
3.  Removes common boilerplate phrases (e.g., "Click here to read more", "Copyright © ...").
4.  Attempts to fix common encoding issues (e.g., smart quotes).
5.  Normalizes Unicode characters to NFC (Normalization Form C).
6.  Trims leading/trailing whitespace from the final text.

-   **Parameters:**
    -   `raw_text` (`str`): The raw string content of an article.
-   **Returns:** (`str`)
    -   The cleaned and normalized text. Returns an empty string if the input `raw_text` is empty or `None`.

**Example:**
```python
noisy_text = "  Visit <b>our site</b> for more info!  \u00e2\u20ac\u2122s a quote.  Click here to read more."
clean_text = processor.clean_article_text(noisy_text)
print(f"Cleaned: '{clean_text}'") # Output might be: "Visit our site for more info! 's a quote."
```

#### `split_into_chunks(self, processed_text: str) -> List[str]`

Splits the already cleaned and processed text into smaller chunks. This is crucial for preparing text for embedding models, which often have input token limits.

The chunking strategy is as follows:
1.  The text is first tokenized into sentences using `nltk.tokenize.sent_tokenize`. If NLTK's `punkt` tokenizer is unavailable (e.g., not downloaded), it falls back to a simpler regex-based sentence splitting.
2.  Sentences are then grouped into chunks, ensuring that the estimated token count of each chunk does not exceed `self.max_tokens_per_chunk`.
3.  Token count is estimated by assuming 1 token ≈ 4 characters (a common heuristic for models like Gemini).
4.  If a single sentence itself exceeds `self.max_tokens_per_chunk`, that sentence is further broken down by words to fit within the limit.

-   **Parameters:**
    -   `processed_text` (`str`): The cleaned and normalized article text.
-   **Returns:** (`List[str]`)
    -   A list of text chunks. Each chunk is a string.
    -   Returns an empty list if `processed_text` is empty or `None`.

**Example:**
```python
long_article_text = "This is the first sentence. This is the second sentence which is a bit longer. And here comes a third one, making the text quite substantial overall."
# Assuming max_tokens_per_chunk is set to a value that causes splitting
chunks = processor.split_into_chunks(long_article_text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```


## NLTK Data Dependency and Tokenization Behavior

The `TextProcessor` can use either NLTK's `punkt` tokenizer or a built-in regex-based sentence splitter for chunking text:

- By default (`use_nltk=False`), a regex-based sentence splitter is used, which does not require any external dependencies.
- If `use_nltk=True`, the class will attempt to use NLTK's `sent_tokenize` for more accurate sentence splitting. This requires the `nltk` package and the `punkt` data resource to be installed.
    - If `nltk_auto_download=True` is also set, the class will attempt to download the required NLTK data automatically if it is missing.
    - If `nltk_auto_download=False` and the data is missing, an error will be raised instructing the user to manually install the required data.

Logs are generated to inform about the status of NLTK data and the tokenizer being used.
