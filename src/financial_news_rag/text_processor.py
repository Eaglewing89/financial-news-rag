"""
Text Processor for Financial News RAG.

This module provides a class for processing raw article content:
1. Cleaning and normalizing text
2. Chunking content for embedding
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional

# Configure module logger
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    A processor for text from financial news articles.

    This class handles:
    - Text cleaning and normalization
    - Content chunking for embedding
    - Content validation
    """

    def __init__(
        self, 
        max_tokens_per_chunk: int, 
        use_nltk: bool = False, 
        nltk_auto_download: bool = False
    ):
        """
        Initialize the text processor.

        Args:
            max_tokens_per_chunk: Maximum token count per chunk for embedding
            use_nltk: Whether to use NLTK for sentence tokenization (default: False)
            nltk_auto_download: Whether to auto-download NLTK data if missing (default: False)
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.use_nltk = use_nltk
        self.nltk_auto_download = nltk_auto_download
        
        # Initialize sentence tokenizer based on configuration
        self._sentence_tokenizer = self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize the sentence tokenizer based on configuration."""
        if not self.use_nltk:
            logger.info("Using regex-based sentence tokenization")
            return self._regex_tokenize
        
        # User wants NLTK - try to set it up
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            
            # Check if punkt data is available and test if sent_tokenize works
            punkt_available = False
            try:
                # Test if sent_tokenize actually works (this will catch version mismatches)
                sent_tokenize("Test sentence.")
                punkt_available = True
                logger.info("Using NLTK sentence tokenization")
                return sent_tokenize
            except LookupError:
                # Punkt data missing or version mismatch
                if self.nltk_auto_download:
                    logger.info("Downloading NLTK punkt tokenizer...")
                    try:
                        # Try the newer punkt_tab first (for newer NLTK versions)
                        nltk.download("punkt_tab", quiet=True)
                        # Test if it works now
                        sent_tokenize("Test sentence.")
                        logger.info("Using NLTK sentence tokenization")
                        return sent_tokenize
                    except:
                        try:
                            # Fall back to the older punkt format
                            nltk.download("punkt", quiet=True)
                            # Test if it works now
                            sent_tokenize("Test sentence.")
                            logger.info("Using NLTK sentence tokenization")
                            return sent_tokenize
                        except:
                            raise RuntimeError(
                                "Failed to download working NLTK punkt tokenizer. "
                                "Please manually run: python -c \"import nltk; nltk.download('punkt_tab')\""
                            )
                else:
                    raise RuntimeError(
                        "NLTK requested but punkt tokenizer not available. "
                        "Run: python -c \"import nltk; nltk.download('punkt_tab')\" or "
                        "python -c \"import nltk; nltk.download('punkt')\" "
                        "or set nltk_auto_download=True"
                    )
        except ImportError:
            raise RuntimeError(
                "NLTK requested but not installed. "
                "Install with: pip install nltk"
            )
    
    def _regex_tokenize(self, text: str) -> List[str]:
        """Fallback regex-based sentence tokenization."""
        return re.split(r"(?<=[.!?])\s+", text)

    def process_and_validate_content(self, raw_text: Optional[str]) -> Dict[str, str]:
        """
        Process and validate article content.

        Args:
            raw_text: Raw article content to process and validate

        Returns:
            Dict containing:
                status: 'SUCCESS' or 'FAILED'
                reason: Reason for failure (empty string if successful)
                content: Cleaned content (empty string if validation failed)
        """
        # Check if raw_text is None or empty
        if not raw_text or raw_text.strip() == "":
            return {"status": "FAILED", "reason": "Empty raw content", "content": ""}

        # Clean the content
        cleaned_content = self.clean_article_text(raw_text)

        # Check if cleaning resulted in empty content
        if not cleaned_content:
            return {
                "status": "FAILED",
                "reason": "No content after cleaning",
                "content": "",
            }

        # Content passed validation
        return {"status": "SUCCESS", "reason": "", "content": cleaned_content}

    def clean_article_text(self, raw_text: str) -> str:
        """
        Clean and normalize article text content from EODHD API.

        Args:
            raw_text: Raw article content

        Returns:
            str: Cleaned and normalized text
        """
        if not raw_text:
            return ""

        text = raw_text

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove common boilerplate phrases
        boilerplate_patterns = [
            r"Click here to read more\.?",
            r"Read the full article at:.*$",
            r"Read more:.*$",
            r"Source:.*$",
            r"For more information, visit.*$",
            r"This article was originally published at.*$",
            r"To continue reading, subscribe to.*$",
            r"Copyright ©.*$",
            r"All rights reserved\.?",
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text)

        # Fix encoding issues
        text = text.replace("\u00e2\u20ac\u2122", "'")  # Smart single quote
        text = text.replace("\u00e2\u20ac\u0153", '"')  # Smart opening double quote
        text = text.replace("\u00e2\u20ac\u009d", '"')  # Smart closing double quote
        text = text.replace("\u00e2\u20ac", '"')  # Another smart quote variant

        # Normalize unicode to NFC form
        text = unicodedata.normalize("NFC", text)

        # Final trim
        text = text.strip()

        return text

    def split_into_chunks(self, processed_text: str) -> List[str]:
        """
        Split processed article text into chunks for embedding.

        Args:
            processed_text: Cleaned and normalized article text

        Returns:
            List[str]: List of text chunks ready for embedding
        """
        if not processed_text:
            return []

        # Tokenize into sentences using the configured tokenizer
        sentences = self._sentence_tokenizer(processed_text)

        chunks = []
        current_chunk = []
        current_length = 0

        # Estimate tokens (1 token ≈ 4 characters for Gemini models)
        token_estimator = lambda s: len(s) // 4

        for sentence in sentences:
            sentence_tokens = token_estimator(sentence)

            # If adding this sentence would exceed max tokens, start a new chunk
            if (
                current_length + sentence_tokens > self.max_tokens_per_chunk
                and current_chunk
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            # If a single sentence is too long, we need to split it further
            if sentence_tokens > self.max_tokens_per_chunk:
                # If we have content in the current chunk, add it
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split the long sentence into smaller parts
                words = sentence.split()
                temp_chunk = []
                temp_length = 0

                for word in words:
                    word_tokens = token_estimator(word + " ")
                    if temp_length + word_tokens <= self.max_tokens_per_chunk:
                        temp_chunk.append(word)
                        temp_length += word_tokens
                    else:
                        if temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = word_tokens

                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_tokens

        # Add the last chunk if any sentences remain
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
