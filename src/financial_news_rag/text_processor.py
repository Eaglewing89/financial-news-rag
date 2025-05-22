"""
Text Processor for Financial News RAG.

This module provides a class for processing raw article content:
1. Cleaning and normalizing text
2. Chunking content for embedding
"""

import logging
import re
import unicodedata
from typing import List, Dict, Optional

import nltk
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK packages are downloaded
def download_nltk_data():
    """Download required NLTK data if not already available."""
    try:
        nltk.data.find('tokenizers/punkt')
        logger.debug("NLTK punkt tokenizer already downloaded")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        try:
            nltk.download('punkt', quiet=True)
            logger.info("Successfully downloaded NLTK punkt tokenizer")
        except Exception as e:
            logger.warning(f"Failed to download NLTK punkt tokenizer: {e}")
            logger.warning("Fallback sentence splitting will be used")

# Try to download NLTK data during module import
download_nltk_data()


class TextProcessor:
    """
    A processor for text from financial news articles.
    
    This class handles:
    - Text cleaning and normalization
    - Content chunking for embedding
    - Content validation
    """
    
    def __init__(self, max_tokens_per_chunk: int = 2048):
        """
        Initialize the text processor.
        
        Args:
            max_tokens_per_chunk: Maximum token count per chunk for embedding
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        
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
            return {
                "status": "FAILED", 
                "reason": "Empty raw content",
                "content": ""
            }
        
        # Clean the content
        cleaned_content = self.clean_article_text(raw_text)
        
        # Check if cleaning resulted in empty content
        if not cleaned_content:
            return {
                "status": "FAILED", 
                "reason": "No content after cleaning",
                "content": ""
            }
        
        # Content passed validation
        return {
            "status": "SUCCESS",
            "reason": "",
            "content": cleaned_content
        }
        
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
        text = re.sub(r'<.*?>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common boilerplate phrases
        boilerplate_patterns = [
            r'Click here to read more\.?',
            r'Read the full article at:.*$',
            r'Read more:.*$',
            r'Source:.*$',
            r'For more information, visit.*$',
            r'This article was originally published at.*$',
            r'To continue reading, subscribe to.*$',
            r'Copyright ©.*$',
            r'All rights reserved\.?'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text)
        
        # Fix encoding issues
        text = text.replace('\u00e2\u20ac\u2122', "'")   # Smart single quote
        text = text.replace('\u00e2\u20ac\u0153', '"')   # Smart opening double quote
        text = text.replace('\u00e2\u20ac\u009d', '"')   # Smart closing double quote
        text = text.replace('\u00e2\u20ac', '"')         # Another smart quote variant
        
        # Normalize unicode to NFC form
        text = unicodedata.normalize('NFC', text)
        
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
            
        # Tokenize into sentences
        try:
            sentences = sent_tokenize(processed_text)
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            # Fallback to simple regex-based sentence splitting
            # This will work even if NLTK data is not available
            sentences = re.split(r'(?<=[.!?])\s+', processed_text)
            logger.info(f"Using fallback sentence tokenization. Split into {len(sentences)} sentences.")
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Estimate tokens (1 token ≈ 4 characters for Gemini models)
        token_estimator = lambda s: len(s) // 4
        
        for sentence in sentences:
            sentence_tokens = token_estimator(sentence)
            
            # If adding this sentence would exceed max tokens, start a new chunk
            if current_length + sentence_tokens > self.max_tokens_per_chunk and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # If a single sentence is too long, we need to split it further
            if sentence_tokens > self.max_tokens_per_chunk:
                # If we have content in the current chunk, add it
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
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
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = word_tokens
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_tokens
                
        # Add the last chunk if any sentences remain
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks


# Convenience functions
def clean_text(text: str) -> str:
    """
    Clean and normalize text using the processor's cleaning function.
    
    Args:
        text: Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    processor = TextProcessor()
    return processor.clean_article_text(text)

def split_text(text: str, max_tokens: int = 2048) -> List[str]:
    """
    Split text into chunks for embedding.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    processor = TextProcessor(max_tokens_per_chunk=max_tokens)
    return processor.split_into_chunks(text)
