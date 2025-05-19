"""
Text Processing Pipeline for Financial News RAG.

This module provides classes and functions for processing raw article content:
1. Cleaning and normalizing text
2. Chunking content for embedding
3. Managing article status in SQLite database
"""

import json
import logging
import os
import re
import sqlite3
import unicodedata
from hashlib import sha256
from typing import Dict, List, Optional, Tuple, Union, Any

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


class TextProcessingPipeline:
    """
    A pipeline for processing text from financial news articles.
    
    This class handles:
    - Database initialization and connection
    - Text cleaning and normalization
    - Content chunking for embedding
    - Status tracking in SQLite
    """
    
    # Database table creation SQL
    ARTICLES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS articles (
        url_hash TEXT PRIMARY KEY NOT NULL,
        title TEXT,
        raw_content TEXT NOT NULL,
        processed_content TEXT,
        url TEXT NOT NULL,
        published_at TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        source_api TEXT,
        symbols TEXT,
        tags TEXT,
        sentiment TEXT,
        source_query_tag TEXT,
        source_query_symbol TEXT,
        status_text_processing TEXT DEFAULT 'PENDING' NOT NULL,
        status_embedding TEXT DEFAULT 'PENDING' NOT NULL,
        embedding_model TEXT,
        vector_db_id TEXT
    )
    """
    
    API_CALL_LOG_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS api_call_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_type TEXT NOT NULL,
        query_value TEXT NOT NULL,
        last_fetched_timestamp TEXT NOT NULL,
        from_date_param TEXT,
        to_date_param TEXT,
        limit_param INTEGER,
        offset_param INTEGER,
        articles_retrieved_count INTEGER,
        oldest_article_date_in_batch TEXT,
        newest_article_date_in_batch TEXT,
        api_call_successful INTEGER NOT NULL,
        http_status_code INTEGER,
        error_message TEXT
    )
    """
    
    API_ERRORS_LOG_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS api_errors_log (
        error_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        request_url TEXT,
        request_params TEXT,
        status_code INTEGER,
        error_message TEXT,
        response_text TEXT,
        client_method TEXT
    )
    """
    
    def __init__(self, db_path: str = None, max_tokens_per_chunk: int = 2048):
        """
        Initialize the text processing pipeline.
        
        Args:
            db_path: Path to SQLite database file. If None, uses DATABASE_PATH from config
            max_tokens_per_chunk: Maximum token count per chunk for embedding
        """
        if db_path is None:
            # Try to import from config, fallback to default
            try:
                from .config import DATABASE_PATH
                self.db_path = DATABASE_PATH
            except (ImportError, AttributeError):
                self.db_path = os.path.join(os.getcwd(), 'financial_news.db')
        else:
            self.db_path = db_path
            
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.conn = None
        
        # Initialize the database
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute(self.ARTICLES_TABLE_SQL)
            cursor.execute(self.API_CALL_LOG_TABLE_SQL)
            cursor.execute(self.API_ERRORS_LOG_TABLE_SQL)
            
            # Create indices for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_text_processing ON articles(status_text_processing)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_embedding ON articles(status_embedding)")
            
            self.conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
            
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection, creating one if needed."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def close_connection(self) -> None:
        """Close the database connection if open."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def _execute_query(self, query: str, params: tuple = (), commit: bool = False) -> Any:
        """Execute a SQL query with parameters."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            if commit:
                conn.commit()
            return cursor
        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}")
            logger.error(f"Query: {query}, Params: {params}")
            raise
    
    def article_exists(self, url_hash: str) -> bool:
        """
        Check if an article with the given URL hash exists in the database.
        
        Args:
            url_hash: SHA-256 hash of the article URL
            
        Returns:
            bool: True if the article exists, False otherwise
        """
        query = "SELECT 1 FROM articles WHERE url_hash = ? LIMIT 1"
        cursor = self._execute_query(query, (url_hash,))
        return cursor.fetchone() is not None
    
    def get_article_status(self, url_hash: str) -> dict:
        """
        Get the processing status of an article.
        
        Args:
            url_hash: SHA-256 hash of the article URL
            
        Returns:
            dict: Article status information or None if not found
        """
        query = """
        SELECT url_hash, status_text_processing, status_embedding, 
               embedding_model, vector_db_id
        FROM articles
        WHERE url_hash = ?
        """
        cursor = self._execute_query(query, (url_hash,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        return {
            'url_hash': row[0],
            'status_text_processing': row[1],
            'status_embedding': row[2],
            'embedding_model': row[3],
            'vector_db_id': row[4]
        }
    
    def get_pending_articles(self, limit: int = 100) -> List[Dict]:
        """
        Get articles pending text processing.
        
        Args:
            limit: Maximum number of articles to retrieve
            
        Returns:
            List of article dictionaries with raw content
        """
        query = """
        SELECT url_hash, raw_content, title, url, published_at, 
               symbols, tags, sentiment
        FROM articles
        WHERE status_text_processing = 'PENDING'
        LIMIT ?
        """
        cursor = self._execute_query(query, (limit,))
        articles = []
        
        for row in cursor.fetchall():
            article = {
                'url_hash': row[0],
                'raw_content': row[1],
                'title': row[2],
                'url': row[3],
                'published_at': row[4],
                'symbols': json.loads(row[5]) if row[5] else [],
                'tags': json.loads(row[6]) if row[6] else [],
                'sentiment': json.loads(row[7]) if row[7] else {}
            }
            articles.append(article)
            
        return articles
    
    def get_processed_articles_for_embedding(self, limit: int = 100) -> List[Dict]:
        """
        Get articles that have been processed but not yet embedded.
        
        Args:
            limit: Maximum number of articles to retrieve
            
        Returns:
            List of article dictionaries with processed content
        """
        query = """
        SELECT url_hash, processed_content, title, url, published_at, 
               symbols, tags, sentiment
        FROM articles
        WHERE status_text_processing = 'SUCCESS' AND status_embedding = 'PENDING'
        LIMIT ?
        """
        cursor = self._execute_query(query, (limit,))
        articles = []
        
        for row in cursor.fetchall():
            article = {
                'url_hash': row[0],
                'processed_content': row[1],
                'title': row[2],
                'url': row[3],
                'published_at': row[4],
                'symbols': json.loads(row[5]) if row[5] else [],
                'tags': json.loads(row[6]) if row[6] else [],
                'sentiment': json.loads(row[7]) if row[7] else {}
            }
            articles.append(article)
            
        return articles
    
    def update_article_processing_status(
        self, 
        url_hash: str, 
        processed_content: Optional[str] = None, 
        status: str = 'SUCCESS',
        error_message: Optional[str] = None
    ) -> None:
        """
        Update an article's text processing status in the database.
        
        Args:
            url_hash: SHA-256 hash of the article URL
            processed_content: Cleaned and processed article content
            status: Processing status ('SUCCESS', 'FAILED', etc.)
            error_message: Error message if processing failed
        """
        if status == 'SUCCESS' and processed_content:
            query = """
            UPDATE articles
            SET processed_content = ?, status_text_processing = ?
            WHERE url_hash = ?
            """
            self._execute_query(query, (processed_content, status, url_hash), commit=True)
        else:
            query = """
            UPDATE articles
            SET status_text_processing = ?
            WHERE url_hash = ?
            """
            self._execute_query(query, (status, url_hash), commit=True)
            
        if error_message:
            # Log the error to a separate table or field if needed
            logger.error(f"Processing error for {url_hash}: {error_message}")
    
    def update_article_embedding_status(
        self, 
        url_hash: str, 
        status: str = 'SUCCESS', 
        embedding_model: Optional[str] = None,
        vector_db_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update an article's embedding status in the database.
        
        Args:
            url_hash: SHA-256 hash of the article URL
            status: Embedding status ('SUCCESS', 'FAILED', etc.)
            embedding_model: Name of the embedding model used
            vector_db_id: ID in the vector database
            error_message: Error message if embedding failed
        """
        query = """
        UPDATE articles
        SET status_embedding = ?, 
            embedding_model = COALESCE(?, embedding_model),
            vector_db_id = COALESCE(?, vector_db_id)
        WHERE url_hash = ?
        """
        self._execute_query(
            query, 
            (status, embedding_model, vector_db_id, url_hash), 
            commit=True
        )
            
        if error_message:
            # Log the error to a separate table or field if needed
            logger.error(f"Embedding error for {url_hash}: {error_message}")
    
    def store_articles(self, articles: List[Dict], replace_existing: bool = False) -> int:
        """
        Store articles in the database.
        
        Args:
            articles: List of article dictionaries from EODHDClient
            replace_existing: Whether to replace existing articles
            
        Returns:
            int: Number of articles stored
        """
        if not articles:
            return 0
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Prepare query based on replace_existing flag
        if replace_existing:
            query = """
            INSERT OR REPLACE INTO articles (
                url_hash, title, raw_content, url, published_at, fetched_at,
                source_api, symbols, tags, sentiment, 
                source_query_tag, source_query_symbol,
                status_text_processing, status_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 'PENDING')
            """
        else:
            query = """
            INSERT OR IGNORE INTO articles (
                url_hash, title, raw_content, url, published_at, fetched_at,
                source_api, symbols, tags, sentiment,
                source_query_tag, source_query_symbol,
                status_text_processing, status_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 'PENDING')
            """
        
        stored_count = 0
        
        try:
            for article in articles:
                # Check required fields
                required_fields = ['url_hash', 'url', 'published_at', 'fetched_at']
                if not all(field in article for field in required_fields):
                    logger.warning(f"Skipping article with missing required fields: {article.get('url_hash', 'unknown')}")
                    continue
                
                # Handle the case where raw_content is None
                raw_content = article.get('raw_content', '')
                if raw_content is None:
                    raw_content = ''
                
                # Convert lists and dicts to JSON strings
                symbols_json = json.dumps(article.get('symbols', []))
                tags_json = json.dumps(article.get('tags', []))
                sentiment_json = json.dumps(article.get('sentiment', {}))
                
                # Source query information
                source_query_tag = article.get('source_query_tag')
                source_query_symbol = article.get('source_query_symbol')
                
                params = (
                    article['url_hash'],
                    article.get('title', ''),
                    raw_content,
                    article['url'],
                    article['published_at'],
                    article['fetched_at'],
                    article.get('source_api', 'unknown'),
                    symbols_json,
                    tags_json,
                    sentiment_json,
                    source_query_tag,
                    source_query_symbol
                )
                
                cursor.execute(query, params)
                stored_count += 1
                
            conn.commit()
            return stored_count
            
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error storing articles: {e}")
            raise
    
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
    
    def process_articles(self, limit: int = 50) -> Tuple[int, int]:
        """
        Process pending articles in the database.
        
        Args:
            limit: Maximum number of articles to process
            
        Returns:
            Tuple[int, int]: (number of articles processed, number of failures)
        """
        articles = self.get_pending_articles(limit=limit)
        
        if not articles:
            logger.info("No pending articles to process")
            return 0, 0
            
        processed_count = 0
        failure_count = 0
        
        for article in articles:
            url_hash = article['url_hash']
            try:
                # Check if raw_content is None or empty
                if not article.get('raw_content'):
                    logger.warning(f"Empty or missing raw content for article {url_hash}")
                    self.update_article_processing_status(
                        url_hash, 
                        status='FAILED', 
                        error_message='Empty or missing raw content'
                    )
                    failure_count += 1
                    continue
                
                # Clean the raw content
                processed_content = self.clean_article_text(article['raw_content'])
                
                if not processed_content:
                    logger.warning(f"No content after cleaning for article {url_hash}")
                    self.update_article_processing_status(
                        url_hash, 
                        status='FAILED', 
                        error_message='No content after cleaning'
                    )
                    failure_count += 1
                    continue
                
                # Update the article with processed content
                self.update_article_processing_status(
                    url_hash,
                    processed_content=processed_content,
                    status='SUCCESS'
                )
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing article {url_hash}: {str(e)}")
                self.update_article_processing_status(
                    url_hash,
                    status='FAILED',
                    error_message=str(e)
                )
                failure_count += 1
                
        logger.info(f"Processed {processed_count} articles with {failure_count} failures")
        return processed_count, failure_count
    
    def get_chunks_for_article(self, url_hash: str) -> List[Dict]:
        """
        Get processed content chunks for a specific article.
        
        Args:
            url_hash: SHA-256 hash of the article URL
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        query = """
        SELECT url_hash, processed_content, title, url, published_at, 
               symbols, tags, sentiment
        FROM articles
        WHERE url_hash = ? AND status_text_processing = 'SUCCESS'
        """
        cursor = self._execute_query(query, (url_hash,))
        row = cursor.fetchone()
        
        if not row:
            return []
            
        article = {
            'url_hash': row[0],
            'processed_content': row[1],
            'title': row[2],
            'url': row[3],
            'published_at': row[4],
            'symbols': json.loads(row[5]) if row[5] else [],
            'tags': json.loads(row[6]) if row[6] else [],
            'sentiment': json.loads(row[7]) if row[7] else {}
        }
        
        # Split the content into chunks
        text_chunks = self.split_into_chunks(article['processed_content'])
        
        # Create result objects with chunk text and parent metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                'text': chunk_text,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'parent_url_hash': article['url_hash'],
                'title': article['title'],
                'url': article['url'],
                'published_at': article['published_at'],
                'symbols': article['symbols'],
                'tags': article['tags'],
                'sentiment': article['sentiment']
            }
            chunks.append(chunk)
            
        return chunks
    
    @staticmethod
    def generate_url_hash(url: str) -> str:
        """
        Generate a SHA-256 hash from a URL for use as a unique identifier.
        
        Args:
            url: Article URL
            
        Returns:
            str: SHA-256 hash of the URL
        """
        return sha256(url.encode('utf-8')).hexdigest()


# Convenience functions
def clean_text(text: str) -> str:
    """
    Clean and normalize text using the pipeline's cleaning function.
    
    Args:
        text: Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    pipeline = TextProcessingPipeline()
    return pipeline.clean_article_text(text)

def split_text(text: str, max_tokens: int = 2048) -> List[str]:
    """
    Split text into chunks for embedding.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    pipeline = TextProcessingPipeline(max_tokens_per_chunk=max_tokens)
    return pipeline.split_into_chunks(text)
