"""
Article Manager for Financial News RAG.

This module provides a class for managing article data in a SQLite database:
1. Database connection and initialization
2. Storing and retrieving articles
3. Updating article processing and embedding status
4. Logging API calls
"""

import json
import logging
import os
import sqlite3
from hashlib import sha256
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArticleManager:
    """
    A class for managing articles in a SQLite database.
    
    This class handles:
    - Database initialization and connection
    - Article storage and retrieval
    - Status tracking in SQLite
    - API call logging
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
    
    def __init__(self, db_path: str = None):
        """
        Initialize the article manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses DATABASE_PATH from config
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
    
    def get_article_by_hash(self, url_hash: str) -> dict:
        """
        Get complete article data by URL hash.
        
        Args:
            url_hash: SHA-256 hash of the article URL
            
        Returns:
            dict: Complete article data or None if not found
        """
        query = """
        SELECT url_hash, title, raw_content, processed_content, url, published_at, 
               symbols, tags, sentiment, status_text_processing, status_embedding
        FROM articles
        WHERE url_hash = ?
        """
        cursor = self._execute_query(query, (url_hash,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        return {
            'url_hash': row[0],
            'title': row[1],
            'raw_content': row[2],
            'processed_content': row[3],
            'url': row[4],
            'published_at': row[5],
            'symbols': json.loads(row[6]) if row[6] else [],
            'tags': json.loads(row[7]) if row[7] else [],
            'sentiment': json.loads(row[8]) if row[8] else {},
            'status_text_processing': row[9],
            'status_embedding': row[10]
        }
    
    def get_processed_articles_for_embedding(self, status: str = 'PENDING', limit: int = 100) -> List[Dict]:
        """
        Get articles that have been processed and are ready for embedding,
        or have a specific embedding status.
        
        Args:
            status: The embedding status to filter articles by (e.g., 'PENDING', 'FAILED').
            limit: Maximum number of articles to retrieve
            
        Returns:
            List of article dictionaries with processed content
        """
        query = """
        SELECT url_hash, processed_content, title, url, published_at, 
               symbols, tags, sentiment
        FROM articles
        WHERE status_text_processing = 'SUCCESS' AND status_embedding = ?
        LIMIT ?
        """
        cursor = self._execute_query(query, (status, limit))
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

    def log_api_call(
        self,
        query_type: str,
        query_value: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        articles_retrieved_count: int = 0,
        fetched_articles: Optional[List[Dict[str, Any]]] = None,
        api_call_successful: bool = True,
        http_status_code: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> int:
        """
        Log an API call to the api_call_log table.
        
        Args:
            query_type: Type of query ('tag' or 'symbol')
            query_value: The tag or symbol used in the query
            from_date: The from date parameter used in the API call
            to_date: The to date parameter used in the API call
            limit: The limit parameter used in the API call
            offset: The offset parameter used in the API call
            articles_retrieved_count: Number of articles retrieved
            fetched_articles: List of articles retrieved from the API
            api_call_successful: Whether the API call was successful
            http_status_code: HTTP status code of the response
            error_message: Error message if the API call failed
            
        Returns:
            int: ID of the logged API call (log_id)
        """
        query = """
        INSERT INTO api_call_log (
            query_type, query_value, last_fetched_timestamp,
            from_date_param, to_date_param, limit_param, offset_param,
            articles_retrieved_count, oldest_article_date_in_batch, newest_article_date_in_batch,
            api_call_successful, http_status_code, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Calculate oldest_article_date and newest_article_date from fetched_articles
        oldest_article_date_val = None
        newest_article_date_val = None
        
        if fetched_articles:
            valid_dates = [
                article['published_at']
                for article in fetched_articles
                if article.get('published_at') and isinstance(article.get('published_at'), str) and article.get('published_at').strip()
            ]
            if valid_dates:
                oldest_article_date_val = min(valid_dates)
                newest_article_date_val = max(valid_dates)
        
        current_timestamp = datetime.now(timezone.utc).isoformat()
        api_call_successful_int = 1 if api_call_successful else 0
        
        params = (
            query_type,
            query_value,
            current_timestamp,
            from_date,
            to_date,
            limit,
            offset,
            articles_retrieved_count,
            oldest_article_date_val,
            newest_article_date_val,
            api_call_successful_int,
            http_status_code,
            error_message
        )
        
        try:
            cursor = self._execute_query(query, params, commit=True)
            log_id = cursor.lastrowid
            logger.info(f"API call logged with ID {log_id}")
            return log_id
        except sqlite3.Error as e:
            logger.error(f"Error logging API call: {e}")
            return -1
    
    def get_articles_by_processing_status(self, status: str, limit: int = 100) -> List[Dict]:
        """
        Get articles by their text processing status.
        
        Args:
            status: The status to filter articles by (e.g., 'FAILED', 'SUCCESS', 'PENDING')
            limit: Maximum number of articles to retrieve
            
        Returns:
            List of article dictionaries matching the specified status
        """
        query = """
        SELECT url_hash, raw_content, title, url, published_at, 
               symbols, tags, sentiment, processed_content, status_text_processing, 
               status_embedding, embedding_model, vector_db_id
        FROM articles
        WHERE status_text_processing = ?
        LIMIT ?
        """
        cursor = self._execute_query(query, (status, limit))
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
                'sentiment': json.loads(row[7]) if row[7] else {},
                'processed_content': row[8],
                'status_text_processing': row[9],
                'status_embedding': row[10],
                'embedding_model': row[11],
                'vector_db_id': row[12]
            }
            articles.append(article)
            
        return articles
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the article database.
        
        Returns:
            Dict containing article database statistics including total counts,
            processing statuses, embedding statuses, article tags, symbols, 
            date ranges, and API call data.
        """
        try:
            cursor = self.conn.cursor()
            
            # Get total article count
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]
            
            # Get counts by text processing status
            cursor.execute(
                """
                SELECT status_text_processing, COUNT(*) 
                FROM articles 
                GROUP BY status_text_processing
                """
            )
            text_processing_status = {status: count for status, count in cursor.fetchall()}
            
            # Get counts by embedding status
            cursor.execute(
                """
                SELECT status_embedding, COUNT(*) 
                FROM articles 
                GROUP BY status_embedding
                """
            )
            embedding_status = {status: count for status, count in cursor.fetchall()}
            
            # Get counts by source query tag
            cursor.execute(
                """
                SELECT source_query_tag, COUNT(*) 
                FROM articles 
                WHERE source_query_tag IS NOT NULL
                GROUP BY source_query_tag
                """
            )
            tag_counts = {tag: count for tag, count in cursor.fetchall() if tag}
            
            # Get counts by source query symbol
            cursor.execute(
                """
                SELECT source_query_symbol, COUNT(*) 
                FROM articles 
                WHERE source_query_symbol IS NOT NULL
                GROUP BY source_query_symbol
                """
            )
            symbol_counts = {symbol: count for symbol, count in cursor.fetchall() if symbol}
            
            # Get date range of articles
            cursor.execute("SELECT MIN(published_at), MAX(published_at) FROM articles")
            oldest_date, newest_date = cursor.fetchone()
            
            # Get API call statistics
            cursor.execute("SELECT COUNT(*) FROM api_call_log")
            total_api_calls = cursor.fetchone()[0]
            
            cursor.execute(
                """
                SELECT SUM(articles_retrieved_count) 
                FROM api_call_log 
                WHERE api_call_successful = 1
                """
            )
            total_articles_retrieved = cursor.fetchone()[0] or 0
            
            status = {
                "total_articles": total_articles,
                "text_processing_status": text_processing_status,
                "embedding_status": embedding_status,
                "articles_by_tag": tag_counts,
                "articles_by_symbol": symbol_counts,
                "date_range": {
                    "oldest_article": oldest_date,
                    "newest_article": newest_date
                },
                "api_calls": {
                    "total_calls": total_api_calls,
                    "total_articles_retrieved": total_articles_retrieved
                }
            }
            
            logger.info(f"Article database status: {total_articles} total articles")
            return status
            
        except sqlite3.Error as e:
            logger.error(f"Error getting article database status: {str(e)}")
            return {
                "error": str(e),
                "status": "FAILED"
            }
