"""
Financial News RAG Orchestrator

This module provides a high-level orchestrator class that integrates low-level components
of the financial-news-rag system to provide a user-friendly interface for fetching, processing,
storing, embedding, and searching financial news articles.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from dotenv import load_dotenv

from financial_news_rag.eodhd import EODHDClient
from financial_news_rag.article_manager import ArticleManager
from financial_news_rag.text_processor import TextProcessor
from financial_news_rag.embeddings import EmbeddingsGenerator
from financial_news_rag.chroma_manager import ChromaDBManager
from financial_news_rag.reranker import ReRanker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinancialNewsRAG:
    """
    A high-level orchestrator class for the Financial News RAG system.
    
    This class integrates all low-level components of the system:
    - EODHD API client for fetching financial news articles
    - Article Manager for storing and retrieving articles from SQLite
    - Text Processor for cleaning and chunking article content
    - Embeddings Generator for creating embeddings from text chunks
    - ChromaDB Manager for storing and querying vector embeddings
    - ReRanker for improving search results using Gemini LLM
    
    It provides a user-friendly interface for:
    - Fetching and storing articles
    - Processing article text
    - Generating and storing embeddings
    - Searching for relevant articles
    - Managing the article and vector databases
    """
    
    def __init__(
        self,
        eodhd_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        db_path: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
        chroma_collection_name: str = "financial_news_embeddings",
        max_tokens_per_chunk: int = 2048,
    ):
        """
        Initialize the Financial News RAG orchestrator.
        
        Args:
            eodhd_api_key: EODHD API key. If None, will be loaded from environment variable.
            gemini_api_key: Gemini API key. If None, will be loaded from environment variable.
            db_path: Path to SQLite database file. If None, uses the default path.
            chroma_persist_dir: Directory to persist ChromaDB. If None, uses current working directory.
            chroma_collection_name: Name of the ChromaDB collection.
            max_tokens_per_chunk: Maximum number of tokens per chunk for text processing.
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize API keys from environment if not provided
        self.eodhd_api_key = eodhd_api_key or os.getenv("EODHD_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        # Check for required API keys
        if not self.eodhd_api_key:
            raise ValueError("EODHD API key not provided and not found in environment variables")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not provided and not found in environment variables")
        
        # Initialize default ChromaDB persist directory if not provided
        if chroma_persist_dir is None:
            chroma_persist_dir = os.path.join(os.getcwd(), 'chroma_db')
        
        # Initialize components
        self.eodhd_client = EODHDClient(api_key=self.eodhd_api_key)
        self.article_manager = ArticleManager(db_path=db_path)
        self.text_processor = TextProcessor(max_tokens_per_chunk=max_tokens_per_chunk)
        self.embeddings_generator = EmbeddingsGenerator(api_key=self.gemini_api_key)
        self.chroma_manager = ChromaDBManager(
            persist_directory=chroma_persist_dir,
            collection_name=chroma_collection_name,
            embedding_dimension=self.embeddings_generator.embedding_dim,
        )
        self.reranker = ReRanker(api_key=self.gemini_api_key)
        
        logger.info("FinancialNewsRAG orchestrator initialized successfully")
    
    def fetch_and_store_articles(
        self,
        tag: Optional[str] = None,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Fetch articles from EODHD API and store them in the database.
        
        Args:
            tag: News tag to filter articles (e.g., "TECHNOLOGY")
            symbol: Stock symbol to fetch news for (e.g., "AAPL")
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            limit: Maximum number of articles to fetch
        
        Returns:
            Dict containing operation summary with counts and status
        
        Raises:
            ValueError: If neither tag nor symbol is provided
        """
        if not tag and not symbol:
            raise ValueError("Either tag or symbol must be provided")
        
        # Initialize results dictionary
        results = {
            "articles_fetched": 0,
            "articles_stored": 0,
            "status": "SUCCESS",
            "errors": [],
        }
        
        try:
            # Handle symbol(s) as a list or single string
            symbols = []
            if symbol:
                if isinstance(symbol, str):
                    if ',' in symbol:
                        symbols = [s.strip() for s in symbol.split(',')]
                    else:
                        symbols = [symbol]
                elif isinstance(symbol, list):
                    symbols = symbol
            
            # Fetch by tag
            if tag:
                logger.info(f"Fetching articles with tag: {tag}")
                fetched_articles = self.eodhd_client.fetch_news(
                    tag=tag,
                    from_date=from_date,
                    to_date=to_date,
                    limit=limit
                )
                
                # Add source query tag to each article
                for article in fetched_articles:
                    # Create raw_content from content
                    if 'content' in article:
                        article['raw_content'] = article.pop('content', '')
                    else:
                        article['raw_content'] = ''
                    article['source_query_tag'] = tag
                
                # Log the API call
                if fetched_articles:
                    oldest_date = min([a.get('published_at', '') for a in fetched_articles]) if fetched_articles else None
                    newest_date = max([a.get('published_at', '') for a in fetched_articles]) if fetched_articles else None
                    
                    self.article_manager.log_api_call(
                        query_type='tag',
                        query_value=tag,
                        from_date=from_date,
                        to_date=to_date,
                        limit=limit,
                        offset=0,
                        articles_retrieved_count=len(fetched_articles),
                        oldest_article_date=oldest_date,
                        newest_article_date=newest_date,
                        api_call_successful=True,
                        http_status_code=200
                    )
                
                # Store articles
                stored_count = self.article_manager.store_articles(fetched_articles)
                
                # Update results
                results["articles_fetched"] += len(fetched_articles)
                results["articles_stored"] += stored_count
            
            # Fetch by symbol(s)
            if symbols:
                for sym in symbols:
                    logger.info(f"Fetching articles for symbol: {sym}")
                    fetched_articles = self.eodhd_client.fetch_news(
                        symbol=sym,
                        from_date=from_date,
                        to_date=to_date,
                        limit=limit
                    )
                    
                    # Add source query symbol to each article
                    for article in fetched_articles:
                        # Create raw_content from content
                        if 'content' in article:
                            article['raw_content'] = article.pop('content', '')
                        else:
                            article['raw_content'] = ''
                        article['source_query_symbol'] = sym
                    
                    # Log the API call
                    if fetched_articles:
                        oldest_date = min([a.get('published_at', '') for a in fetched_articles]) if fetched_articles else None
                        newest_date = max([a.get('published_at', '') for a in fetched_articles]) if fetched_articles else None
                        
                        self.article_manager.log_api_call(
                            query_type='symbol',
                            query_value=sym,
                            from_date=from_date,
                            to_date=to_date,
                            limit=limit,
                            offset=0,
                            articles_retrieved_count=len(fetched_articles),
                            oldest_article_date=oldest_date,
                            newest_article_date=newest_date,
                            api_call_successful=True,
                            http_status_code=200
                        )
                    
                    # Store articles
                    stored_count = self.article_manager.store_articles(fetched_articles)
                    
                    # Update results
                    results["articles_fetched"] += len(fetched_articles)
                    results["articles_stored"] += stored_count
            
            logger.info(f"Fetched {results['articles_fetched']} articles, stored {results['articles_stored']}")
            
        except Exception as e:
            logger.error(f"Error fetching and storing articles: {str(e)}")
            results["status"] = "FAILED"
            results["errors"].append(str(e))
        
        return results
    
    def process_pending_articles(self, limit: int = 100) -> Dict[str, Any]:
        """
        Process pending articles by cleaning their raw content.
        
        Args:
            limit: Maximum number of pending articles to process
        
        Returns:
            Dict containing operation summary with counts and status
        """
        # Initialize results dictionary
        results = {
            "articles_processed": 0,
            "articles_failed": 0,
            "status": "SUCCESS",
            "errors": [],
        }
        
        try:
            # Get pending articles
            pending_articles = self.article_manager.get_pending_articles(limit=limit)
            logger.info(f"Found {len(pending_articles)} pending articles for processing")
            
            if not pending_articles:
                logger.info("No pending articles found for processing")
                return results
            
            # Process each article
            for article in pending_articles:
                url_hash = article['url_hash']
                try:
                    if not article.get('raw_content'):
                        logger.warning(f"Empty raw content for article {url_hash}")
                        self.article_manager.update_article_processing_status(
                            url_hash, 
                            status='FAILED', 
                            error_message='Empty raw content'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    # Process the raw content with TextProcessor
                    processed_content = self.text_processor.clean_article_text(article['raw_content'])
                    
                    if not processed_content:
                        logger.warning(f"No content after cleaning for article {url_hash}")
                        self.article_manager.update_article_processing_status(
                            url_hash, 
                            status='FAILED', 
                            error_message='No content after cleaning'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    # Update article with processed content
                    self.article_manager.update_article_processing_status(
                        url_hash,
                        processed_content=processed_content,
                        status='SUCCESS'
                    )
                    results["articles_processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing article {url_hash}: {str(e)}")
                    self.article_manager.update_article_processing_status(
                        url_hash,
                        status='FAILED',
                        error_message=str(e)
                    )
                    results["articles_failed"] += 1
                    results["errors"].append(f"Error processing article {url_hash}: {str(e)}")
            
            logger.info(f"Processed {results['articles_processed']} articles, {results['articles_failed']} failed")
            
        except Exception as e:
            logger.error(f"Error in process_pending_articles: {str(e)}")
            results["status"] = "FAILED"
            results["errors"].append(str(e))
        
        return results
    
    def get_failed_text_processing_articles(self, limit: int = 100) -> List[Dict]:
        """
        Get articles with failed text processing.
        
        Args:
            limit: Maximum number of failed articles to retrieve
        
        Returns:
            List of article dictionaries with failed text processing
        """
        try:
            cursor = self.article_manager.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM articles
                WHERE status_text_processing = 'FAILED'
                LIMIT ?
                """,
                (limit,)
            )
            columns = [col[0] for col in cursor.description]
            failed_articles = [dict(zip(columns, row)) for row in cursor.fetchall()]
            logger.info(f"Found {len(failed_articles)} articles with failed text processing")
            return failed_articles
        except Exception as e:
            logger.error(f"Error getting failed text processing articles: {str(e)}")
            return []
    
    def reprocess_failed_articles(self, limit: int = 100) -> Dict[str, Any]:
        """
        Reprocess articles with failed text processing.
        
        Args:
            limit: Maximum number of failed articles to reprocess
        
        Returns:
            Dict containing operation summary with counts and status
        """
        # Initialize results dictionary
        results = {
            "articles_reprocessed": 0,
            "articles_failed": 0,
            "status": "SUCCESS",
            "errors": [],
        }
        
        try:
            # Get failed articles
            failed_articles = self.get_failed_text_processing_articles(limit=limit)
            logger.info(f"Found {len(failed_articles)} articles with failed text processing")
            
            if not failed_articles:
                logger.info("No failed articles found for reprocessing")
                return results
            
            # Reprocess each article
            for article in failed_articles:
                url_hash = article['url_hash']
                try:
                    if not article.get('raw_content'):
                        logger.warning(f"Empty raw content for article {url_hash}")
                        self.article_manager.update_article_processing_status(
                            url_hash, 
                            status='FAILED', 
                            error_message='Empty raw content'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    # Process the raw content with TextProcessor
                    processed_content = self.text_processor.clean_article_text(article['raw_content'])
                    
                    if not processed_content:
                        logger.warning(f"No content after cleaning for article {url_hash}")
                        self.article_manager.update_article_processing_status(
                            url_hash, 
                            status='FAILED', 
                            error_message='No content after cleaning'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    # Update article with processed content
                    self.article_manager.update_article_processing_status(
                        url_hash,
                        processed_content=processed_content,
                        status='SUCCESS'
                    )
                    results["articles_reprocessed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error reprocessing article {url_hash}: {str(e)}")
                    self.article_manager.update_article_processing_status(
                        url_hash,
                        status='FAILED',
                        error_message=str(e)
                    )
                    results["articles_failed"] += 1
                    results["errors"].append(f"Error reprocessing article {url_hash}: {str(e)}")
            
            logger.info(f"Reprocessed {results['articles_reprocessed']} articles, {results['articles_failed']} failed")
            
        except Exception as e:
            logger.error(f"Error in reprocess_failed_articles: {str(e)}")
            results["status"] = "FAILED"
            results["errors"].append(str(e))
        
        return results
    
    def embed_processed_articles(self, limit: int = 100) -> Dict[str, Any]:
        """
        Generate embeddings for processed articles and store them in ChromaDB.
        
        Args:
            limit: Maximum number of processed articles to embed
        
        Returns:
            Dict containing operation summary with counts and status
        """
        # Initialize results dictionary
        results = {
            "articles_embedded": 0,
            "articles_failed": 0,
            "status": "SUCCESS",
            "errors": [],
        }
        
        try:
            # Get processed articles ready for embedding
            articles_for_embedding = self.article_manager.get_processed_articles_for_embedding(limit=limit)
            logger.info(f"Found {len(articles_for_embedding)} processed articles ready for embedding")
            
            if not articles_for_embedding:
                logger.info("No processed articles found for embedding")
                return results
            
            # Generate embeddings for each article
            for article in articles_for_embedding:
                url_hash = article['url_hash']
                try:
                    processed_content = article.get('processed_content')
                    
                    if not processed_content:
                        logger.warning(f"Missing processed content for article {url_hash}")
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            error_message='Missing processed content'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    # Split processed content into chunks
                    chunks = self.text_processor.split_into_chunks(processed_content)
                    
                    if not chunks:
                        logger.warning(f"No chunks generated for article {url_hash}")
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            error_message='No chunks generated'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    logger.info(f"Generated {len(chunks)} chunks for article {url_hash}")
                    
                    # Generate embeddings for these chunks
                    chunk_embeddings = self.embeddings_generator.generate_embeddings(chunks)
                    
                    # Check for zero vectors in the embeddings
                    zero_vec = [0.0] * self.embeddings_generator.embedding_dim
                    has_zero_vector = any(emb == zero_vec for emb in chunk_embeddings)
                    
                    if has_zero_vector:
                        logger.warning(f"One or more chunk embeddings failed for article {url_hash}")
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            embedding_model=self.embeddings_generator.model_name,
                            error_message='One or more chunk embeddings failed (zero vector)'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    # Prepare chunks with embeddings for ChromaDB storage
                    formatted_chunk_embeddings = []
                    
                    # Get article published_at and convert to timestamp if available
                    published_at = article.get('published_at')
                    try:
                        # Try to convert ISO format date to UNIX timestamp
                        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        published_at_timestamp = int(dt.timestamp())
                    except (ValueError, AttributeError, TypeError):
                        published_at_timestamp = None
                    
                    # Get source query tag and symbol if available
                    source_query_tag = article.get('source_query_tag')
                    source_query_symbol = article.get('source_query_symbol')
                    
                    # Format each chunk with its embedding for ChromaDB
                    for i, (chunk_text, embedding_vector) in enumerate(zip(chunks, chunk_embeddings)):
                        chunk_id = f"{url_hash}_{i}"
                        
                        # Prepare metadata
                        metadata = {
                            "article_url_hash": url_hash,
                            "chunk_index": i
                        }
                        
                        # Add optional metadata if available
                        if published_at_timestamp:
                            metadata["published_at_timestamp"] = published_at_timestamp
                        if source_query_tag:
                            metadata["source_query_tag"] = source_query_tag
                        if source_query_symbol:
                            metadata["source_query_symbol"] = source_query_symbol
                        
                        formatted_chunk_embeddings.append({
                            "chunk_id": chunk_id,
                            "embedding": embedding_vector,
                            "text": chunk_text,
                            "metadata": metadata
                        })
                    
                    # Store embeddings in ChromaDB
                    storage_success = self.chroma_manager.add_embeddings(
                        url_hash,
                        formatted_chunk_embeddings
                    )
                    
                    if storage_success:
                        # Update article status in SQLite
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='SUCCESS',
                            embedding_model=self.embeddings_generator.model_name,
                            vector_db_id=url_hash  # Using url_hash as the vector_db_id
                        )
                        logger.info(f"Successfully stored embeddings for article {url_hash} in ChromaDB")
                        results["articles_embedded"] += 1
                    else:
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            embedding_model=self.embeddings_generator.model_name,
                            error_message='Failed to store embeddings in ChromaDB'
                        )
                        logger.error(f"Failed to store embeddings for article {url_hash} in ChromaDB")
                        results["articles_failed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error embedding article {url_hash}: {str(e)}")
                    self.article_manager.update_article_embedding_status(
                        url_hash=url_hash,
                        status='FAILED',
                        error_message=str(e)
                    )
                    results["articles_failed"] += 1
                    results["errors"].append(f"Error embedding article {url_hash}: {str(e)}")
            
            logger.info(f"Embedded {results['articles_embedded']} articles, {results['articles_failed']} failed")
            
        except Exception as e:
            logger.error(f"Error in embed_processed_articles: {str(e)}")
            results["status"] = "FAILED"
            results["errors"].append(str(e))
        
        return results
    
    def get_failed_embedding_articles(self, limit: int = 100) -> List[Dict]:
        """
        Get articles with failed embedding generation.
        
        Args:
            limit: Maximum number of failed articles to retrieve
        
        Returns:
            List of article dictionaries with failed embedding generation
        """
        try:
            cursor = self.article_manager.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM articles
                WHERE status_embedding = 'FAILED'
                LIMIT ?
                """,
                (limit,)
            )
            columns = [col[0] for col in cursor.description]
            failed_articles = [dict(zip(columns, row)) for row in cursor.fetchall()]
            logger.info(f"Found {len(failed_articles)} articles with failed embedding generation")
            return failed_articles
        except Exception as e:
            logger.error(f"Error getting failed embedding articles: {str(e)}")
            return []
    
    def re_embed_failed_articles(self, limit: int = 100) -> Dict[str, Any]:
        """
        Re-generate embeddings for articles with failed embedding status.
        
        Args:
            limit: Maximum number of failed articles to re-embed
        
        Returns:
            Dict containing operation summary with counts and status
        """
        # Initialize results dictionary
        results = {
            "articles_reembedded": 0,
            "articles_failed": 0,
            "status": "SUCCESS",
            "errors": [],
        }
        
        try:
            # Get failed embedding articles
            failed_articles = self.get_failed_embedding_articles(limit=limit)
            logger.info(f"Found {len(failed_articles)} articles with failed embedding generation")
            
            if not failed_articles:
                logger.info("No failed embedding articles found for re-embedding")
                return results
            
            # Re-embed each article
            for article in failed_articles:
                url_hash = article['url_hash']
                try:
                    processed_content = article.get('processed_content')
                    
                    if not processed_content:
                        logger.warning(f"Missing processed content for article {url_hash}")
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            error_message='Missing processed content'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    # Split processed content into chunks
                    chunks = self.text_processor.split_into_chunks(processed_content)
                    
                    if not chunks:
                        logger.warning(f"No chunks generated for article {url_hash}")
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            error_message='No chunks generated'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    logger.info(f"Generated {len(chunks)} chunks for article {url_hash}")
                    
                    # Generate embeddings for these chunks
                    chunk_embeddings = self.embeddings_generator.generate_embeddings(chunks)
                    
                    # Check for zero vectors in the embeddings
                    zero_vec = [0.0] * self.embeddings_generator.embedding_dim
                    has_zero_vector = any(emb == zero_vec for emb in chunk_embeddings)
                    
                    if has_zero_vector:
                        logger.warning(f"One or more chunk embeddings failed for article {url_hash}")
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            embedding_model=self.embeddings_generator.model_name,
                            error_message='One or more chunk embeddings failed (zero vector)'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    # Prepare chunks with embeddings for ChromaDB storage
                    formatted_chunk_embeddings = []
                    
                    # Get article published_at and convert to timestamp if available
                    published_at = article.get('published_at')
                    try:
                        # Try to convert ISO format date to UNIX timestamp
                        dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        published_at_timestamp = int(dt.timestamp())
                    except (ValueError, AttributeError, TypeError):
                        published_at_timestamp = None
                    
                    # Get source query tag and symbol if available
                    source_query_tag = article.get('source_query_tag')
                    source_query_symbol = article.get('source_query_symbol')
                    
                    # Format each chunk with its embedding for ChromaDB
                    for i, (chunk_text, embedding_vector) in enumerate(zip(chunks, chunk_embeddings)):
                        chunk_id = f"{url_hash}_{i}"
                        
                        # Prepare metadata
                        metadata = {
                            "article_url_hash": url_hash,
                            "chunk_index": i
                        }
                        
                        # Add optional metadata if available
                        if published_at_timestamp:
                            metadata["published_at_timestamp"] = published_at_timestamp
                        if source_query_tag:
                            metadata["source_query_tag"] = source_query_tag
                        if source_query_symbol:
                            metadata["source_query_symbol"] = source_query_symbol
                        
                        formatted_chunk_embeddings.append({
                            "chunk_id": chunk_id,
                            "embedding": embedding_vector,
                            "text": chunk_text,
                            "metadata": metadata
                        })
                    
                    # Delete any existing embeddings for this article
                    self.chroma_manager.delete_embeddings_by_article(url_hash)
                    
                    # Store embeddings in ChromaDB
                    storage_success = self.chroma_manager.add_embeddings(
                        url_hash,
                        formatted_chunk_embeddings
                    )
                    
                    if storage_success:
                        # Update article status in SQLite
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='SUCCESS',
                            embedding_model=self.embeddings_generator.model_name,
                            vector_db_id=url_hash  # Using url_hash as the vector_db_id
                        )
                        logger.info(f"Successfully re-embedded article {url_hash} in ChromaDB")
                        results["articles_reembedded"] += 1
                    else:
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            embedding_model=self.embeddings_generator.model_name,
                            error_message='Failed to store embeddings in ChromaDB'
                        )
                        logger.error(f"Failed to re-embed article {url_hash} in ChromaDB")
                        results["articles_failed"] += 1
                    
                except Exception as e:
                    logger.error(f"Error re-embedding article {url_hash}: {str(e)}")
                    self.article_manager.update_article_embedding_status(
                        url_hash=url_hash,
                        status='FAILED',
                        error_message=str(e)
                    )
                    results["articles_failed"] += 1
                    results["errors"].append(f"Error re-embedding article {url_hash}: {str(e)}")
            
            logger.info(f"Re-embedded {results['articles_reembedded']} articles, {results['articles_failed']} failed")
            
        except Exception as e:
            logger.error(f"Error in re_embed_failed_articles: {str(e)}")
            results["status"] = "FAILED"
            results["errors"].append(str(e))
        
        return results
    
    def get_article_database_status(self) -> Dict[str, Any]:
        """
        Get statistics about the article database.
        
        Returns:
            Dict containing article database statistics
        """
        try:
            cursor = self.article_manager.conn.cursor()
            
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
            
        except Exception as e:
            logger.error(f"Error getting article database status: {str(e)}")
            return {
                "error": str(e),
                "status": "FAILED"
            }
    
    def get_vector_database_status(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dict containing vector database statistics
        """
        try:
            # Get ChromaDB collection status
            collection_status = self.chroma_manager.get_collection_status()
            logger.info(f"Vector database status: {collection_status['total_chunks']} total chunks")
            return collection_status
            
        except Exception as e:
            logger.error(f"Error getting vector database status: {str(e)}")
            return {
                "error": str(e),
                "status": "FAILED"
            }
    
    def search_articles(
        self,
        query: str,
        n_results: int = 5,
        sort_by_metadata: Optional[Dict[str, str]] = None,
        rerank: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for articles relevant to a query.
        
        Args:
            query: The search query
            n_results: Maximum number of results to return
            sort_by_metadata: Dictionary for filtering/sorting based on metadata
                e.g., {"published_at_timestamp": "desc"}
            rerank: Whether to apply re-ranking with Gemini LLM
        
        Returns:
            List of article dictionaries with relevance scores
        """
        try:
            logger.info(f"Searching for articles with query: '{query}'")
            
            # Generate embedding for the query
            query_embedding = self.embeddings_generator.generate_embeddings([query])[0]
            
            # Translate sort_by_metadata to ChromaDB filter if possible
            filter_metadata = None
            if sort_by_metadata:
                # This is a simplified implementation
                # ChromaDB may have limited filtering capabilities
                filter_metadata = {}
                for key, value in sort_by_metadata.items():
                    # For timestamp ranges, could create a range filter
                    if key == "published_at_timestamp" and value.lower() in ["desc", "asc"]:
                        # This would need to be implemented properly with actual date ranges
                        pass
            
            # Query the vector database
            chroma_results = self.chroma_manager.query_embeddings(
                query_embedding=query_embedding,
                n_results=n_results if not rerank else max(n_results * 2, 10),  # Get more results for reranking
                filter_metadata=filter_metadata
            )
            
            if not chroma_results:
                logger.warning("No relevant articles found in ChromaDB")
                return []
            
            logger.info(f"Retrieved {len(chroma_results)} relevant chunks from ChromaDB")
            
            # Extract unique article hashes from the results
            unique_article_hashes = set()
            for result in chroma_results:
                article_hash = result['metadata'].get('article_url_hash')
                if article_hash:
                    unique_article_hashes.add(article_hash)
            
            # Fetch full article content for the retrieved articles
            articles = []
            for url_hash in unique_article_hashes:
                article = self.article_manager.get_article_by_hash(url_hash)
                if article and article.get('processed_content'):
                    # Add a similarity score based on the best chunk match
                    best_score = 0
                    for result in chroma_results:
                        if result['metadata'].get('article_url_hash') == url_hash:
                            # ChromaDB may return distance, convert to similarity
                            similarity = 1.0 - (result.get('distance', 0) / 2.0)  # Normalize to 0-1
                            best_score = max(best_score, similarity)
                    
                    article['similarity_score'] = best_score
                    articles.append(article)
            
            if not articles:
                logger.warning("Could not retrieve any article content from the database")
                return []
            
            logger.info(f"Successfully fetched content for {len(articles)} articles")
            
            # Sort by similarity score (highest first)
            articles.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # Apply re-ranking if requested
            if rerank and articles:
                logger.info("Applying re-ranking with Gemini LLM")
                reranked_articles = self.reranker.rerank_articles(query, articles)
                final_articles = reranked_articles[:n_results]  # Limit to requested number
            else:
                final_articles = articles[:n_results]  # Limit to requested number
            
            logger.info(f"Returning {len(final_articles)} articles")
            return final_articles
            
        except Exception as e:
            logger.error(f"Error searching articles: {str(e)}")
            return []
    
    def delete_article_data(self, article_url_hash: str) -> Dict[str, Any]:
        """
        Delete an article and its associated embeddings.
        
        Args:
            article_url_hash: The URL hash of the article to delete
        
        Returns:
            Dict containing operation result and status
        """
        results = {
            "status": "SUCCESS",
            "message": f"Article {article_url_hash} deleted successfully",
            "article_deleted": False,
            "embeddings_deleted": False,
            "errors": []
        }
        
        try:
            # First try to delete embeddings from ChromaDB
            embeddings_deleted = self.chroma_manager.delete_embeddings_by_article(article_url_hash)
            results["embeddings_deleted"] = embeddings_deleted
            
            if not embeddings_deleted:
                logger.warning(f"No embeddings found for article {article_url_hash}")
                results["message"] = f"No embeddings found for article {article_url_hash}"
            
            # Then delete the article from SQLite
            cursor = self.article_manager.conn.cursor()
            cursor.execute("DELETE FROM articles WHERE url_hash = ?", (article_url_hash,))
            deleted_count = cursor.rowcount
            self.article_manager.conn.commit()
            
            results["article_deleted"] = deleted_count > 0
            
            if deleted_count == 0:
                logger.warning(f"No article found with URL hash {article_url_hash}")
                results["message"] = f"No article found with URL hash {article_url_hash}"
            
            # Overall status
            if not results["article_deleted"] and not results["embeddings_deleted"]:
                results["status"] = "WARNING"
                results["message"] = f"No data found for article {article_url_hash}"
            
            logger.info(f"Deleted article {article_url_hash}: {results['message']}")
            
        except Exception as e:
            logger.error(f"Error deleting article {article_url_hash}: {str(e)}")
            results["status"] = "FAILED"
            results["message"] = f"Error deleting article: {str(e)}"
            results["errors"].append(str(e))
        
        return results
    
    def close(self):
        """
        Close database connections and clean up resources.
        """
        try:
            self.article_manager.close_connection()
            logger.info("FinancialNewsRAG resources released")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
