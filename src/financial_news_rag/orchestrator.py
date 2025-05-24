"""
Financial News RAG Orchestrator

This module provides a high-level orchestrator class that integrates low-level components
of the financial-news-rag system to provide a user-friendly interface for fetching, processing,
storing, embedding, and searching financial news articles.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta, timezone

from financial_news_rag.eodhd import EODHDClient
from financial_news_rag.article_manager import ArticleManager
from financial_news_rag.text_processor import TextProcessor
from financial_news_rag.embeddings import EmbeddingsGenerator
from financial_news_rag.chroma_manager import ChromaDBManager
from financial_news_rag.reranker import ReRanker
from financial_news_rag.config import Config
from financial_news_rag import utils

# Configure logging
# logging.basicConfig(
# level=logging.INFO,
# format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__)

# Configure basic logging if no handlers are configured for the root logger.
# This allows the application (e.g., the notebook) to set up its own logging
# configuration if desired, without the library overriding it or adding duplicate handlers.
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


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
        chroma_collection_name: Optional[str] = None,
        max_tokens_per_chunk: Optional[int] = None,
    ):
        """
        Initialize the Financial News RAG orchestrator.
        
        Args:
            eodhd_api_key: EODHD API key. If None, will be loaded from Config.
            gemini_api_key: Gemini API key. If None, will be loaded from Config.
            db_path: Path to SQLite database file. If None, will be loaded from Config.
            chroma_persist_dir: Directory to persist ChromaDB. If None, will be loaded from Config.
            chroma_collection_name: Name of the ChromaDB collection. If None, will be loaded from Config.
            max_tokens_per_chunk: Maximum number of tokens per chunk for text processing. If None, will be loaded from Config.
        """
        # Initialize the config
        self.config = Config()
        
        # Get API keys from parameters or config
        self.eodhd_api_key = eodhd_api_key or self.config.eodhd_api_key
        self.gemini_api_key = gemini_api_key or self.config.gemini_api_key
        
        # Check for required API keys
        if not self.eodhd_api_key:
            raise ValueError("EODHD API key not provided in parameters or config")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not provided in parameters or config")
        
        # Get configuration values from parameters or config
        db_path_to_use = db_path or self.config.database_path
        chroma_persist_dir_to_use = chroma_persist_dir or self.config.chroma_default_persist_directory
        chroma_collection_name_to_use = chroma_collection_name or self.config.chroma_default_collection_name
        max_tokens_per_chunk_to_use = max_tokens_per_chunk or self.config.textprocessor_max_tokens_per_chunk
        
        # Initialize components with configuration
        self.eodhd_client = EODHDClient(
            api_key=self.eodhd_api_key,
            api_url=self.config.eodhd_api_url,
            default_timeout=self.config.eodhd_default_timeout,
            default_max_retries=self.config.eodhd_default_max_retries,
            default_backoff_factor=self.config.eodhd_default_backoff_factor,
            default_limit=self.config.eodhd_default_limit
        )
        
        self.article_manager = ArticleManager(db_path=db_path_to_use)
        
        self.text_processor = TextProcessor(max_tokens_per_chunk=max_tokens_per_chunk_to_use)
        
        self.embeddings_generator = EmbeddingsGenerator(
            api_key=self.gemini_api_key,
            model_name=self.config.embeddings_default_model,
            model_dimensions=self.config.embeddings_model_dimensions,
            task_type=self.config.embeddings_default_task_type
        )
        
        self.chroma_manager = ChromaDBManager(
            persist_directory=chroma_persist_dir_to_use,
            collection_name=chroma_collection_name_to_use,
            embedding_dimension=self.config.chroma_default_embedding_dimension,
        )
        
        self.reranker = ReRanker(
            api_key=self.gemini_api_key,
            model_name=self.config.reranker_default_model
        )
        
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
                Note: Only a single symbol string is accepted, not a comma-separated list.
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            limit: Maximum number of articles to fetch
        
        Returns:
            Dict containing operation summary with counts and status
        
        Raises:
            ValueError: If neither tag nor symbol is provided
            ValueError: If both tag and symbol are provided (mutually exclusive)
        """
        if not tag and not symbol:
            raise ValueError("Either tag or symbol must be provided")
        
        if tag and symbol:
            raise ValueError("tag and symbol parameters are mutually exclusive - provide only one")
        
        # Initialize results dictionary
        results = {
            "articles_fetched": 0,
            "articles_stored": 0,
            "status": "SUCCESS",
            "errors": [],
        }
        
        try:
            # Fetch by tag
            if tag:
                logger.info(f"Fetching articles with tag: {tag}")
                api_result = self.eodhd_client.fetch_news(
                    tag=tag,
                    from_date=from_date,
                    to_date=to_date,
                    limit=limit
                )
                
                # Extract articles and status info from the result dictionary
                fetched_articles = api_result["articles"]
                status_code = api_result["status_code"]
                success = api_result["success"]
                error_message = api_result["error_message"]
                
                # Log the API call with accurate status information
                self.article_manager.log_api_call(
                    query_type='tag',
                    query_value=tag,
                    from_date=from_date,
                    to_date=to_date,
                    limit=limit,
                    offset=0,
                    articles_retrieved_count=len(fetched_articles),
                    fetched_articles=fetched_articles,
                    api_call_successful=success,
                    http_status_code=status_code,
                    error_message=error_message
                )
                
                # Only store articles if the API call was successful
                stored_count = 0
                if success and fetched_articles:
                    stored_count = self.article_manager.store_articles(fetched_articles)
                
                # Update results
                results["articles_fetched"] += len(fetched_articles)
                results["articles_stored"] += stored_count
            
            # Fetch by symbol
            elif symbol:
                logger.info(f"Fetching articles for symbol: {symbol}")
                try:
                    api_result = self.eodhd_client.fetch_news(
                        symbol=symbol,
                        from_date=from_date,
                        to_date=to_date,
                        limit=limit
                    )
                    
                    # Extract articles and status info from the result dictionary
                    fetched_articles = api_result["articles"]
                    status_code = api_result["status_code"]
                    success = api_result["success"]
                    error_message = api_result["error_message"]
                    
                    # Log the API call with accurate status information
                    self.article_manager.log_api_call(
                        query_type='symbol',
                        query_value=symbol,
                        from_date=from_date,
                        to_date=to_date,
                        limit=limit,
                        offset=0,
                        articles_retrieved_count=len(fetched_articles),
                        fetched_articles=fetched_articles,
                        api_call_successful=success,
                        http_status_code=status_code,
                        error_message=error_message
                    )
                    
                    # Only store articles if the API call was successful
                    stored_count = 0
                    if success and fetched_articles:
                        stored_count = self.article_manager.store_articles(fetched_articles)
                    
                    # Update results
                    results["articles_fetched"] += len(fetched_articles)
                    results["articles_stored"] += stored_count
                except Exception as e:
                    logger.error(f"Error fetching news for symbol {symbol}: {str(e)}")
                    results["status"] = "FAILED"
                    results["errors"].append(f"Error fetching news for symbol {symbol}: {str(e)}")
            
            logger.info(f"Fetched {results['articles_fetched']} articles, stored {results['articles_stored']}")
            
        except Exception as e:
            logger.error(f"Error fetching and storing articles: {str(e)}")
            results["status"] = "FAILED"
            results["errors"].append(str(e))
        
        return results
    
    def process_articles_by_status(self, status: str = 'PENDING', limit: int = 100) -> Dict[str, Any]:
        """
        Process articles filtered by their processing status.
        
        Args:
            status: The status to filter articles by ('PENDING' or 'FAILED')
            limit: Maximum number of articles to process
        
        Returns:
            Dict containing operation summary with counts and status:
            - articles_processed: Number of articles processed successfully
            - articles_failed: Number of articles that failed processing
            - status: Overall operation status ('SUCCESS' or 'FAILED')
            - errors: List of error messages
        """
        # Initialize results dictionary
        results = {
            "articles_processed": 0,
            "articles_failed": 0,
            "status": "SUCCESS",
            "errors": [],
        }
        
        try:
            # Get articles by status
            articles = self.article_manager.get_articles_by_processing_status(status=status, limit=limit)
            logger.info(f"Found {len(articles)} articles with {status} status for processing")
            
            if not articles:
                logger.info(f"No articles with {status} status found for processing")
                return results
            
            # Process each article
            for article in articles:
                url_hash = article['url_hash']
                try:
                    # Process and validate the content using TextProcessor
                    validation_result = self.text_processor.process_and_validate_content(article.get('raw_content'))
                    
                    if validation_result["status"] == "SUCCESS":
                        # Update article with processed content
                        self.article_manager.update_article_processing_status(
                            url_hash,
                            processed_content=validation_result["content"],
                            status='SUCCESS'
                        )
                        results["articles_processed"] += 1
                    else:
                        # Content validation failed
                        self.article_manager.update_article_processing_status(
                            url_hash, 
                            status='FAILED', 
                            error_message=validation_result["reason"]
                        )
                        results["articles_failed"] += 1
                    
                except Exception as e_article:
                    logger.error(f"Error processing article {url_hash} for status {status}: {str(e_article)}")
                    self.article_manager.update_article_processing_status(
                        url_hash,
                        status='FAILED',
                        error_message=str(e_article)
                    )
                    results["articles_failed"] += 1
                    results["errors"].append(f"Error processing article {url_hash}: {str(e_article)}")
            
            logger.info(f"For status '{status}': Processed {results['articles_processed']} articles, {results['articles_failed']} failed")
            
        except Exception as e_main:
            logger.error(f"Error in process_articles_by_status for {status}: {str(e_main)}")
            results["status"] = "FAILED"
            results["errors"].append(str(e_main))
        
        return results
    
    def embed_processed_articles(self, status: str = 'PENDING', limit: int = 100) -> Dict[str, Any]:
        """
        Generate embeddings for processed articles and store them in ChromaDB.
        Can process articles with a 'PENDING' embedding status (initial attempt)
        or 'FAILED' embedding status (re-attempt).
        
        Args:
            status: The embedding status of articles to process ('PENDING' or 'FAILED').
                    Defaults to 'PENDING'.
            limit: Maximum number of articles to embed.
        
        Returns:
            Dict containing operation summary with counts and status.
        """
        # Initialize results dictionary
        results = {
            "articles_embedding_succeeded": 0,
            "articles_failed": 0,
            "status": "SUCCESS",
            "errors": [],
        }
        
        try:
            # Get articles based on the specified embedding status
            articles_for_embedding = self.article_manager.get_processed_articles_for_embedding(
                status=status, limit=limit
            )
            logger.info(f"Found {len(articles_for_embedding)} articles with embedding status '{status}' for processing")
            
            if not articles_for_embedding:
                logger.info(f"No articles with embedding status '{status}' found for processing")
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
                    
                    # Generate and verify embeddings for these chunks
                    embedding_result = self.embeddings_generator.generate_and_verify_embeddings(chunks)
                    chunk_embeddings = embedding_result["embeddings"]
                    all_embeddings_valid = embedding_result["all_valid"]
                    
                    if not all_embeddings_valid:
                        logger.warning(f"One or more chunk embeddings failed for article {url_hash}")
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            embedding_model=self.embeddings_generator.model_name,
                            error_message='One or more chunk embeddings failed (zero vector)'
                        )
                        results["articles_failed"] += 1
                        continue
                    
                    # Prepare article data for ChromaDB
                    article_data_for_chroma = {
                        'published_at': article.get('published_at'),
                        'source_query_tag': article.get('source_query_tag'),
                        'source_query_symbol': article.get('source_query_symbol')
                    }
                    
                    # If re-embedding (status was 'FAILED'), delete existing embeddings first
                    if status == 'FAILED':
                        logger.info(f"Deleting existing embeddings for article {url_hash} before re-embedding.")
                        self.chroma_manager.delete_embeddings_by_article(url_hash)
                        
                    # Store embeddings in ChromaDB using the new method
                    storage_success = self.chroma_manager.add_article_chunks(
                        url_hash,
                        chunks,
                        chunk_embeddings,
                        article_data_for_chroma
                    )
                    
                    if storage_success:
                        # Update article status in SQLite
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='SUCCESS',
                            embedding_model=self.embeddings_generator.model_name,
                            vector_db_id=url_hash  # Using url_hash as the vector_db_id
                        )
                        action = "re-embedded" if status == 'FAILED' else "embedded"
                        logger.info(f"Successfully {action} article {url_hash} in ChromaDB")
                        results["articles_embedding_succeeded"] += 1
                    else:
                        self.article_manager.update_article_embedding_status(
                            url_hash=url_hash,
                            status='FAILED',
                            embedding_model=self.embeddings_generator.model_name,
                            error_message='Failed to store embeddings in ChromaDB'
                        )
                        action = "re-embed" if status == 'FAILED' else "embed"
                        logger.error(f"Failed to {action} article {url_hash} in ChromaDB")
                        results["articles_failed"] += 1
                    
                except Exception as e:
                    action = "re-embedding" if status == 'FAILED' else "embedding"
                    logger.error(f"Error {action} article {url_hash}: {str(e)}")
                    self.article_manager.update_article_embedding_status(
                        url_hash=url_hash,
                        status='FAILED',
                        error_message=str(e)
                    )
                    results["articles_failed"] += 1
                    results["errors"].append(f"Error {action} article {url_hash}: {str(e)}")
            
            logger.info(
                f"For embedding status '{status}': "
                f"Successfully processed {results['articles_embedding_succeeded']} articles, "
                f"{results['articles_failed']} failed"
            )
            
        except Exception as e:
            logger.error(f"Error in embed_processed_articles (status: {status}): {str(e)}")
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
            # Call the ArticleManager's get_database_statistics method
            status = self.article_manager.get_database_statistics()
            
            logger.info(f"Article database status: {status.get('total_articles', 0)} total articles")
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
        rerank: bool = False,
        from_date_str: Optional[str] = None,
        to_date_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for articles relevant to a query.
        
        Args:
            query: The search query
            n_results: Maximum number of results to return
            sort_by_metadata: Dictionary for filtering/sorting based on metadata
                e.g., {"published_at_timestamp": "desc"}
            rerank: Whether to apply re-ranking with Gemini LLM
            from_date_str: Optional ISO format string (YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD)
                for filtering articles published on or after this date
            to_date_str: Optional ISO format string (YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD)
                for filtering articles published on or before this date
        
        Returns:
            List of article dictionaries with relevance scores
        """
        try:
            logger.info(f"Searching for articles with query: '{query}'")
            
            # Generate embedding for the query
            query_embedding = self.embeddings_generator.generate_embeddings([query])[0]
            
            # Initialize filter_metadata
            filter_metadata = {} if sort_by_metadata else None
            
            # Process sort_by_metadata
            if sort_by_metadata:
                for key, value in sort_by_metadata.items():
                    # For timestamp ranges, could create a range filter
                    if key == "published_at_timestamp" and value.lower() in ["desc", "asc"]:
                        # This would need to be implemented properly with actual date ranges
                        pass
            
            # Query the vector database with similarity scores
            chroma_results = self.chroma_manager.query_embeddings(
                query_embedding=query_embedding,
                n_results=n_results if not rerank else max(n_results * 2, 10),  # Get more results for reranking
                filter_metadata=filter_metadata,
                from_date_str=from_date_str,
                to_date_str=to_date_str,
                return_similarity_score=True  # Use the new parameter to get similarity scores
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
            article_scores = {}  # Track best score for each article
            
            # Use similarity scores directly from ChromaDBManager
            for result in chroma_results:
                url_hash = result['metadata'].get('article_url_hash')
                if url_hash:
                    # Use the similarity_score from ChromaDBManager
                    similarity = result.get('similarity_score', 0)
                    
                    # Update best score for this article
                    if url_hash not in article_scores or similarity > article_scores[url_hash]:
                        article_scores[url_hash] = similarity
            
            # Get article content and assign scores
            articles = []
            for url_hash in article_scores:
                article = self.article_manager.get_article_by_hash(url_hash)
                if article and article.get('processed_content'):
                    article['similarity_score'] = article_scores[url_hash]
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
    

    
    def delete_articles_older_than(self, days: int = 180) -> Dict[str, Any]:
        """
        Delete articles older than the specified number of days.
        
        Args:
            days: Number of days to use as the age threshold. Default is 180 (6 months).
                 Articles older than this will be deleted.
        
        Returns:
            Dict containing operation summary with counts and status:
            - targeted_articles: Number of articles targeted for deletion
            - deleted_from_sqlite: Number of articles successfully deleted from SQLite
            - deleted_from_chroma: Number of articles successfully deleted from ChromaDB
            - status: Overall operation status ('SUCCESS', 'PARTIAL_FAILURE', or 'FAILED')
            - errors: List of error messages for specific articles that failed deletion
        """
        # Initialize results dictionary
        results = {
            "targeted_articles": 0,
            "deleted_from_sqlite": 0,
            "deleted_from_chroma": 0,
            "status": "SUCCESS",
            "errors": []
        }
        
        try:
            # Calculate cutoff timestamp using utility functions
            current_time = utils.get_utc_now()
            cutoff_date = utils.get_cutoff_datetime(days)
            cutoff_timestamp = int(cutoff_date.timestamp())
            
            logger.info(f"Deleting articles older than {cutoff_date.isoformat()} ({days} days ago)")
            
            # Get article hashes older than the cutoff date
            article_hashes = self.chroma_manager.get_article_hashes_by_date_range(older_than_timestamp=cutoff_timestamp)
            results["targeted_articles"] = len(article_hashes)
            
            if not article_hashes:
                logger.info(f"No articles found older than {days} days")
                return results
            
            logger.info(f"Found {len(article_hashes)} articles older than {days} days that will be deleted")
            
            # Delete each article from both databases
            for url_hash in article_hashes:
                try:
                    # Delete from ChromaDB first
                    chroma_deleted = self.chroma_manager.delete_embeddings_by_article(url_hash)
                    if chroma_deleted:
                        results["deleted_from_chroma"] += 1
                    
                    # Delete from SQLite
                    sqlite_deleted = self.article_manager.delete_article_by_hash(url_hash)
                    if sqlite_deleted:
                        results["deleted_from_sqlite"] += 1
                    
                    # Log the result for this article
                    if chroma_deleted and sqlite_deleted:
                        logger.info(f"Successfully deleted article {url_hash} from both databases")
                    elif chroma_deleted:
                        logger.warning(f"Deleted embeddings for article {url_hash} but article not found in SQLite")
                        results["errors"].append(f"Article {url_hash} not found in SQLite")
                    elif sqlite_deleted:
                        logger.warning(f"Deleted article {url_hash} from SQLite but no embeddings found in ChromaDB")
                    else:
                        logger.warning(f"Article {url_hash} not found in either database")
                        results["errors"].append(f"Article {url_hash} not found in either database")
                    
                except Exception as e:
                    logger.error(f"Error deleting article {url_hash}: {str(e)}")
                    results["errors"].append(f"Error deleting article {url_hash}: {str(e)}")
            
            # Set overall status based on deletion results
            if results["deleted_from_sqlite"] == 0 and results["deleted_from_chroma"] == 0:
                results["status"] = "FAILED"
                logger.error("Failed to delete any articles")
            elif results["errors"]:
                results["status"] = "PARTIAL_FAILURE"
                logger.warning(f"Partially failed to delete some articles. {len(results['errors'])} errors occurred.")
            else:
                results["status"] = "SUCCESS"
            
            # Log summary
            logger.info(
                f"Delete operation summary: {results['deleted_from_sqlite']} deleted from SQLite, "
                f"{results['deleted_from_chroma']} deleted from ChromaDB, "
                f"{len(results['errors'])} errors"
            )
            
        except Exception as e:
            logger.error(f"Error in delete_articles_older_than: {str(e)}")
            results["status"] = "FAILED"
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
