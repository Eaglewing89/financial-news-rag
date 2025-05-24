"""
ChromaDB Manager for Financial News RAG.

This module provides a class for managing ChromaDB operations, including:
- Initializing ChromaDB client and collection
- Adding and retrieving embeddings
- Managing associations between ChromaDB entries and the SQLite database
- Querying the vector database for similar embeddings
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union

from chromadb import Client, Collection
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    A class for managing ChromaDB operations for financial news article embeddings.
    
    This class handles:
    - Initializing and connecting to ChromaDB
    - Adding article chunk embeddings to the vector store
    - Querying for similar embeddings
    - Maintaining links between ChromaDB entries and SQLite database
    - Status reporting and error handling
    """
    
    DEFAULT_COLLECTION_NAME = "financial_news_embeddings"
    DEFAULT_EMBEDDING_DIMENSION = 768  # text-embedding-004 dimension
    
    def __init__(self, 
                 persist_directory: Optional[str] = None,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
                 in_memory: bool = False):
        """
        Initialize the ChromaDBManager with connection parameters.
        
        Args:
            persist_directory: Path for persistent storage of embeddings.
                If None and not in_memory, uses './chroma_db' in current directory.
            collection_name: Name of the ChromaDB collection to use.
            embedding_dimension: Dimension of the embedding vectors.
            in_memory: Whether to use an in-memory ChromaDB instance (for testing).
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.in_memory = in_memory
        
        # Set up persistence directory if not in-memory
        if in_memory:
            persist_directory = None
            logger.info("Initializing in-memory ChromaDB instance")
        else:
            if not persist_directory:
                # Use default directory relative to current working directory
                persist_directory = os.path.join(os.getcwd(), 'chroma_db')
            # Ensure the directory exists
            os.makedirs(persist_directory, exist_ok=True)
            logger.info(f"Using ChromaDB persistence directory: {persist_directory}")
            
        self.persist_directory = persist_directory
        
        # Initialize client and collection
        self._initialize_client_and_collection()
    
    def _initialize_client_and_collection(self) -> None:
        """Initialize the ChromaDB client and get or create the collection."""
        try:
            # Create ChromaDB client
            if self.in_memory:
                self.client = chromadb.Client(Settings(is_persistent=False))
            else:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Try to get the existing collection or create a new one
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
            except (ValueError, chromadb.errors.NotFoundError):
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"dimension": self.embedding_dimension}
                )
                logger.info(f"Created new collection: {self.collection_name} with dimension {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    # The add_embeddings method has been removed as it's not used by the FinancialNewsRAG orchestrator.
    # The add_article_chunks method handles the needed functionality.
    
    def query_embeddings(
        self, 
        query_embedding: List[float], 
        n_results: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        from_date_str: Optional[str] = None,
        to_date_str: Optional[str] = None,
        return_similarity_score: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for the most similar embeddings to the query embedding.
        
        Args:
            query_embedding: The embedding vector of the search query.
            n_results: Number of similar embeddings to retrieve.
            filter_metadata: Optional dictionary for metadata filtering
                (e.g., {"article_url_hash": "some_hash"})
            from_date_str: Optional ISO format string (YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD)
                for filtering articles published on or after this date
            to_date_str: Optional ISO format string (YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD)
                for filtering articles published on or before this date
            return_similarity_score: If True, converts distance to similarity score
                and returns it in place of distance in the results.
            
        Returns:
            List of results, each including chunk_id, distance/similarity_score, metadata, and text
        """
        try:
            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["metadatas", "documents", "distances"]
            }
            
            # Process from_date_str and to_date_str
            timestamp_filter_conditions = {}
            
            # Parse from_date_str if provided
            if from_date_str:
                try:
                    from datetime import datetime
                    # Try to parse the date string into a datetime object
                    dt = datetime.fromisoformat(from_date_str.replace('Z', '+00:00'))
                    from_timestamp = int(dt.timestamp())
                    timestamp_filter_conditions["$gte"] = from_timestamp
                    logger.info(f"Filtering articles published on or after {from_date_str} (timestamp {from_timestamp})")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse from_date_str '{from_date_str}': {e}")
            
            # Parse to_date_str if provided
            if to_date_str:
                try:
                    from datetime import datetime
                    # Try to parse the date string into a datetime object
                    dt = datetime.fromisoformat(to_date_str.replace('Z', '+00:00'))
                    to_timestamp = int(dt.timestamp())
                    timestamp_filter_conditions["$lte"] = to_timestamp
                    logger.info(f"Filtering articles published on or before {to_date_str} (timestamp {to_timestamp})")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse to_date_str '{to_date_str}': {e}")
            
            # Initialize where_clause 
            where_clause = {}
            
            # Process filter_metadata and timestamp filters together
            if filter_metadata or timestamp_filter_conditions:
                # When we have multiple conditions with different fields, we need to use $and
                and_conditions = []
                
                # Add metadata filter conditions
                if filter_metadata:
                    for key, value in filter_metadata.items():
                        # Skip published_at_timestamp as it will be handled separately
                        if key != 'published_at_timestamp':
                            and_conditions.append({key: value})
                
                # Add timestamp filter conditions
                if timestamp_filter_conditions:
                    for op, val in timestamp_filter_conditions.items():
                        and_conditions.append({"published_at_timestamp": {op: val}})
                
                # If we have more than one condition, use $and
                if len(and_conditions) > 1:
                    where_clause["$and"] = and_conditions
                elif len(and_conditions) == 1:
                    # If only one condition, add it directly
                    key, value = next(iter(and_conditions[0].items()))
                    where_clause[key] = value
            
            # Add where clause to query params if not empty
            if where_clause:
                query_params["where"] = where_clause
            
            # Execute query
            results = self.collection.query(**query_params)
            
            # Format response
            formatted_results = []
            if results and results['ids']:
                # Get the first (and only) query result
                for i, chunk_id in enumerate(results['ids'][0]):
                    result = {
                        'chunk_id': chunk_id,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                        'text': results['documents'][0][i] if 'documents' in results else None
                    }
                    
                    # Add distance or similarity_score based on return_similarity_score parameter
                    if 'distances' in results:
                        distance = results['distances'][0][i]
                        if return_similarity_score:
                            # Convert distance to similarity score using the formula: 1.0 - (distance / 2.0)
                            result['similarity_score'] = 1.0 - (distance / 2.0)
                        else:
                            result['distance'] = distance
                    
                    formatted_results.append(result)
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error querying embeddings: {e}")
            return []
    
    def get_collection_status(self) -> Dict[str, Any]:
        """
        Get status information about the ChromaDB collection.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            # Get collection details
            count = self.collection.count()
            
            # Query for any entry to check if collection is empty
            peek_results = self.collection.peek(limit=1)
            is_empty = len(peek_results.get('ids', [])) == 0
            
            # Get unique article count (unique article_url_hash values)
            unique_articles = set()
            if not is_empty:
                # If collection has entries, query for all article_url_hash values
                # Note: For large collections, we might want to approach this differently
                all_metadatas = self.collection.get(
                    include=["metadatas"],
                    limit=count  # Get all entries
                )
                
                if all_metadatas and all_metadatas['metadatas']:
                    for metadata in all_metadatas['metadatas']:
                        article_hash = metadata.get('article_url_hash')
                        if article_hash:
                            unique_articles.add(article_hash)
            
            return {
                'collection_name': self.collection_name,
                'total_chunks': count,
                'unique_articles': len(unique_articles),
                'embedding_dimension': self.embedding_dimension,
                'is_empty': is_empty,
                'persist_directory': self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection status: {e}")
            return {
                'error': str(e),
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
    
    def delete_embeddings_by_article(self, article_url_hash: str) -> bool:
        """
        Delete all embeddings associated with a given article_url_hash.
        
        Args:
            article_url_hash: The unique identifier of the article
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Using the 'where' clause to find all chunks for this article
            self.collection.delete(
                where={"article_url_hash": article_url_hash}
            )
            logger.info(f"Successfully deleted embeddings for article {article_url_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings for article {article_url_hash}: {e}")
            return False
    
    def add_article_chunks(self, article_url_hash: str, chunk_texts: List[str], 
                          chunk_vectors: List[List[float]], article_data: Dict[str, Any]) -> bool:
        """
        Add chunk data for an article to ChromaDB.
        
        Args:
            article_url_hash: The unique identifier from the SQLite articles table.
            chunk_texts: A list of raw text chunks from the article.
            chunk_vectors: A list of embedding vectors corresponding to each chunk.
            article_data: A dictionary containing article-level information like:
                - published_at: ISO format date string (optional)
                - source_query_tag: Source query tag (optional)
                - source_query_symbol: Source query symbol (optional)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize empty lists for batch addition to ChromaDB
            ids = []
            embeddings_list = []
            metadatas_list = []
            documents_list = []
            
            # Process each chunk and create metadata
            for i, (chunk_text, chunk_vector) in enumerate(zip(chunk_texts, chunk_vectors)):
                # Create unique chunk ID
                chunk_id = f"{article_url_hash}_{i}"
                
                # Create metadata dictionary
                current_metadata = {
                    "article_url_hash": article_url_hash,
                    "chunk_index": i
                }
                
                # Process published_at if available
                published_at_str = article_data.get('published_at')
                if published_at_str is not None:
                    try:
                        from datetime import datetime
                        # Convert ISO format to UNIX timestamp
                        dt = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                        timestamp = int(dt.timestamp())
                        current_metadata["published_at_timestamp"] = timestamp
                    except (ValueError, AttributeError, TypeError) as e:
                        logger.warning(f"Failed to convert published_at date '{published_at_str}' to timestamp: {e}")
                
                # Add additional metadata if available
                if article_data.get('source_query_tag'):
                    current_metadata["source_query_tag"] = article_data.get('source_query_tag')
                
                if article_data.get('source_query_symbol'):
                    current_metadata["source_query_symbol"] = article_data.get('source_query_symbol')
                
                # Add to batch lists
                ids.append(chunk_id)
                embeddings_list.append(chunk_vector)
                metadatas_list.append(current_metadata)
                documents_list.append(chunk_text)
            
            # Check if we have any valid chunks to add
            if not ids:
                logger.warning(f"No valid chunk embeddings to add for article {article_url_hash}")
                return False
            
            # Add embeddings to ChromaDB collection
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas_list,
                documents=documents_list
            )
            
            logger.info(f"Successfully upserted {len(ids)} chunk embeddings for article {article_url_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting embeddings for article {article_url_hash}: {e}")
            return False
        
    def get_article_hashes_by_date_range(self, older_than_timestamp: Optional[int] = None, newer_than_timestamp: Optional[int] = None) -> List[str]:
        """
        Retrieve article hashes based on published_at_timestamp within a specified date range.
        
        Args:
            older_than_timestamp: Optional upper bound timestamp (inclusive).
                Will filter for published_at_timestamp <= older_than_timestamp
            newer_than_timestamp: Optional lower bound timestamp (inclusive).
                Will filter for published_at_timestamp >= newer_than_timestamp
                
        Returns:
            List[str]: A list of unique article_url_hash values that match the criteria.
                Returns an empty list if no criteria are specified or if an error occurs.
        """
        try:
            # Check if at least one timestamp filter is provided
            if older_than_timestamp is None and newer_than_timestamp is None:
                logger.warning("No timestamp criteria provided for get_article_hashes_by_date_range. Returning empty list.")
                return []
            
            # Construct where clause based on provided parameters
            where_clause = {}
            
            # ChromaDB requires separate where conditions for multiple operators on the same field
            if older_than_timestamp is not None and newer_than_timestamp is not None:
                # Both timestamps provided - need to use $and operator for multiple conditions
                where_clause = {"$and": [
                    {"published_at_timestamp": {"$lte": older_than_timestamp}},
                    {"published_at_timestamp": {"$gte": newer_than_timestamp}}
                ]}
            elif older_than_timestamp is not None:
                # Only older_than provided
                where_clause = {"published_at_timestamp": {"$lte": older_than_timestamp}}
            elif newer_than_timestamp is not None:
                # Only newer_than provided
                where_clause = {"published_at_timestamp": {"$gte": newer_than_timestamp}}
            
            # Query ChromaDB with the where clause
            results = self.collection.get(
                where=where_clause, 
                include=["metadatas"]
            )
            
            # Extract unique article_url_hash values from metadata
            unique_article_hashes = set()
            
            if results and 'metadatas' in results and results['metadatas']:
                for metadata in results['metadatas']:
                    article_hash = metadata.get('article_url_hash')
                    if article_hash:
                        unique_article_hashes.add(article_hash)
            
            return list(unique_article_hashes)
            
        except Exception as e:
            logger.error(f"Error retrieving article hashes by date range: {e}")
            return []
