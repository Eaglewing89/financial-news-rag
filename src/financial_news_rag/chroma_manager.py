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
    
    def add_embeddings(self, article_url_hash: str, chunk_embeddings: List[Dict[str, Any]]) -> bool:
        """
        Add chunk embeddings for an article to ChromaDB.
        
        Args:
            article_url_hash: The unique identifier from the SQLite articles table.
            chunk_embeddings: A list of dictionaries, where each contains:
                - chunk_id: Unique ID for the chunk (e.g., "{article_url_hash}_{chunk_index}")
                - embedding: The embedding vector (list of floats)
                - text: The actual text of the chunk
                - metadata: A dictionary containing minimal metadata:
                  - article_url_hash: The url_hash of the parent article
                  - chunk_index: Index of this chunk within the article
                  - published_at_timestamp: Optional timestamp for date filtering
                  - ... other metadata for direct filtering in ChromaDB
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not chunk_embeddings:
            logger.warning(f"No chunk embeddings provided for article {article_url_hash}")
            return False
            
        try:
            # Prepare data for batch addition to ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for chunk in chunk_embeddings:
                # Extract required fields
                chunk_id = chunk.get('chunk_id')
                embedding = chunk.get('embedding')
                text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                
                # Validate required fields
                if not chunk_id or not embedding:
                    logger.warning(f"Skipping chunk with missing chunk_id or embedding: {chunk}")
                    continue
                
                # Ensure article_url_hash is in metadata
                if 'article_url_hash' not in metadata:
                    metadata['article_url_hash'] = article_url_hash
                
                ids.append(chunk_id)
                embeddings.append(embedding)
                metadatas.append(metadata)
                documents.append(text)
            
            if not ids:
                logger.warning(f"No valid chunk embeddings to add for article {article_url_hash}")
                return False
            
            # Add embeddings to ChromaDB collection
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Successfully added {len(ids)} chunk embeddings for article {article_url_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings for article {article_url_hash}: {e}")
            return False
    
    def query_embeddings(
        self, 
        query_embedding: List[float], 
        n_results: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for the most similar embeddings to the query embedding.
        
        Args:
            query_embedding: The embedding vector of the search query.
            n_results: Number of similar embeddings to retrieve.
            filter_metadata: Optional dictionary for metadata filtering
                (e.g., {"article_url_hash": "some_hash"})
            
        Returns:
            List of results, each including chunk_id, distance/score, metadata, and text
        """
        try:
            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["metadatas", "documents", "distances"]
            }
            
            # Add metadata filter if provided
            if filter_metadata:
                where_clause = {}
                for key, value in filter_metadata.items():
                    where_clause[key] = value
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
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                        'text': results['documents'][0][i] if 'documents' in results else None
                    }
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
