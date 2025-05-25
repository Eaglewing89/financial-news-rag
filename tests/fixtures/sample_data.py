"""
Test data factories for generating consistent test data across the test suite.

This module provides factory functions and classes for creating test articles,
embeddings, and other data structures used throughout the test suite.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import random
import string
import numpy as np
from financial_news_rag.utils import generate_url_hash


class ArticleFactory:
    """Factory for creating test articles with realistic data."""
    
    SAMPLE_TITLES = [
        "Apple Reports Strong Q4 Earnings",
        "Microsoft Announces New Cloud Services",
        "Tech Stocks Rally on Market Optimism",
        "Federal Reserve Signals Interest Rate Changes", 
        "AI Revolution Transforms Healthcare Industry",
        "Renewable Energy Investments Surge Globally",
        "Cryptocurrency Market Shows Volatility",
        "Supply Chain Disruptions Affect Manufacturing"
    ]
    
    SAMPLE_CONTENT_FRAGMENTS = [
        "The company reported strong financial results for the quarter",
        "Market analysts are optimistic about future growth prospects",
        "Technological innovations continue to drive industry transformation",
        "Regulatory changes may impact business operations significantly",
        "Consumer demand remains robust despite economic uncertainties",
        "Strategic partnerships are expected to accelerate expansion plans",
        "Digital transformation initiatives show promising early results",
        "Sustainability efforts align with environmental regulations"
    ]
    
    SYMBOLS = [
        "AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US", "TSLA.US",
        "META.US", "NVDA.US", "JPM.US", "JNJ.US", "PG.US"
    ]
    
    TAGS = [
        "TECHNOLOGY", "FINANCE", "EARNINGS", "HEALTHCARE", "ENERGY",
        "AUTOMOTIVE", "TELECOMMUNICATIONS", "RETAIL", "MANUFACTURING"
    ]
    
    @classmethod
    def create_article(
        cls,
        title: Optional[str] = None,
        url: Optional[str] = None,
        published_at: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        source_query_tag: Optional[str] = None,
        source_query_symbol: Optional[str] = None,
        include_html: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a test article with realistic data.
        
        Args:
            title: Article title (random if None)
            url: Article URL (random if None)
            published_at: Publication date (recent random if None)
            symbols: Stock symbols (random selection if None)
            tags: Article tags (random selection if None)
            source_query_tag: Tag used for article query
            source_query_symbol: Symbol used for article query
            include_html: Whether to include HTML tags in content
            **kwargs: Additional article fields
            
        Returns:
            Dictionary representing a test article
        """
        # Generate random ID for uniqueness
        random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        # Use provided values or generate defaults
        if title is None:
            title = random.choice(cls.SAMPLE_TITLES)
            
        if url is None:
            url = f"https://example.com/article-{random_id}"
            
        if published_at is None:
            # Random time in the last 30 days
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            pub_time = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
            published_at = pub_time.isoformat()
            
        if symbols is None:
            symbols = random.sample(cls.SYMBOLS, random.randint(1, 3))
            
        if tags is None:
            tags = random.sample(cls.TAGS, random.randint(1, 2))
        
        # Generate content
        content_fragments = random.sample(cls.SAMPLE_CONTENT_FRAGMENTS, random.randint(2, 4))
        content = ". ".join(content_fragments) + "."
        
        if include_html:
            content = f"<p>{content}</p><p>Additional content with <b>HTML</b> formatting.</p>"
        
        # Base article structure
        article = {
            'title': title,
            'raw_content': content,
            'url': url,
            'published_at': published_at,
            'source_api': 'EODHD',
            'symbols': symbols,
            'tags': tags,
            'sentiment': {
                'polarity': round(random.uniform(-1, 1), 2),
                'neg': round(random.uniform(0, 0.5), 2),
                'neu': round(random.uniform(0.3, 0.7), 2),
                'pos': round(random.uniform(0, 0.5), 2)
            }
        }
        
        # Add query-specific fields if provided
        if source_query_tag:
            article['source_query_tag'] = source_query_tag
            
        if source_query_symbol:
            article['source_query_symbol'] = source_query_symbol
        
        # Add any additional fields
        article.update(kwargs)
        
        return article
    
    @classmethod
    def create_articles_batch(
        cls,
        count: int,
        **common_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create a batch of test articles.
        
        Args:
            count: Number of articles to create
            **common_kwargs: Common fields for all articles
            
        Returns:
            List of article dictionaries
        """
        return [cls.create_article(**common_kwargs) for _ in range(count)]
    
    @classmethod
    def create_eodhd_articles_batch(
        cls,
        count: int,
        symbols: Optional[List[str]] = None,
        **common_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create a batch of test articles with EODHD-specific formatting.
        
        Args:
            count: Number of articles to create
            symbols: List of symbols to assign to articles
            **common_kwargs: Common fields for all articles
            
        Returns:
            List of article dictionaries in EODHD format
        """
        articles = []
        for i in range(count):
            # If symbols provided, assign them cyclically to articles
            article_symbols = None
            if symbols:
                # Each article gets one symbol from the list
                article_symbols = [symbols[i % len(symbols)]]
            
            article = cls.create_article(symbols=article_symbols, **common_kwargs)
            articles.append(article)
        
        return articles
    
    @classmethod
    def create_article_with_status(
        cls,
        processing_status: str = "PENDING",
        embedding_status: str = "PENDING",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create an article with specific processing/embedding status.
        
        Args:
            processing_status: Text processing status
            embedding_status: Embedding generation status
            **kwargs: Additional article fields
            
        Returns:
            Article dictionary with status fields
        """
        article = cls.create_article(**kwargs)
        article['processing_status'] = processing_status
        article['embedding_status'] = embedding_status
        
        if processing_status == "SUCCESS":
            article['processed_content'] = "Processed content without HTML tags."
            
        return article


class ChunkFactory:
    """Factory for creating test text chunks and embeddings."""
    
    SAMPLE_CHUNKS = [
        "This is a sample text chunk about technology companies.",
        "Financial markets showed strong performance this quarter.",
        "Artificial intelligence is transforming various industries.",
        "Environmental sustainability remains a key business priority.",
        "Consumer behavior patterns are shifting toward digital platforms.",
        "Regulatory frameworks are adapting to technological changes.",
        "Global supply chains face ongoing disruption challenges.",
        "Innovation drives competitive advantage in modern markets."
    ]
    
    @classmethod
    def create_chunks(
        cls,
        count: int,
        custom_content: Optional[List[str]] = None
    ) -> List[str]:
        """
        Create a list of text chunks for testing.
        
        Args:
            count: Number of chunks to create
            custom_content: Custom chunk content (uses samples if None)
            
        Returns:
            List of text chunks
        """
        if custom_content:
            return custom_content[:count]
        
        if count <= len(cls.SAMPLE_CHUNKS):
            return random.sample(cls.SAMPLE_CHUNKS, count)
        else:
            # Repeat samples if more chunks needed
            repeated = cls.SAMPLE_CHUNKS * ((count // len(cls.SAMPLE_CHUNKS)) + 1)
            return repeated[:count]
    
    @classmethod
    def create_embeddings(
        cls,
        count: int,
        dimension: int = 768
    ) -> List[List[float]]:
        """
        Create mock embeddings for testing.
        
        Args:
            count: Number of embeddings to create
            dimension: Embedding dimension
            
        Returns:
            List of embedding vectors
        """
        return [np.random.rand(dimension).tolist() for _ in range(count)]
    
    @classmethod
    def create_article_chunks_data(
        cls,
        article_url_hash: str,
        chunk_count: int = 3,
        embedding_dimension: int = 768,
        published_at: Optional[str] = None,
        source_query_tag: Optional[str] = None,
        source_query_symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create complete article chunks data for ChromaDB testing.
        
        Args:
            article_url_hash: Hash of the article URL
            chunk_count: Number of chunks to create
            embedding_dimension: Dimension of embeddings
            published_at: Publication timestamp
            source_query_tag: Tag used for querying
            source_query_symbol: Symbol used for querying
            
        Returns:
            Dictionary with chunks, embeddings, and metadata
        """
        chunks = cls.create_chunks(chunk_count)
        embeddings = cls.create_embeddings(chunk_count, embedding_dimension)
        
        article_data = {}
        if published_at:
            article_data['published_at'] = published_at
        if source_query_tag:
            article_data['source_query_tag'] = source_query_tag
        if source_query_symbol:
            article_data['source_query_symbol'] = source_query_symbol
        
        return {
            'article_url_hash': article_url_hash,
            'chunk_texts': chunks,
            'chunk_vectors': embeddings,
            'article_data': article_data
        }


class EODHDResponseFactory:
    """Factory for creating mock EODHD API responses."""
    
    @classmethod
    def create_response(
        cls,
        count: int = 2,
        articles: Optional[List[Dict[str, Any]]] = None,
        symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a raw EODHD API response (list of articles).
        
        Args:
            count: Number of articles to generate
            articles: List of articles (generates if None)
            symbols: List of symbols to include in generated articles
            
        Returns:
            List of article dictionaries (raw EODHD format)
        """
        if articles is None:
            articles = ArticleFactory.create_eodhd_articles_batch(count, symbols=symbols)
        return articles
    
    @classmethod
    def create_success_response(
        cls,
        articles: Optional[List[Dict[str, Any]]] = None,
        count: int = 2
    ) -> Dict[str, Any]:
        """
        Create a successful EODHD API response.
        
        Args:
            articles: List of articles (generates if None)
            count: Number of articles to generate if articles is None
            
        Returns:
            Mock API response dictionary
        """
        if articles is None:
            articles = ArticleFactory.create_articles_batch(count)
        
        return {
            "articles": articles,
            "status_code": 200,
            "success": True,
            "message": "Successfully fetched articles"
        }
    
    @classmethod
    def create_error_response(
        cls,
        status_code: int = 400,
        error_message: str = "API request failed"
    ) -> Dict[str, Any]:
        """
        Create an error EODHD API response.
        
        Args:
            status_code: HTTP status code
            error_message: Error message
            
        Returns:
            Mock error response dictionary
        """
        return {
            "articles": [],
            "status_code": status_code,
            "success": False,
            "message": error_message
        }
    
    @classmethod
    def create_empty_response(cls) -> Dict[str, Any]:
        """
        Create an empty but successful EODHD API response.
        
        Returns:
            Mock empty response dictionary
        """
        return {
            "articles": [],
            "status_code": 200,
            "success": True,
            "message": "No articles found"
        }


class ChromaResultsFactory:
    """Factory for creating mock ChromaDB query results."""
    
    @classmethod
    def create_query_results(
        cls,
        count: int,
        article_hashes: Optional[List[str]] = None,
        distances: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create mock ChromaDB query results.
        
        Args:
            count: Number of results to create
            article_hashes: Specific article hashes to use
            distances: Specific distances to use
            
        Returns:
            List of query result dictionaries
        """
        if article_hashes is None:
            article_hashes = [f"article_hash_{i}" for i in range(count)]
        
        if distances is None:
            distances = [round(random.uniform(0.1, 0.9), 3) for _ in range(count)]
        
        results = []
        for i in range(count):
            chunk_text = ChunkFactory.SAMPLE_CHUNKS[i % len(ChunkFactory.SAMPLE_CHUNKS)]
            
            result = {
                "chunk_id": f"{article_hashes[i % len(article_hashes)]}_{i}",
                "text": chunk_text,
                "metadata": {
                    "article_url_hash": article_hashes[i % len(article_hashes)],
                    "chunk_index": i,
                    "published_at_timestamp": int(datetime.now(timezone.utc).timestamp()) - (i * 3600)
                },
                "distance": distances[i % len(distances)]
            }
            
            # Add optional metadata
            if i % 2 == 0:
                result["metadata"]["source_query_tag"] = random.choice(ArticleFactory.TAGS)
            else:
                result["metadata"]["source_query_symbol"] = random.choice(ArticleFactory.SYMBOLS)
            
            results.append(result)
        
        return results


class EmbeddingFactory:
    """Factory for creating test embeddings with various configurations."""
    
    @classmethod
    def create_normalized_embedding(cls, dimension: int = 768) -> np.ndarray:
        """
        Create a normalized random embedding vector for testing.
        
        Args:
            dimension: Embedding dimension
            
        Returns:
            Normalized numpy array
        """
        embedding = np.random.rand(dimension)
        return embedding / np.linalg.norm(embedding)
    
    @classmethod
    def create_normalized_embeddings(cls, count: int, dimension: int = 768) -> List[np.ndarray]:
        """
        Create multiple normalized embedding vectors for testing.
        
        Args:
            count: Number of embeddings to create
            dimension: Embedding dimension
            
        Returns:
            List of normalized numpy arrays
        """
        return [cls.create_normalized_embedding(dimension) for _ in range(count)]
    
    @classmethod
    def create_similar_embeddings(
        cls, 
        base_embedding: Optional[np.ndarray] = None, 
        count: int = 3, 
        dimension: int = 768,
        noise_factor: float = 0.1
    ) -> List[np.ndarray]:
        """
        Create embeddings that are similar to a base embedding for testing similarity searches.
        
        Args:
            base_embedding: Base embedding to create similar ones from
            count: Number of similar embeddings to create
            dimension: Embedding dimension
            noise_factor: Amount of noise to add (0.0 = identical, 1.0 = completely random)
            
        Returns:
            List of similar normalized numpy arrays
        """
        if base_embedding is None:
            base_embedding = cls.create_normalized_embedding(dimension)
        
        similar_embeddings = []
        for _ in range(count):
            # Add small random noise to the base embedding
            noise = np.random.rand(dimension) * noise_factor
            similar = base_embedding + noise
            # Normalize the result
            similar_embeddings.append(similar / np.linalg.norm(similar))
        
        return similar_embeddings


# Convenience functions for quick data creation
def create_test_article(**kwargs) -> Dict[str, Any]:
    """Quick function to create a test article."""
    return ArticleFactory.create_article(**kwargs)


def create_test_articles(count: int, **kwargs) -> List[Dict[str, Any]]:
    """Quick function to create multiple test articles."""
    return ArticleFactory.create_articles_batch(count, **kwargs)


def create_test_chunks(count: int = 3) -> List[str]:
    """Quick function to create test chunks."""
    return ChunkFactory.create_chunks(count)


def create_test_embeddings(count: int = 3, dimension: int = 768) -> List[List[float]]:
    """Quick function to create test embeddings."""
    return ChunkFactory.create_embeddings(count, dimension)


def create_test_article_with_hash(**kwargs) -> Dict[str, Any]:
    """
    Create a test article with url_hash field included.
    
    This is a convenience function that creates an article and automatically
    generates the url_hash field using the article's URL.
    
    Args:
        **kwargs: Arguments to pass to ArticleFactory.create_article()
        
    Returns:
        Article dictionary with url_hash field included
    """
    from financial_news_rag.utils import generate_url_hash
    
    article = ArticleFactory.create_article(**kwargs)
    article['url_hash'] = generate_url_hash(article['url'])
    return article


def create_test_normalized_embedding(dimension: int = 768) -> np.ndarray:
    """Quick function to create a normalized embedding."""
    return EmbeddingFactory.create_normalized_embedding(dimension)


def create_test_normalized_embeddings(count: int, dimension: int = 768) -> List[np.ndarray]:
    """Quick function to create multiple normalized embeddings."""
    return EmbeddingFactory.create_normalized_embeddings(count, dimension)
