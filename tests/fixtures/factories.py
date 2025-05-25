"""
Factory classes for creating test objects with realistic data.

This module provides factory classes that can generate complex test objects
with relationships and realistic data for comprehensive testing.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import random

try:
    import factory
    import factory.fuzzy
    from factory import Factory, Faker, SubFactory, LazyFunction, LazyAttribute
    FACTORY_BOY_AVAILABLE = True
except ImportError:
    # Fallback to simple dictionary factories if factory_boy is not available
    FACTORY_BOY_AVAILABLE = False


if FACTORY_BOY_AVAILABLE:
    
    class ArticleFactory(Factory):
        """Factory for creating Article objects using factory_boy."""
        
        class Meta:
            model = dict  # We're creating dictionaries, not model instances
        
        title = Faker('sentence', nb_words=4)
        url = Faker('url')
        published_at = factory.LazyFunction(
            lambda: (datetime.now(timezone.utc) - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23)
            )).isoformat()
        )
        source_api = 'EODHD'
        raw_content = factory.LazyFunction(
            lambda: f"<p>{Faker('text').generate()}</p><p>Additional content with <b>HTML</b> formatting.</p>"
        )
        
        @factory.lazy_attribute
        def symbols(self):
            symbols_pool = ["AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US", "TSLA.US"]
            return random.sample(symbols_pool, random.randint(1, 3))
        
        @factory.lazy_attribute  
        def tags(self):
            tags_pool = ["TECHNOLOGY", "FINANCE", "EARNINGS", "HEALTHCARE", "ENERGY"]
            return random.sample(tags_pool, random.randint(1, 2))
        
        @factory.lazy_attribute
        def sentiment(self):
            return {
                'polarity': round(random.uniform(-1, 1), 2),
                'neg': round(random.uniform(0, 0.5), 2),
                'neu': round(random.uniform(0.3, 0.7), 2),
                'pos': round(random.uniform(0, 0.5), 2)
            }
    
    
    class TechnologyArticleFactory(ArticleFactory):
        """Factory for technology-specific articles."""
        
        title = factory.LazyFunction(
            lambda: random.choice([
                "Apple Reports Strong Q4 Earnings",
                "Microsoft Announces New Cloud Services", 
                "Tech Stocks Rally on Market Optimism",
                "AI Revolution Transforms Healthcare Industry"
            ])
        )
        tags = ["TECHNOLOGY"]
        source_query_tag = "TECHNOLOGY"
        
        @factory.lazy_attribute
        def symbols(self):
            return random.sample(["AAPL.US", "MSFT.US", "GOOGL.US"], 1)
    
    
    class FinanceArticleFactory(ArticleFactory):
        """Factory for finance-specific articles."""
        
        title = factory.LazyFunction(
            lambda: random.choice([
                "Federal Reserve Signals Interest Rate Changes",
                "Banking Sector Shows Resilience",
                "Cryptocurrency Market Shows Volatility",
                "Investment Outlook Remains Positive"
            ])
        )
        tags = ["FINANCE"]
        source_query_tag = "FINANCE"
        
        @factory.lazy_attribute
        def symbols(self):
            return random.sample(["JPM.US", "BAC.US", "WFC.US"], 1)
    
    
    class ProcessedArticleFactory(ArticleFactory):
        """Factory for articles that have been processed."""
        
        processing_status = "SUCCESS"
        processed_content = Faker('text')
        
        @factory.lazy_attribute
        def processed_at(self):
            pub_time = datetime.fromisoformat(self.published_at.replace('Z', '+00:00'))
            return (pub_time + timedelta(minutes=random.randint(1, 60))).isoformat()


else:
    # Fallback simple factories when factory_boy is not available
    class SimpleArticleFactory:
        """Simple article factory without factory_boy dependency."""
        
        @staticmethod
        def create(**kwargs):
            """Create a simple test article."""
            from .sample_data import ArticleFactory
            return ArticleFactory.create_article(**kwargs)
        
        @staticmethod
        def create_batch(size, **kwargs):
            """Create multiple test articles."""
            from .sample_data import ArticleFactory
            return ArticleFactory.create_articles_batch(size, **kwargs)
    
    # Alias for consistency
    ArticleFactory = SimpleArticleFactory


class TestScenarioFactory:
    """Factory for creating complete test scenarios with related data."""
    
    @classmethod
    def create_full_pipeline_scenario(cls) -> Dict[str, Any]:
        """
        Create a complete test scenario for the full RAG pipeline.
        
        Returns:
            Dictionary containing all related test data for a complete scenario
        """
        from .sample_data import ArticleFactory, ChunkFactory, EODHDResponseFactory
        
        # Create source articles
        raw_articles = ArticleFactory.create_articles_batch(3, source_query_tag="TECHNOLOGY")
        
        # Create EODHD API response
        api_response = EODHDResponseFactory.create_success_response(raw_articles)
        
        # Create processed chunks for each article
        article_chunks = {}
        for article in raw_articles:
            from financial_news_rag.utils import generate_url_hash
            url_hash = generate_url_hash(article['url'])
            chunks_data = ChunkFactory.create_article_chunks_data(
                article_url_hash=url_hash,
                chunk_count=3,
                published_at=article['published_at'],
                source_query_tag=article.get('source_query_tag')
            )
            article_chunks[url_hash] = chunks_data
        
        return {
            'raw_articles': raw_articles,
            'api_response': api_response,
            'article_chunks': article_chunks,
            'scenario_type': 'full_pipeline'
        }
    
    @classmethod
    def create_search_scenario(cls) -> Dict[str, Any]:
        """
        Create a test scenario for article search functionality.
        
        Returns:
            Dictionary containing search-related test data
        """
        from .sample_data import ChunkFactory, ChromaResultsFactory
        
        # Create query
        search_query = "artificial intelligence technology trends"
        query_embedding = ChunkFactory.create_embeddings(1, 768)[0]
        
        # Create mock ChromaDB results
        chroma_results = ChromaResultsFactory.create_query_results(5)
        
        # Create corresponding articles
        articles = {}
        for result in chroma_results:
            article_hash = result['metadata']['article_url_hash']
            if article_hash not in articles:
                article = ArticleFactory.create_article()
                article['url_hash'] = article_hash
                articles[article_hash] = article
        
        return {
            'search_query': search_query,
            'query_embedding': query_embedding,
            'chroma_results': chroma_results,
            'articles': articles,
            'scenario_type': 'search'
        }
    
    @classmethod
    def create_embedding_scenario(cls) -> Dict[str, Any]:
        """
        Create a test scenario for embedding generation and storage.
        
        Returns:
            Dictionary containing embedding-related test data
        """
        from .sample_data import ArticleFactory, ChunkFactory
        
        # Create articles with processed content
        articles = []
        for i in range(3):
            article = ArticleFactory.create_article_with_status(
                processing_status="SUCCESS",
                embedding_status="PENDING"
            )
            article['processed_content'] = f"Processed content for article {i+1}."
            articles.append(article)
        
        # Create chunks and embeddings for each article
        embedding_data = {}
        for article in articles:
            from financial_news_rag.utils import generate_url_hash
            url_hash = generate_url_hash(article['url'])
            
            chunks = ChunkFactory.create_chunks(3)
            embeddings = ChunkFactory.create_embeddings(3, 768)
            
            embedding_data[url_hash] = {
                'article': article,
                'chunks': chunks,
                'embeddings': embeddings
            }
        
        return {
            'articles': articles,
            'embedding_data': embedding_data,
            'scenario_type': 'embedding'
        }
    
    @classmethod
    def create_error_scenario(cls, error_type: str = "api_error") -> Dict[str, Any]:
        """
        Create a test scenario for error handling.
        
        Args:
            error_type: Type of error to simulate
            
        Returns:
            Dictionary containing error scenario data
        """
        from .sample_data import EODHDResponseFactory
        
        scenarios = {
            "api_error": {
                "api_response": EODHDResponseFactory.create_error_response(500, "Internal server error"),
                "expected_exception": Exception,
                "error_message": "EODHD API error"
            },
            "empty_response": {
                "api_response": EODHDResponseFactory.create_empty_response(),
                "expected_articles": 0,
                "error_message": None
            },
            "processing_error": {
                "articles": ArticleFactory.create_articles_batch(2),
                "processing_error": True,
                "expected_status": "FAILED"
            }
        }
        
        scenario = scenarios.get(error_type, scenarios["api_error"])
        scenario['scenario_type'] = f'error_{error_type}'
        
        return scenario


class ConfigDataFactory:
    """Factory for creating configuration test data."""
    
    @classmethod
    def create_eodhd_env_overrides(cls) -> Dict[str, str]:
        """Create environment variables for EODHD configuration overrides."""
        return {
            "EODHD_API_KEY": "test_eodhd_api_key",
            "EODHD_API_URL_OVERRIDE": "https://test.api.url",
            "EODHD_DEFAULT_TIMEOUT_OVERRIDE": "200",
            "EODHD_DEFAULT_MAX_RETRIES_OVERRIDE": "5",
            "EODHD_DEFAULT_BACKOFF_FACTOR_OVERRIDE": "2.0",
            "EODHD_DEFAULT_LIMIT_OVERRIDE": "100"
        }
    
    @classmethod
    def create_gemini_env_overrides(cls) -> Dict[str, str]:
        """Create environment variables for Gemini configuration overrides."""
        return {
            "GEMINI_API_KEY": "test_gemini_api_key",
            "EMBEDDINGS_DEFAULT_MODEL": "custom-embedding-model",
            "EMBEDDINGS_DEFAULT_TASK_TYPE": "CUSTOM_TASK",
            "EMBEDDINGS_MODEL_DIMENSIONS": '{"custom-embedding-model": 1024, "text-embedding-004": 999}',
            "RERANKER_DEFAULT_MODEL": "gemini-3.0-pro",
            "TEXTPROCESSOR_MAX_TOKENS_PER_CHUNK": "3000"
        }
    
    @classmethod
    def create_all_env_overrides(cls) -> Dict[str, str]:
        """Create complete set of environment variable overrides."""
        overrides = {}
        overrides.update(cls.create_eodhd_env_overrides())
        overrides.update(cls.create_gemini_env_overrides())
        return overrides
    
    @classmethod
    def create_database_path_override(cls, custom_path: str = "/custom/path/to/database.db") -> Dict[str, str]:
        """Create environment variable for database path override."""
        return {"DATABASE_PATH_OVERRIDE": custom_path}
    
    @classmethod
    def create_invalid_json_dimensions(cls) -> Dict[str, str]:
        """Create environment variables with invalid JSON for testing error handling."""
        return {
            "EMBEDDINGS_MODEL_DIMENSIONS": "invalid json"
        }
