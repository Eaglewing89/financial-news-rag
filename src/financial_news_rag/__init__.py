"""
Financial News RAG Module.

This package implements a Retrieval Augmented Generation (RAG) system for financial news,
providing functions to fetch, process, and search news articles with semantic understanding.
"""

from financial_news_rag.chroma_manager import ChromaDBManager
from financial_news_rag.config import get_config
from financial_news_rag.data import (
    fetch_marketaux_news_snippets,
    normalize_marketaux_news_data,
    search_marketaux_entities,
    get_marketaux_entity_types,
    get_marketaux_industry_list,
    get_marketaux_news_sources
)
from financial_news_rag.embeddings import EmbeddingsGenerator
from financial_news_rag.eodhd import EODHDClient, EODHDApiError
from financial_news_rag.marketaux import MarketauxNewsFetcher, MarketauxRateLimiter, fetch_marketaux_with_retry
from financial_news_rag.text_processor import TextProcessingPipeline, clean_text, split_text

__all__ = [
    # EODHD API
    "EODHDClient",
    "EODHDApiError",
    
    # Marketaux API
    "MarketauxNewsFetcher", 
    "MarketauxRateLimiter", 
    "fetch_marketaux_with_retry",
    
    # Configuration
    "get_config",
    
    # Data functions
    "fetch_marketaux_news_snippets",
    "normalize_marketaux_news_data",
    "search_marketaux_entities",
    "get_marketaux_entity_types",
    "get_marketaux_industry_list",
    "get_marketaux_news_sources",
    
    # Text processing
    "TextProcessingPipeline",
    "clean_text",
    "split_text",
    
    # Embedding generation
    "EmbeddingsGenerator",

    # Chroma database manager
    "ChromaDBManager"
]
