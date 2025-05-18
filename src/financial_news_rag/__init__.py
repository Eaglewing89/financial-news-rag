"""
Financial News RAG Module.

This package implements a Retrieval Augmented Generation (RAG) system for financial news,
providing functions to fetch, process, and search news articles with semantic understanding.
"""

from financial_news_rag.config import get_config
from financial_news_rag.data import (
    fetch_marketaux_news_snippets,
    normalize_news_data,
    search_entities,
    get_entity_types,
    get_industry_list,
    get_news_sources
)
from financial_news_rag.marketaux import MarketauxNewsFetcher, RateLimiter, fetch_with_retry

__all__ = [
    "MarketauxNewsFetcher", 
    "RateLimiter", 
    "fetch_with_retry",
    "get_config",
    "fetch_marketaux_news_snippets",
    "normalize_news_data",
    "search_entities",
    "get_entity_types",
    "get_industry_list",
    "get_news_sources"
]
