"""
Financial News RAG Module.

This package implements a Retrieval Augmented Generation (RAG) system for financial news,
providing functions to fetch, process, and search news articles with semantic understanding.
"""

from financial_news_rag.article_manager import ArticleManager
from financial_news_rag.chroma_manager import ChromaDBManager
from financial_news_rag.config import Config
from financial_news_rag.embeddings import EmbeddingsGenerator
from financial_news_rag.eodhd import EODHDClient, EODHDApiError
from financial_news_rag.text_processor import TextProcessor

__all__ = [
    # EODHD API
    "EODHDClient",
    "EODHDApiError",
    
    # Configuration
    "Config",
    
    # Text processing
    "TextProcessor",
    "ArticleManager",
    
    # Embedding generation
    "EmbeddingsGenerator",

    # Chroma database manager
    "ChromaDBManager"
]
