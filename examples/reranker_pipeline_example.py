"""
ReRanker Pipeline Example

This example demonstrates the use of the ReRanker class in a pipeline:
1. Initialize pipeline components (EmbeddingsGenerator, ChromaDBManager, ArticleManager, ReRanker)
2. Define a sample financial news query
3. Generate embeddings for the query
4. Retrieve relevant articles from ChromaDB
5. Fetch full article content from SQLite
6. Re-rank articles using the ReRanker
7. Compare and display original and re-ranked results
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_news_rag.article_manager import ArticleManager
from financial_news_rag.embeddings import EmbeddingsGenerator
from financial_news_rag.chroma_manager import ChromaDBManager
from financial_news_rag.reranker import ReRanker


def main():
    """Run the ReRanker pipeline example."""
    load_dotenv()
    
    # Check for required API keys
    if not os.getenv('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your GEMINI_API_KEY.")
        return
    
    print("Initializing pipeline components...")
    # Initialize components
    article_manager = ArticleManager()  # Uses default 'financial_news.db'
    embedder = EmbeddingsGenerator()
    
    # Initialize ChromaDBManager
    chroma_persist_dir = os.path.join(os.getcwd(), 'chroma_db')
    chroma_manager = ChromaDBManager(persist_directory=chroma_persist_dir)
    
    # Initialize ReRanker
    reranker = ReRanker()
    
    # Print ChromaDB collection status
    collection_status = chroma_manager.get_collection_status()
    print("\nChromaDB Collection Status:")
    print(f"- Collection name: {collection_status['collection_name']}")
    print(f"- Total chunks: {collection_status['total_chunks']}")
    print(f"- Unique articles: {collection_status['unique_articles']}")
    
    if collection_status['total_chunks'] == 0:
        print("\nNo articles found in the ChromaDB collection.")
        print("Please run the end_to_end_pipeline_example.py first to populate the database.")
        return
    
    # 1. Define a sample query
    query = "Artificial intelligence innovations in finance"
    print(f"\nSample query: '{query}'")
    
    # 2. Generate embeddings for the query
    print("Generating embeddings for the query...")
    query_embedding = embedder.generate_embeddings([query])[0]
    
    # 3. Initial retrieval from ChromaDB
    print("Retrieving relevant articles from ChromaDB...")
    n_results = 5  # Number of results to retrieve
    
    # Query the vector database
    chroma_results = chroma_manager.query_embeddings(
        query_embedding=query_embedding,
        n_results=n_results
    )
    
    if not chroma_results:
        print("No relevant articles found in ChromaDB.")
        article_manager.close_connection()
        return
    
    print(f"Retrieved {len(chroma_results)} relevant chunks from ChromaDB.")
    
    # 4. Extract unique article hashes from the results
    unique_article_hashes = set()
    for result in chroma_results:
        article_hash = result['metadata'].get('article_url_hash')
        if article_hash:
            unique_article_hashes.add(article_hash)
    
    # 5. Fetch full article content for the retrieved articles
    print(f"Fetching full content for {len(unique_article_hashes)} unique articles...")
    articles = []
    
    for url_hash in unique_article_hashes:
        article = article_manager.get_article_by_hash(url_hash)
        if article and article.get('processed_content'):
            articles.append(article)
    
    if not articles:
        print("Could not retrieve any article content from the database.")
        article_manager.close_connection()
        return
    
    print(f"Successfully fetched content for {len(articles)} articles.")
    
    # 6. Perform re-ranking using ReRanker
    print("\nPerforming relevance re-ranking with Gemini 2.0 Flash...")
    reranked_articles = reranker.rerank_articles(query, articles)
    
    # 7. Display the results
    print("\n--- ORIGINAL RANKING (BY EMBEDDING SIMILARITY) ---")
    for i, article in enumerate(articles):
        print(f"{i+1}. {article.get('title', 'Untitled')}")
        print(f"   URL Hash: {article.get('url_hash')}")
        content_preview = article.get('processed_content', '')[:150] + "..." if article.get('processed_content') else 'No content'
        print(f"   Content preview: {content_preview}")
        print()
    
    print("\n--- RE-RANKED RESULTS (BY GEMINI 2.0 FLASH) ---")
    for i, article in enumerate(reranked_articles):
        print(f"{i+1}. {article.get('title', 'Untitled')} (Score: {article.get('rerank_score', 0.0):.2f})")
        print(f"   URL Hash: {article.get('url_hash')}")
        content_preview = article.get('processed_content', '')[:150] + "..." if article.get('processed_content') else 'No content'
        print(f"   Content preview: {content_preview}")
        print()
    
    # 8. Compare rankings
    print("\n--- RANKING COMPARISON ---")
    original_order = {articles[i]['url_hash']: i+1 for i in range(len(articles))}
    reranked_order = {reranked_articles[i]['url_hash']: i+1 for i in range(len(reranked_articles))}
    
    print("URL Hash | Original Rank | Re-ranked Rank | Change")
    print("---------|--------------|----------------|-------")
    
    for url_hash in original_order:
        original_rank = original_order[url_hash]
        reranked_rank = reranked_order.get(url_hash, "N/A")
        
        if reranked_rank != "N/A":
            change = original_rank - reranked_rank
            change_str = f"+{change}" if change > 0 else str(change)
        else:
            change_str = "N/A"
        
        print(f"{url_hash[:8]}... | {original_rank} | {reranked_rank} | {change_str}")
    
    # Close the database connection
    article_manager.close_connection()
    print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
