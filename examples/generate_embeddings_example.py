"""
Example script demonstrating the embedding generation workflow.

This example shows how the EmbeddingsGenerator fits into the pipeline,
from processing text to generating embeddings for article chunks.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import NLTK and download 'punkt' if not present
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK data files ('punkt')...")
    nltk.download('punkt', quiet=True)

from financial_news_rag.text_processor import TextProcessingPipeline
from financial_news_rag.embeddings import EmbeddingsGenerator
from financial_news_rag.eodhd import EODHDClient
from datetime import datetime, timedelta


def main():
    load_dotenv()
    
    # Ensure GEMINI_API_KEY is set
    if not os.getenv('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your GEMINI_API_KEY.")
        return

    # Initialize pipeline and embedder
    pipeline = TextProcessingPipeline()  # Uses default 'financial_news.db'
    embedder = EmbeddingsGenerator()

    print("Fetching processed articles that need embedding...")
    
    # --- Setup for example data (mimicking an orchestrator's first steps) ---
    # This part ensures we have data in the 'processed_content' field
    from financial_news_rag.eodhd import EODHDClient
    from datetime import datetime, timedelta
    
    if not os.getenv('EODHD_API_KEY'):
        print("EODHD_API_KEY needed for example data setup. Skipping example data generation.")
        pipeline.close_connection()
        return
        
    eodhd_client = EODHDClient()
    today = datetime.now()
    week_ago = today - timedelta(days=30)  # Wider range to ensure some news
    fetched_articles = eodhd_client.fetch_news(
        tag="TECHNOLOGY",  # A common tag
        from_date=week_ago.strftime("%Y-%m-%d"),
        to_date=today.strftime("%Y-%m-%d"),
        limit=5
    )

    if not fetched_articles:
        print("No articles fetched by EODHD client for example setup. Exiting.")
        pipeline.close_connection()
        return

    print(f"Fetched {len(fetched_articles)} raw articles for setup.")
    for article in fetched_articles:
        article['raw_content'] = article.pop('content', '')
        article['source_query_tag'] = "TECHNOLOGY"
    
    pipeline.store_articles(fetched_articles)
    processed_count, _ = pipeline.process_articles()
    print(f"Processed {processed_count} articles for setup.")
    # --- End of setup for example data ---

    # Now, fetch articles that have 'processed_content' and are ready for embedding
    articles_data = pipeline.get_processed_articles_for_embedding(limit=5)

    if not articles_data:
        print("No processed articles found to generate embeddings for.")
        pipeline.close_connection()
        return

    print(f"Found {len(articles_data)} processed articles to generate embeddings for.")

    all_article_embeddings = []

    for article in articles_data:
        print(f"\nProcessing article: {article.get('title', 'Untitled')} (url_hash: {article['url_hash']})")
        processed_content = article.get('processed_content')
        
        if not processed_content:
            print(f"Skipping article {article['url_hash']} due to missing processed_content.")
            continue

        # Get chunks for the article's processed_content
        chunks_text_only = pipeline.split_into_chunks(processed_content)

        if not chunks_text_only:
            print(f"No chunks generated for article {article['url_hash']}.")
            continue
        
        print(f"Generated {len(chunks_text_only)} chunks for the article.")

        # Generate embeddings for these chunks
        print("Generating embeddings...")
        try:
            chunk_embeddings = embedder.generate_embeddings(chunks_text_only)
            print(f"Successfully generated {len(chunk_embeddings)} embeddings for the chunks.")

            # Prepare article metadata (excluding processed_content)
            article_metadata = {k: v for k, v in article.items() if k != 'processed_content'}
            
            all_article_embeddings.append({
                "article_metadata": article_metadata,
                "chunk_embeddings": chunk_embeddings,
                "chunks_text": chunks_text_only  # For reference
            })
            
            # Update status in the database to indicate successful embedding
            pipeline.update_article_embedding_status(
                url_hash=article['url_hash'],
                status='SUCCESS',
                embedding_model='text-embedding-004'
            )
            print(f"Updated embedding status for article {article['url_hash']} to SUCCESS.")
            
            # In a real pipeline, here you would call vector_db_manager.store_embeddings(...)
            
        except Exception as e:
            print(f"Error generating embeddings for article {article['url_hash']}: {e}")
            
            # Update status to indicate failure
            pipeline.update_article_embedding_status(
                url_hash=article['url_hash'],
                status='FAILED',
                error_message=str(e)
            )

    print(f"\n--- Example Finished ---")
    if all_article_embeddings:
        print(f"Generated embeddings for {len(all_article_embeddings)} articles.")
        print("Sample of generated data for the first article:")
        print(f"  Title: {all_article_embeddings[0]['article_metadata'].get('title')}")
        print(f"  Number of chunk embeddings: {len(all_article_embeddings[0]['chunk_embeddings'])}")
        if all_article_embeddings[0]['chunk_embeddings']:
            print(f"  Dimension of first chunk embedding: {len(all_article_embeddings[0]['chunk_embeddings'][0])}")

    # Close the pipeline's database connection
    pipeline.close_connection()
    print("Database connection closed.")

if __name__ == "__main__":
    main()
