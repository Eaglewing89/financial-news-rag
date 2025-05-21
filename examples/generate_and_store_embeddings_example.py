"""
Example script demonstrating the embedding generation and storage workflow.

This example shows the end-to-end process:
1. Retrieving processed articles from SQLite
2. Generating embeddings for each article's chunks
3. Storing embeddings in ChromaDB with links to SQLite database
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
from financial_news_rag.chroma_manager import ChromaDBManager
# from financial_news_rag.eodhd import EODHDClient
# from datetime import datetime, timedelta


def main():
    load_dotenv()
    
    # Ensure GEMINI_API_KEY is set
    if not os.getenv('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your GEMINI_API_KEY.")
        return

    # Initialize pipeline, embedder, and ChromaDB manager
    pipeline = TextProcessingPipeline()  # Uses default 'financial_news.db'
    embedder = EmbeddingsGenerator()
    
    # Initialize ChromaDBManager
    # Use a directory relative to current directory for ChromaDB persistence
    chroma_persist_dir = os.path.join(os.getcwd(), 'chroma_db')
    chroma_manager = ChromaDBManager(persist_directory=chroma_persist_dir)
    
    # Print ChromaDB collection status
    collection_status = chroma_manager.get_collection_status()
    print("\nChromaDB Collection Status:")
    print(f"- Collection name: {collection_status['collection_name']}")
    print(f"- Total chunks: {collection_status['total_chunks']}")
    print(f"- Unique articles: {collection_status['unique_articles']}")
    print(f"- Persistence location: {collection_status['persist_directory']}")
    print("")

    print("Fetching processed articles that need embedding...")
    
    # --- Setup for example data (mimicking an orchestrator's first steps) ---
    # This part is commented out to avoid fetching new data on every run.
    # Uncomment if you need fresh data.
    #
    # if not os.getenv('EODHD_API_KEY'):
    #     print("EODHD_API_KEY needed for example data setup. Skipping example data generation.")
    #     pipeline.close_connection()
    #     return
    #     
    # eodhd_client = EODHDClient()
    # today = datetime.now()
    # week_ago = today - timedelta(days=30)  # Wider range to ensure some news
    # fetched_articles = eodhd_client.fetch_news(
    #     tag="TECHNOLOGY",  # A common tag
    #     from_date=week_ago.strftime("%Y-%m-%d"),
    #     to_date=today.strftime("%Y-%m-%d"),
    #     limit=5
    # )
    #
    # if not fetched_articles:
    #     print("No articles fetched by EODHD client for example setup. Exiting.")
    #     pipeline.close_connection()
    #     return
    # 
    # print(f"Fetched {len(fetched_articles)} raw articles for setup.")
    # for article in fetched_articles:
    #     article['raw_content'] = article.pop('content', '')
    #     article['source_query_tag'] = "TECHNOLOGY"
    # 
    # pipeline.store_articles(fetched_articles)
    # processed_count, _ = pipeline.process_articles()
    # print(f"Processed {processed_count} articles for setup.")
    # --- End of setup for example data ---

    # Now, fetch articles that have 'processed_content' and are ready for embedding
    articles_data = pipeline.get_processed_articles_for_embedding(limit=5)

    if not articles_data:
        print("No processed articles found to generate embeddings for.")
        pipeline.close_connection()
        return

    print(f"Found {len(articles_data)} processed articles to generate embeddings for.")

    successful_articles = 0

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
            chunk_embeddings_vectors = embedder.generate_embeddings(chunks_text_only)
            print(f"Successfully generated {len(chunk_embeddings_vectors)} embeddings for the chunks.")

            # Check for zero vectors in the embeddings
            zero_vec = [0.0] * embedder.embedding_dim
            has_zero_vector = any(
                emb == zero_vec for emb in chunk_embeddings_vectors
            )

            if has_zero_vector:
                print(f"One or more chunk embeddings failed for article {article['url_hash']}. Marking as FAILED.")
                pipeline.update_article_embedding_status(
                    url_hash=article['url_hash'],
                    status='FAILED',
                    embedding_model=embedder.model_name,
                    error_message='One or more chunk embeddings failed (zero vector)'
                )
                continue

            # Prepare chunks with embeddings for ChromaDB storage
            formatted_chunk_embeddings = []
            
            # Get article published_at and convert to timestamp if available
            published_at = article.get('published_at')
            try:
                # Try to convert ISO format date to UNIX timestamp
                from datetime import datetime
                dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                published_at_timestamp = int(dt.timestamp())
            except (ValueError, AttributeError, TypeError):
                published_at_timestamp = None
                
            # Get source query tag if available
            source_query_tag = article.get('source_query_tag')
            
            # Format each chunk with its embedding for ChromaDB
            for i, (chunk_text, embedding_vector) in enumerate(zip(chunks_text_only, chunk_embeddings_vectors)):
                chunk_id = f"{article['url_hash']}_{i}"
                
                # Prepare metadata
                metadata = {
                    "article_url_hash": article['url_hash'],
                    "chunk_index": i
                }
                
                # Add optional metadata if available
                if published_at_timestamp:
                    metadata["published_at_timestamp"] = published_at_timestamp
                if source_query_tag:
                    metadata["source_query_tag"] = source_query_tag
                
                formatted_chunk_embeddings.append({
                    "chunk_id": chunk_id,
                    "embedding": embedding_vector,
                    "text": chunk_text,
                    "metadata": metadata
                })
            
            # Store embeddings in ChromaDB
            print(f"Storing {len(formatted_chunk_embeddings)} embeddings in ChromaDB...")
            storage_success = chroma_manager.add_embeddings(
                article['url_hash'],
                formatted_chunk_embeddings
            )
            
            if storage_success:
                # Update article status in SQLite
                pipeline.update_article_embedding_status(
                    url_hash=article['url_hash'],
                    status='SUCCESS',
                    embedding_model=embedder.model_name,
                    vector_db_id=article['url_hash']  # Using url_hash as the vector_db_id
                )
                print(f"Successfully stored embeddings for article {article['url_hash']} in ChromaDB.")
                print(f"Updated embedding status for article {article['url_hash']} to SUCCESS.")
                successful_articles += 1
            else:
                pipeline.update_article_embedding_status(
                    url_hash=article['url_hash'],
                    status='FAILED',
                    embedding_model=embedder.model_name,
                    error_message='Failed to store embeddings in ChromaDB'
                )
                print(f"Failed to store embeddings for article {article['url_hash']} in ChromaDB.")

        except Exception as e:
            print(f"Error processing article {article['url_hash']}: {e}")
            # Update status to indicate failure
            pipeline.update_article_embedding_status(
                url_hash=article['url_hash'],
                status='FAILED',
                error_message=str(e)
            )

    # Get final ChromaDB collection status
    final_collection_status = chroma_manager.get_collection_status()
    
    print(f"\n--- Example Finished ---")
    print(f"Successfully processed and stored embeddings for {successful_articles}/{len(articles_data)} articles.")
    print("\nFinal ChromaDB Collection Status:")
    print(f"- Total chunks: {final_collection_status['total_chunks']}")
    print(f"- Unique articles: {final_collection_status['unique_articles']}")
    
    # Perform a simple query to demonstrate retrieval
    if successful_articles > 0 and formatted_chunk_embeddings:
        print("\nDemonstrating a simple query:")
        # Use the last embedding we generated as a query
        sample_query_embedding = formatted_chunk_embeddings[-1]["embedding"]
        
        results = chroma_manager.query_embeddings(
            query_embedding=sample_query_embedding,
            n_results=2
        )
        
        if results:
            print(f"Found {len(results)} similar chunks:")
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  Chunk ID: {result['chunk_id']}")
                print(f"  Distance: {result['distance']}")
                print(f"  Article URL Hash: {result['metadata']['article_url_hash']}")
                print(f"  Text: {result['text'][:100]}..." if len(result['text']) > 100 else result['text'])
                print()
    
    # Close the pipeline's database connection
    pipeline.close_connection()
    print("Database connection closed.")

if __name__ == "__main__":
    main()
