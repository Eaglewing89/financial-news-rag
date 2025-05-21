"""
End-to-End Financial News Pipeline Example

This example demonstrates the complete processing pipeline:
1. Fetching news articles from EODHD API
2. Storing articles in SQLite database with ArticleManager
3. Processing article text with TextProcessor
4. Generating embeddings for processed content
5. Storing embeddings in ChromaDB with metadata links to the SQLite database
"""

import os
import sys
from datetime import datetime, timedelta
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

from financial_news_rag.article_manager import ArticleManager
from financial_news_rag.text_processor import TextProcessor
from financial_news_rag.embeddings import EmbeddingsGenerator
from financial_news_rag.chroma_manager import ChromaDBManager
from financial_news_rag.eodhd import EODHDClient


def main():
    """Run the end-to-end financial news processing pipeline example."""
    load_dotenv()
    
    # Check for required API keys
    if not os.getenv('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your GEMINI_API_KEY.")
        return
    
    if not os.getenv('EODHD_API_KEY'):
        print("Error: EODHD_API_KEY not found in environment variables.")
        print("Please create a .env file with your EODHD_API_KEY.")
        return

    # Initialize components
    article_manager = ArticleManager()  # Uses default 'financial_news.db'
    text_processor = TextProcessor(max_tokens_per_chunk=2048)
    embedder = EmbeddingsGenerator()
    
    # Initialize ChromaDBManager
    chroma_persist_dir = os.path.join(os.getcwd(), 'chroma_db')
    chroma_manager = ChromaDBManager(persist_directory=chroma_persist_dir)
    
    # Print ChromaDB collection status at start
    collection_status = chroma_manager.get_collection_status()
    print("\nChromaDB Collection Status (Before):")
    print(f"- Collection name: {collection_status['collection_name']}")
    print(f"- Total chunks: {collection_status['total_chunks']}")
    print(f"- Unique articles: {collection_status['unique_articles']}")
    print(f"- Persistence location: {collection_status['persist_directory']}")
    print("")

    # STEP 1: Fetch articles from EODHD API
    print("Fetching news articles from EODHD API...")
    eodhd_client = EODHDClient()
    today = datetime.now()
    week_ago = today - timedelta(days=7)
    
    fetched_articles = eodhd_client.fetch_news(
        tag="TECHNOLOGY",  # A common tag
        from_date=week_ago.strftime("%Y-%m-%d"),
        to_date=today.strftime("%Y-%m-%d"),
        limit=5
    )
    
    if not fetched_articles:
        print("No articles fetched by EODHD client. Exiting.")
        article_manager.close_connection()
        return
    
    print(f"Fetched {len(fetched_articles)} articles from EODHD API.")
    
    # Log the API call
    if fetched_articles:
        oldest_date = min([a.get('published_at', '') for a in fetched_articles]) if fetched_articles else None
        newest_date = max([a.get('published_at', '') for a in fetched_articles]) if fetched_articles else None
        
        article_manager.log_api_call(
            query_type='tag',
            query_value='TECHNOLOGY',
            from_date=week_ago.strftime("%Y-%m-%d"),
            to_date=today.strftime("%Y-%m-%d"),
            limit=5,
            offset=0,
            articles_retrieved_count=len(fetched_articles),
            oldest_article_date=oldest_date,
            newest_article_date=newest_date,
            api_call_successful=True,
            http_status_code=200
        )
    
    # STEP 2: Prepare and store articles in SQLite
    print("Preparing articles for storage...")
    for article in fetched_articles:
        # Create raw_content from content and add source query tag
        if 'content' in article:
            article['raw_content'] = article.pop('content', '')
        else:
            article['raw_content'] = ''
        
        article['source_query_tag'] = "TECHNOLOGY"
    
    # Store articles in the database
    stored_count = article_manager.store_articles(fetched_articles)
    print(f"Stored {stored_count} articles in SQLite database.")
    
    # STEP 3: Process raw content to clean text
    print("\nProcessing articles...")
    processed_count = 0
    failed_count = 0
    
    # Get pending articles for processing
    pending_articles = article_manager.get_pending_articles()
    
    for article in pending_articles:
        url_hash = article['url_hash']
        try:
            if not article.get('raw_content'):
                print(f"Empty raw content for article {url_hash}")
                article_manager.update_article_processing_status(
                    url_hash, 
                    status='FAILED', 
                    error_message='Empty raw content'
                )
                failed_count += 1
                continue
            
            # Process the raw content with TextProcessor
            processed_content = text_processor.clean_article_text(article['raw_content'])
            
            if not processed_content:
                print(f"No content after cleaning for article {url_hash}")
                article_manager.update_article_processing_status(
                    url_hash, 
                    status='FAILED', 
                    error_message='No content after cleaning'
                )
                failed_count += 1
                continue
            
            # Update article with processed content
            article_manager.update_article_processing_status(
                url_hash,
                processed_content=processed_content,
                status='SUCCESS'
            )
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing article {url_hash}: {str(e)}")
            article_manager.update_article_processing_status(
                url_hash,
                status='FAILED',
                error_message=str(e)
            )
            failed_count += 1
    
    print(f"Processed {processed_count} articles, {failed_count} failed.")
    
    # STEP 4: Generate embeddings for processed articles
    print("\nFetching processed articles for embedding...")
    articles_for_embedding = article_manager.get_processed_articles_for_embedding(limit=5)
    
    if not articles_for_embedding:
        print("No processed articles found for embedding generation.")
        article_manager.close_connection()
        return
    
    print(f"Found {len(articles_for_embedding)} processed articles ready for embedding.")
    successful_articles = 0
    
    for article in articles_for_embedding:
        print(f"\nGenerating embeddings for article: {article.get('title', 'Untitled')} (url_hash: {article['url_hash']})")
        processed_content = article.get('processed_content')
        
        if not processed_content:
            print(f"Skipping article {article['url_hash']} due to missing processed content.")
            continue
        
        # Split processed content into chunks
        chunks = text_processor.split_into_chunks(processed_content)
        
        if not chunks:
            print(f"No chunks generated for article {article['url_hash']}.")
            continue
        
        print(f"Generated {len(chunks)} chunks for the article.")
        
        # Generate embeddings for these chunks
        print("Generating embeddings...")
        try:
            # Generate embeddings for these chunks
            chunk_embeddings = embedder.generate_embeddings(chunks)
            print(f"Successfully generated {len(chunk_embeddings)} embeddings.")
            
            # Check for zero vectors in the embeddings
            zero_vec = [0.0] * embedder.embedding_dim
            has_zero_vector = any(emb == zero_vec for emb in chunk_embeddings)
            
            if has_zero_vector:
                print(f"One or more chunk embeddings failed for article {article['url_hash']}. Marking as FAILED.")
                article_manager.update_article_embedding_status(
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
                dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                published_at_timestamp = int(dt.timestamp())
            except (ValueError, AttributeError, TypeError):
                published_at_timestamp = None
            
            # Get source query tag if available
            source_query_tag = article.get('source_query_tag')
            
            # Format each chunk with its embedding for ChromaDB
            for i, (chunk_text, embedding_vector) in enumerate(zip(chunks, chunk_embeddings)):
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
            
            # STEP 5: Store embeddings in ChromaDB
            print(f"Storing {len(formatted_chunk_embeddings)} embeddings in ChromaDB...")
            storage_success = chroma_manager.add_embeddings(
                article['url_hash'],
                formatted_chunk_embeddings
            )
            
            if storage_success:
                # Update article status in SQLite
                article_manager.update_article_embedding_status(
                    url_hash=article['url_hash'],
                    status='SUCCESS',
                    embedding_model=embedder.model_name,
                    vector_db_id=article['url_hash']  # Using url_hash as the vector_db_id
                )
                print(f"Successfully stored embeddings for article {article['url_hash']} in ChromaDB.")
                print(f"Updated embedding status for article {article['url_hash']} to SUCCESS.")
                successful_articles += 1
            else:
                article_manager.update_article_embedding_status(
                    url_hash=article['url_hash'],
                    status='FAILED',
                    embedding_model=embedder.model_name,
                    error_message='Failed to store embeddings in ChromaDB'
                )
                print(f"Failed to store embeddings for article {article['url_hash']} in ChromaDB.")
                
        except Exception as e:
            print(f"Error generating embeddings for article {article['url_hash']}: {e}")
            article_manager.update_article_embedding_status(
                url_hash=article['url_hash'],
                status='FAILED',
                error_message=str(e)
            )
    
    # Get final ChromaDB collection status
    final_collection_status = chroma_manager.get_collection_status()
    
    print(f"\n--- End-to-End Pipeline Example Finished ---")
    print(f"Successfully processed and stored embeddings for {successful_articles}/{len(articles_for_embedding)} articles.")
    print("\nFinal ChromaDB Collection Status:")
    print(f"- Total chunks: {final_collection_status['total_chunks']}")
    print(f"- Unique articles: {final_collection_status['unique_articles']}")
    
    # STEP 6: Demonstrate a simple query
    if successful_articles > 0 and formatted_chunk_embeddings:
        print("\nDemonstrating a simple query:")
        # Generate a query embedding from text
        query_text = "Latest technology trends and innovations"
        print(f"Query text: '{query_text}'")
        
        # Get embedding for the query text
        query_embedding = embedder.generate_embeddings([query_text])[0]
        
        # Query the vector database
        results = chroma_manager.query_embeddings(
            query_embedding=query_embedding,
            n_results=3
        )
        
        if results:
            print(f"Found {len(results)} similar chunks:")
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  Chunk ID: {result['chunk_id']}")
                print(f"  Distance: {result['distance']}")
                print(f"  Article URL Hash: {result['metadata']['article_url_hash']}")
                print(f"  Text: {result['text'][:150]}..." if len(result['text']) > 150 else result['text'])
                print()
    
    # Close the database connection
    article_manager.close_connection()
    print("Database connection closed.")


if __name__ == "__main__":
    main()
