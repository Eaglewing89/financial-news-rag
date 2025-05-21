"""
Example script demonstrating how to use the ArticleManager and TextProcessor.

This script shows:
1. Fetching articles with EODHDClient
2. Storing them in SQLite using ArticleManager
3. Processing the raw content into clean text using TextProcessor
4. Generating chunks for embedding
"""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to sys.path if running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
# Import and download nltk data before importing our modules
import nltk
try:
    # Try to find punkt package
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # Download punkt package if not found
    print("Downloading required NLTK data files...")
    nltk.download('punkt', quiet=True)

from financial_news_rag.eodhd import EODHDClient
from financial_news_rag.article_manager import ArticleManager
from financial_news_rag.text_processor import TextProcessor


def main():
    """Run the financial news processing example."""
    # Load environment variables
    load_dotenv()
    
    # Check if API key is available
    api_key = os.getenv('EODHD_API_KEY')
    if not api_key:
        print("Error: EODHD_API_KEY not found in environment variables.")
        print("Please create a .env file with your EODHD_API_KEY.")
        return
    
    # Initialize the article manager and text processor
    article_manager = ArticleManager()
    text_processor = TextProcessor()
    
    # Initialize the EODHD client
    client = EODHDClient()
    
    # Define date range (last 7 days)
    today = datetime.now()
    week_ago = today - timedelta(days=7)
    today_str = today.strftime("%Y-%m-%d")
    week_ago_str = week_ago.strftime("%Y-%m-%d")
    
    # Fetch news articles for a specific tag
    print(f"Fetching news for 'MERGERS AND ACQUISITIONS' from {week_ago_str} to {today_str}...")
    articles = client.fetch_news(
        tag="MERGERS AND ACQUISITIONS",
        from_date=week_ago_str,
        to_date=today_str,
        limit=10  # Increased to demonstrate duplication handling
    )
    
    # Log the API call
    if articles:
        # Find the oldest and newest article dates
        oldest_date = min([a.get('published_at', '') for a in articles]) if articles else None
        newest_date = max([a.get('published_at', '') for a in articles]) if articles else None
        
        # Log the successful API call
        article_manager.log_api_call(
            query_type='tag',
            query_value='MERGERS AND ACQUISITIONS',
            from_date=week_ago_str,
            to_date=today_str,
            limit=10,
            offset=0,
            articles_retrieved_count=len(articles),
            oldest_article_date=oldest_date,
            newest_article_date=newest_date,
            api_call_successful=True,
            http_status_code=200  # Assuming success
        )
    
    if not articles:
        print("No articles found. Try adjusting the date range or topic tag.")
        return
    
    print(f"Fetched {len(articles)} articles")
    
    # Add raw_content to match the expected schema
    for article in articles:
        if 'content' in article:
            article['raw_content'] = article.pop('content')
        else:
            article['raw_content'] = ''
        
        # Add source query tag for tracking
        article['source_query_tag'] = "MERGERS AND ACQUISITIONS"
    
    # Check how many articles already exist in the database
    existing_count = 0
    for article in articles:
        if article_manager.article_exists(article['url_hash']):
            existing_count += 1
    
    print(f"Found {existing_count} articles that already exist in the database")
    
    # Store articles in the database
    print("Storing articles in SQLite database...")
    stored_count = article_manager.store_articles(articles)
    print(f"Stored {stored_count} articles (duplicates are automatically skipped)")
    
    # Process the articles (clean and normalize text)
    print("Processing articles...")
    processed_count = 0
    failed_count = 0
    
    # Get pending articles
    pending_articles = article_manager.get_pending_articles()
    
    for article in pending_articles:
        url_hash = article['url_hash']
        try:
            # Check if raw_content is None or empty
            if not article.get('raw_content'):
                print(f"Empty or missing raw content for article {url_hash}")
                article_manager.update_article_processing_status(
                    url_hash, 
                    status='FAILED', 
                    error_message='Empty or missing raw content'
                )
                failed_count += 1
                continue
            
            # Clean the raw content using TextProcessor
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
            
            # Update the article with processed content using ArticleManager
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
    
    print(f"Processed {processed_count} articles, {failed_count} failed")
    
    # Get chunks for the first article (if any were processed)
    if processed_count > 0 and articles:
        print("\nGetting chunks for the first article:")
        url_hash = articles[0]['url_hash']
        
        # Get the article by hash
        article = article_manager.get_article_by_hash(url_hash)
        
        if article and article['processed_content']:
            # Split processed content into chunks using TextProcessor
            chunks = text_processor.split_into_chunks(article['processed_content'])
            
            print(f"Article: {article.get('title', 'Untitled')}")
            print(f"Split into {len(chunks)} chunks")
            
            # Create chunk objects with metadata for demonstration
            chunk_objects = []
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    'text': chunk_text,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'parent_url_hash': article['url_hash'],
                    'title': article['title'],
                    'url': article['url'],
                    'published_at': article['published_at'],
                    'symbols': article['symbols'],
                    'tags': article['tags'],
                    'sentiment': article.get('sentiment', {})
                }
                chunk_objects.append(chunk)
            
            # Print excerpt from first chunk
            if chunk_objects:
                print("\nExcerpt from first chunk:")
                excerpt = chunk_objects[0]['text'][:250] + "..." if len(chunk_objects[0]['text']) > 250 else chunk_objects[0]['text']
                print(excerpt)
                
                print("\nMetadata:")
                print(f"Published: {chunk_objects[0]['published_at']}")
                print(f"Symbols: {chunk_objects[0]['symbols']}")
                print(f"Tags: {chunk_objects[0]['tags']}")
    
    # Close the database connection
    article_manager.close_connection()
    print("\nFinished processing")


if __name__ == "__main__":
    main()
