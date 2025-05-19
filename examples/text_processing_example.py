"""
Example script demonstrating how to use the TextProcessingPipeline.

This script shows:
1. Fetching articles with EODHDClient
2. Storing them in SQLite
3. Processing the raw content into clean text
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
from financial_news_rag.text_processor import TextProcessingPipeline


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
    
    # Initialize the pipeline
    pipeline = TextProcessingPipeline()
    
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
        pipeline.log_api_call(
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
        print("No articles found. Trying another tag...")
        articles = client.fetch_news(
            tag="EARNINGS RELEASES AND OPERATING RESULTS",
            from_date=week_ago_str,
            to_date=today_str,
            limit=10
        )
        
        # Log the API call for the fallback tag
        if articles:
            oldest_date = min([a.get('published_at', '') for a in articles]) if articles else None
            newest_date = max([a.get('published_at', '') for a in articles]) if articles else None
            
            pipeline.log_api_call(
                query_type='tag',
                query_value='EARNINGS RELEASES AND OPERATING RESULTS',
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
        if pipeline.article_exists(article['url_hash']):
            existing_count += 1
    
    print(f"Found {existing_count} articles that already exist in the database")
    
    # Store articles in the database
    print("Storing articles in SQLite database...")
    stored_count = pipeline.store_articles(articles)
    print(f"Stored {stored_count} articles (duplicates are automatically skipped)")
    
    # Process the articles (clean and normalize text)
    print("Processing articles...")
    processed_count, failed_count = pipeline.process_articles()
    print(f"Processed {processed_count} articles, {failed_count} failed")
    
    # Get chunks for the first article (if any were processed)
    if processed_count > 0 and articles:
        print("\nGetting chunks for the first article:")
        url_hash = articles[0]['url_hash']
        chunks = pipeline.get_chunks_for_article(url_hash)
        
        print(f"Article: {articles[0].get('title', 'Untitled')}")
        print(f"Split into {len(chunks)} chunks")
        
        # Print excerpt from first chunk
        if chunks:
            print("\nExcerpt from first chunk:")
            excerpt = chunks[0]['text'][:250] + "..." if len(chunks[0]['text']) > 250 else chunks[0]['text']
            print(excerpt)
            
            print("\nMetadata:")
            print(f"Published: {chunks[0]['published_at']}")
            print(f"Symbols: {chunks[0]['symbols']}")
            print(f"Tags: {chunks[0]['tags']}")
    
    # Close the database connection
    pipeline.close_connection()
    print("\nFinished processing")


if __name__ == "__main__":
    main()
