"""
Example usage of the refactored EODHDClient with Config class.

This example shows how to use the Config class to instantiate an EODHDClient.
"""

import logging
from financial_news_rag.config import Config
from financial_news_rag.eodhd import EODHDClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the example."""
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = Config()
        
        # Create EODHDClient with configuration
        logger.info("Creating EODHDClient...")
        client = EODHDClient(
            api_key=config.eodhd_api_key,
            api_url=config.eodhd_api_url,
            timeout=config.eodhd_default_timeout,
            max_retries=config.eodhd_default_max_retries,
            backoff_factor=config.eodhd_default_backoff_factor,
            default_limit=config.eodhd_default_limit
        )
        
        # Fetch news by tag
        logger.info("Fetching news for 'earnings' tag...")
        earnings_news = client.fetch_news(tag="earnings", limit=5)
        
        # Print results
        logger.info(f"Fetched {len(earnings_news.get('articles', []))} articles")
        for i, article in enumerate(earnings_news.get("articles", []), 1):
            logger.info(f"Article {i}: {article['title']}")
        
        logger.info("Example completed successfully.")
    
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
        raise

if __name__ == "__main__":
    main()
