"""
ReRanker for Financial News RAG.

This module provides a class for re-ranking a list of articles based on their relevance to a query
using the Gemini LLM.
"""

import logging
import json
from typing import List, Dict, Any, Union
import re

from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPIError, ServiceUnavailable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReRanker:
    """
    A class for re-ranking articles based on relevance to a query using the Gemini LLM.
    
    This class is responsible for:
    - Initializing a Gemini API client
    - Re-ranking articles based on their content relevance to a query
    - Handling API errors gracefully with retry logic
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the ReRanker with API key and model name.
        
        Args:
            api_key: The Gemini API key.
            model_name: The Gemini model to use. Defaults to "gemini-2.0-flash".
        
        Raises:
            ValueError: If the API key is not provided.
        """
        if not api_key:
            raise ValueError("Gemini API key is required.")
            
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        logger.info(f"ReRanker initialized with model: {self.model_name}")
    
    @retry(
        retry=retry_if_exception_type((GoogleAPIError, ServiceUnavailable, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def _assess_article_relevance(self, query: str, article_content: str, url_hash: str) -> Dict[str, Any]:
        """
        Assess the relevance of an article to a query using Gemini 2.0 Flash.
        
        Args:
            query: The original user query
            article_content: The processed content of the article
            url_hash: The unique identifier for the article
            
        Returns:
            A dictionary with the article url_hash and its relevance score
            
        Raises:
            GoogleAPIError: If there's an issue with the API call
            ValueError: If the relevance assessment fails
        """
        # Skip empty content
        if not article_content or not article_content.strip():
            logger.warning(f"Empty content for article {url_hash}")
            return {"id": url_hash, "score": 0.0}
        
        # Use a truncated version if the content is too long
        content_preview = article_content[:10000] + "..." if len(article_content) > 10000 else article_content
        
        # Create the prompt for relevance assessment
        system_instruction = """
        You are a financial analyst assistant. Your task is to evaluate the relevance 
        of financial news articles to a user's query. Rate each article on a scale of 0-10, 
        where 10 means the article is perfectly relevant to answering the query, and 0 means 
        it's completely irrelevant. Consider specificity, recency, and depth of information.
        
        Return a JSON object with the article's ID and its relevance score:
        {"id": "article-id", "score": 8.5}
        
        Only return the JSON object, nothing else.
        """
        
        input_contents = f"""
        USER QUERY: {query}
        
        ARTICLE TO EVALUATE:
        Article ID: {url_hash}
        Content: {content_preview}
        
        Please rate the relevance of this article to the query.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=input_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=256,
                    temperature=0.1
                )
            )
            
            # Parse the response to extract the score
            response_text = response.text.strip()
            
            # Try to parse the JSON response
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the score using regex
                score_match = re.search(r'"score":\s*(\d+(\.\d+)?)', response_text)
                if score_match:
                    score = float(score_match.group(1))
                    return {"id": url_hash, "score": score}
                else:
                    logger.error(f"Failed to parse relevance score from response: {response_text}")
                    return {"id": url_hash, "score": 0.0}
                
        except Exception as e:
            logger.error(f"Error assessing article relevance: {e}")
            return {"id": url_hash, "score": 0.0}
    
    def rerank_articles(self, query: str, articles: List[Dict]) -> List[Dict]:
        """
        Re-rank a list of articles based on their relevance to a query.
        
        Args:
            query: The original user query
            articles: A list of dictionaries, each representing an article and containing:
                - url_hash: The unique identifier for the article
                - processed_content: The text content of the article
                - Other fields from the original retrieval
                
        Returns:
            A list of article dictionaries, sorted by relevance score in descending order,
            with an additional 'rerank_score' field
        """
        if not articles:
            logger.warning("No articles provided for re-ranking.")
            return []
        
        if not query or not query.strip():
            logger.warning("Empty query provided for re-ranking.")
            return articles
        
        # Create a copy of the articles list to avoid modifying the original
        reranked_articles = articles.copy()
        
        try:
            # Assess the relevance of each article
            for article in reranked_articles:
                if 'url_hash' not in article or 'processed_content' not in article:
                    logger.warning(f"Article missing required fields: {article.get('url_hash', 'unknown')}")
                    article['rerank_score'] = 0.0
                    continue
                
                # Get relevance score from the Gemini model
                relevance_result = self._assess_article_relevance(
                    query=query,
                    article_content=article['processed_content'],
                    url_hash=article['url_hash']
                )
                
                # Store the relevance score
                article['rerank_score'] = relevance_result.get('score', 0.0)
            
            # Sort the articles by relevance score in descending order
            reranked_articles.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            
            return reranked_articles
            
        except Exception as e:
            logger.error(f"Error in rerank_articles: {e}")
            logger.error("Returning original article list.")
            # If any error occurs, return the original list unmodified
            return articles
