"""
Custom assertion functions for the Financial News RAG test suite.

This module provides domain-specific assertion functions that make tests
more readable and provide better error messages.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime, timezone


def assert_article_structure(article: Dict[str, Any], 
                           required_fields: Optional[List[str]] = None) -> None:
    """
    Assert that an article has the expected structure.
    
    Args:
        article: Article dictionary to validate
        required_fields: List of required fields (uses default if None)
    """
    if required_fields is None:
        required_fields = [
            'title', 'raw_content', 'url', 'published_at', 
            'source_api', 'symbols', 'tags'
        ]
    
    assert isinstance(article, dict), "Article must be a dictionary"
    
    for field in required_fields:
        assert field in article, f"Article missing required field: {field}"
        assert article[field] is not None, f"Article field '{field}' cannot be None"


def assert_article_metadata(article: Dict[str, Any],
                          expected_title: str = None,
                          expected_url: str = None,
                          expected_source: str = None) -> None:
    """
    Assert article metadata matches expectations.
    
    Args:
        article: Article dictionary
        expected_title: Expected article title
        expected_url: Expected article URL
        expected_source: Expected source API
    """
    if expected_title:
        assert article['title'] == expected_title, \
            f"Title mismatch: expected '{expected_title}', got '{article['title']}'"
    
    if expected_url:
        assert article['url'] == expected_url, \
            f"URL mismatch: expected '{expected_url}', got '{article['url']}'"
    
    if expected_source:
        assert article['source_api'] == expected_source, \
            f"Source mismatch: expected '{expected_source}', got '{article['source_api']}'"


def assert_article_processing_status(article: Dict[str, Any],
                                    expected_status: str,
                                    check_content: bool = True) -> None:
    """
    Assert article processing status and related fields.
    
    Args:
        article: Article dictionary
        expected_status: Expected processing status
        check_content: Whether to check for processed content
    """
    assert 'processing_status' in article, "Article missing processing_status field"
    assert article['processing_status'] == expected_status, \
        f"Processing status mismatch: expected '{expected_status}', got '{article['processing_status']}'"
    
    if check_content and expected_status == "SUCCESS":
        assert 'processed_content' in article, "Successful processing should have processed_content"
        assert article['processed_content'], "Processed content should not be empty"
    elif expected_status == "FAILED":
        # Failed processing might or might not have processed_content
        pass


def assert_article_embedding_status(article: Dict[str, Any],
                                   expected_status: str,
                                   check_model: bool = True) -> None:
    """
    Assert article embedding status and related fields.
    
    Args:
        article: Article dictionary
        expected_status: Expected embedding status
        check_model: Whether to check for embedding model info
    """
    assert 'embedding_status' in article, "Article missing embedding_status field"
    assert article['embedding_status'] == expected_status, \
        f"Embedding status mismatch: expected '{expected_status}', got '{article['embedding_status']}'"
    
    if check_model and expected_status == "SUCCESS":
        assert 'embedding_model' in article, "Successful embedding should have embedding_model"
        assert article['embedding_model'], "Embedding model should not be empty"


def assert_database_statistics(stats: Dict[str, Any],
                              min_total_articles: int = 0,
                              expected_statuses: List[str] = None) -> None:
    """
    Assert database statistics have expected structure and values.
    
    Args:
        stats: Database statistics dictionary
        min_total_articles: Minimum expected total articles
        expected_statuses: List of statuses that should be present
    """
    required_fields = [
        'total_articles', 'text_processing_status', 'embedding_status',
        'articles_by_tag', 'articles_by_symbol', 'date_range'
    ]
    
    for field in required_fields:
        assert field in stats, f"Statistics missing required field: {field}"
    
    assert isinstance(stats['total_articles'], int), "total_articles must be an integer"
    assert stats['total_articles'] >= min_total_articles, \
        f"Expected at least {min_total_articles} articles, got {stats['total_articles']}"
    
    if expected_statuses:
        for status in expected_statuses:
            assert status in stats['text_processing_status'], \
                f"Expected status '{status}' in text_processing_status"
            assert status in stats['embedding_status'], \
                f"Expected status '{status}' in embedding_status"


def assert_chroma_collection_status(status: Dict[str, Any],
                                   min_chunks: int = 0,
                                   min_articles: int = 0) -> None:
    """
    Assert ChromaDB collection status has expected structure and values.
    
    Args:
        status: Collection status dictionary
        min_chunks: Minimum expected chunks
        min_articles: Minimum expected unique articles
    """
    required_fields = ['total_chunks', 'unique_articles', 'is_empty', 'collection_name']
    
    for field in required_fields:
        assert field in status, f"Collection status missing required field: {field}"
    
    assert isinstance(status['total_chunks'], int), "total_chunks must be an integer"
    assert isinstance(status['unique_articles'], int), "unique_articles must be an integer"
    assert isinstance(status['is_empty'], bool), "is_empty must be a boolean"
    
    assert status['total_chunks'] >= min_chunks, \
        f"Expected at least {min_chunks} chunks, got {status['total_chunks']}"
    assert status['unique_articles'] >= min_articles, \
        f"Expected at least {min_articles} articles, got {status['unique_articles']}"
    
    # Consistency check
    if status['total_chunks'] == 0:
        assert status['is_empty'] is True, "Collection with 0 chunks should be empty"
        assert status['unique_articles'] == 0, "Empty collection should have 0 unique articles"


def assert_chroma_query_results(results: List[Dict[str, Any]],
                               expected_count: int = None,
                               min_count: int = 0,
                               max_distance: float = 1.0) -> None:
    """
    Assert ChromaDB query results have expected structure and properties.
    
    Args:
        results: List of query result dictionaries
        expected_count: Exact expected number of results
        min_count: Minimum expected number of results
        max_distance: Maximum allowed distance value
    """
    assert isinstance(results, list), "Results must be a list"
    
    if expected_count is not None:
        assert len(results) == expected_count, \
            f"Expected exactly {expected_count} results, got {len(results)}"
    else:
        assert len(results) >= min_count, \
            f"Expected at least {min_count} results, got {len(results)}"
    
    required_fields = ['chunk_id', 'text', 'metadata']
    
    for i, result in enumerate(results):
        assert isinstance(result, dict), f"Result {i} must be a dictionary"
        
        for field in required_fields:
            assert field in result, f"Result {i} missing required field: {field}"
        
        # Check metadata structure
        metadata = result['metadata']
        assert 'article_url_hash' in metadata, f"Result {i} metadata missing article_url_hash"
        assert 'chunk_index' in metadata, f"Result {i} metadata missing chunk_index"
        
        # Check distance if present
        if 'distance' in result:
            distance = result['distance']
            assert isinstance(distance, (int, float)), f"Result {i} distance must be numeric"
            assert 0 <= distance <= max_distance, \
                f"Result {i} distance {distance} outside valid range [0, {max_distance}]"


def assert_embedding_vector(embedding: List[float],
                           expected_dimension: int = 768,
                           check_normalized: bool = False) -> None:
    """
    Assert embedding vector has expected properties.
    
    Args:
        embedding: Embedding vector
        expected_dimension: Expected embedding dimension
        check_normalized: Whether to check if vector is normalized
    """
    assert isinstance(embedding, list), "Embedding must be a list"
    assert len(embedding) == expected_dimension, \
        f"Expected embedding dimension {expected_dimension}, got {len(embedding)}"
    
    # Check all values are numeric
    for i, value in enumerate(embedding):
        assert isinstance(value, (int, float)), \
            f"Embedding value at index {i} must be numeric, got {type(value)}"
        assert not np.isnan(value), f"Embedding value at index {i} is NaN"
        assert not np.isinf(value), f"Embedding value at index {i} is infinite"
    
    if check_normalized:
        # Check if vector is approximately normalized (L2 norm ≈ 1)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01, f"Vector not normalized: L2 norm = {norm}"


def assert_text_chunks(chunks: List[str],
                      min_chunks: int = 1,
                      max_chunks: int = None,
                      min_chunk_length: int = 10) -> None:
    """
    Assert text chunks have expected properties.
    
    Args:
        chunks: List of text chunks
        min_chunks: Minimum expected number of chunks
        max_chunks: Maximum expected number of chunks
        min_chunk_length: Minimum expected chunk length
    """
    assert isinstance(chunks, list), "Chunks must be a list"
    assert len(chunks) >= min_chunks, \
        f"Expected at least {min_chunks} chunks, got {len(chunks)}"
    
    if max_chunks is not None:
        assert len(chunks) <= max_chunks, \
            f"Expected at most {max_chunks} chunks, got {len(chunks)}"
    
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, str), f"Chunk {i} must be a string"
        assert len(chunk) >= min_chunk_length, \
            f"Chunk {i} too short: {len(chunk)} < {min_chunk_length}"
        assert chunk.strip(), f"Chunk {i} is empty or only whitespace"


def assert_api_response(response: Dict[str, Any],
                       expected_success: bool = True,
                       expected_status_code: int = None,
                       min_articles: int = 0) -> None:
    """
    Assert API response has expected structure and status.
    
    Args:
        response: API response dictionary
        expected_success: Expected success status
        expected_status_code: Expected HTTP status code
        min_articles: Minimum expected number of articles
    """
    required_fields = ['articles', 'status_code', 'success']
    
    for field in required_fields:
        assert field in response, f"API response missing required field: {field}"
    
    assert isinstance(response['articles'], list), "Articles must be a list"
    assert isinstance(response['success'], bool), "Success must be a boolean"
    assert isinstance(response['status_code'], int), "Status code must be an integer"
    
    assert response['success'] == expected_success, \
        f"Expected success={expected_success}, got {response['success']}"
    
    if expected_status_code is not None:
        assert response['status_code'] == expected_status_code, \
            f"Expected status code {expected_status_code}, got {response['status_code']}"
    
    if expected_success:
        assert len(response['articles']) >= min_articles, \
            f"Expected at least {min_articles} articles, got {len(response['articles'])}"
    
    # If success is False, check for error message
    if not expected_success:
        assert 'message' in response, "Error response should have a message field"


def assert_orchestrator_result(result: Dict[str, Any],
                              expected_status: str,
                              expected_fields: List[str] = None) -> None:
    """
    Assert orchestrator operation result has expected structure.
    
    Args:
        result: Operation result dictionary
        expected_status: Expected status value
        expected_fields: List of expected fields in result
    """
    assert isinstance(result, dict), "Result must be a dictionary"
    assert 'status' in result, "Result missing status field"
    assert result['status'] == expected_status, \
        f"Expected status '{expected_status}', got '{result['status']}'"
    
    if expected_fields:
        for field in expected_fields:
            assert field in result, f"Result missing expected field: {field}"


def assert_date_range_valid(date_range: Dict[str, str]) -> None:
    """
    Assert date range has valid structure and dates.
    
    Args:
        date_range: Date range dictionary with 'earliest_article' and 'latest_article'
    """
    required_fields = ['earliest_article', 'latest_article']
    
    for field in required_fields:
        assert field in date_range, f"Date range missing required field: {field}"
    
    # Parse dates to ensure they're valid ISO format
    try:
        earliest = datetime.fromisoformat(date_range['earliest_article'].replace('Z', '+00:00'))
        latest = datetime.fromisoformat(date_range['latest_article'].replace('Z', '+00:00'))
    except ValueError as e:
        raise AssertionError(f"Invalid date format in date range: {e}")
    
    # Logical consistency check
    assert earliest <= latest, \
        f"Earliest date {earliest} should be <= latest date {latest}"


def assert_search_results(results: List[Dict[str, Any]],
                         query: str,
                         min_results: int = 0,
                         max_results: int = None,
                         check_relevance: bool = False) -> None:
    """
    Assert search results have expected properties.
    
    Args:
        results: List of search result dictionaries
        query: Original search query
        min_results: Minimum expected results
        max_results: Maximum expected results
        check_relevance: Whether to perform basic relevance checks
    """
    assert isinstance(results, list), "Results must be a list"
    assert len(results) >= min_results, \
        f"Expected at least {min_results} results, got {len(results)}"
    
    if max_results is not None:
        assert len(results) <= max_results, \
            f"Expected at most {max_results} results, got {len(results)}"
    
    required_fields = ['title', 'content', 'url', 'score']
    
    for i, result in enumerate(results):
        assert isinstance(result, dict), f"Result {i} must be a dictionary"
        
        for field in required_fields:
            assert field in result, f"Result {i} missing required field: {field}"
        
        # Check score is numeric and reasonable
        score = result['score']
        assert isinstance(score, (int, float)), f"Result {i} score must be numeric"
        assert 0 <= score <= 1, f"Result {i} score {score} outside valid range [0, 1]"
        
        if check_relevance:
            # Basic relevance check - result should contain at least one query term
            content_lower = result['content'].lower()
            title_lower = result['title'].lower()
            query_terms = query.lower().split()
            
            has_relevance = any(
                term in content_lower or term in title_lower
                for term in query_terms
            )
            assert has_relevance, f"Result {i} appears irrelevant to query '{query}'"


def assert_search_result_structure(result: Dict[str, Any],
                                  has_rerank_score: bool = None) -> None:
    """
    Assert search result has the expected structure from orchestrator.search_articles().
    
    Args:
        result: Search result dictionary from orchestrator
        has_rerank_score: Whether to expect rerank_score field (auto-detected if None)
    """
    assert isinstance(result, dict), "Search result must be a dictionary"
    
    # Required fields that should always be present in search results
    required_fields = [
        'url_hash', 'title', 'processed_content', 'url', 'published_at',
        'source_api', 'symbols', 'tags', 'similarity_score'
    ]
    
    for field in required_fields:
        assert field in result, f"Search result missing required field: {field}"
        assert result[field] is not None, f"Search result field '{field}' cannot be None"
    
    # Validate similarity_score
    similarity_score = result['similarity_score']
    assert isinstance(similarity_score, (int, float)), "similarity_score must be numeric"
    assert 0 <= similarity_score <= 1, f"similarity_score {similarity_score} outside valid range [0, 1]"
    
    # Check for rerank_score if specified or auto-detect
    if has_rerank_score is None:
        has_rerank_score = 'rerank_score' in result
    
    if has_rerank_score:
        assert 'rerank_score' in result, "Expected rerank_score field in reranked result"
        rerank_score = result['rerank_score']
        assert isinstance(rerank_score, (int, float)), "rerank_score must be numeric"
        assert 0 <= rerank_score <= 1, f"rerank_score {rerank_score} outside valid range [0, 1]"
    
    # Validate article content fields
    assert isinstance(result['title'], str), "title must be a string"
    assert len(result['title'].strip()) > 0, "title cannot be empty"
    
    assert isinstance(result['processed_content'], str), "processed_content must be a string"
    assert len(result['processed_content'].strip()) > 0, "processed_content cannot be empty"
    
    assert isinstance(result['url'], str), "url must be a string"
    assert result['url'].startswith(('http://', 'https://')), "url must be a valid HTTP/HTTPS URL"
    
    # Validate lists
    assert isinstance(result['symbols'], list), "symbols must be a list"
    assert isinstance(result['tags'], list), "tags must be a list"
    
    # Validate published_at is a valid datetime string
    try:
        datetime.fromisoformat(result['published_at'].replace('Z', '+00:00'))
    except ValueError:
        raise AssertionError(f"published_at '{result['published_at']}' is not a valid ISO datetime string")
