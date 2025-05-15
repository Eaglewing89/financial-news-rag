# Model Specifications for Financial News RAG

This document provides detailed specifications for the AI models used in the Financial News RAG module.

## Gemini Models Overview

A token is equivalent to about 4 characters for Gemini models. 100 tokens are about 60-80 English words.

## Gemini Embedding Model: text-embedding-004

This is the primary embedding model used for vectorizing financial news articles.

### Model Details

| Property | Description |
|----------|-------------|
| Model code | `models/text-embedding-004` |
| Supported data types (Input) | Text |
| Supported data types (Output) | Text embeddings |
| Input token limit | 2,048 |
| Output dimension size | 768 |
| Rate limits | 1,500 requests per minute |
| Adjustable safety settings | Not supported |
| Latest update | April 2024 |

### Supported Task Types

- **SEMANTIC_SIMILARITY**: Used to generate embeddings that are optimized to assess text similarity.
- **CLASSIFICATION**: Used to generate embeddings that are optimized to classify texts according to preset labels.
- **CLUSTERING**: Used to generate embeddings that are optimized to cluster texts based on their similarities.
- **RETRIEVAL_DOCUMENT**, **RETRIEVAL_QUERY**, **QUESTION_ANSWERING**, and **FACT_VERIFICATION**: Used to generate embeddings that are optimized for document search or information retrieval.
- **CODE_RETRIEVAL_QUERY**: Used to retrieve a code block based on a natural language query, such as sort an array or reverse a linked list. Embeddings of the code blocks are computed using RETRIEVAL_DOCUMENT.

### Chunking Strategy

Since text-embedding-004 can handle up to 2,048 tokens (approximately 1,500-1,600 words), the chunking strategy for financial news articles will be optimized for this limit:

```python
def split_into_chunks(text, max_tokens=2048):
    """Split text into chunks of appropriate size for embedding with text-embedding-004.
    
    Args:
        text: The article text to split
        max_tokens: Maximum number of tokens per chunk
        
    Returns:
        List of text chunks
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Approximate tokens using character count (4 chars ≈ 1 token)
    token_estimator = lambda s: len(s) // 4
    
    for sentence in sentences:
        sentence_tokens = token_estimator(sentence)
        
        if current_length + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks
```


### Example use text-embedding-004
```python
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=gemini_api_key)

input_contents="What is the meaning of life?"

input_config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")

result = client.models.embed_content(
        model="text-embedding-004",
        contents=input_contents,
        config=input_config
)
print(result.embeddings)
# [ContentEmbedding(values=[-0.0001, 0.0002, 0.0003, ...], statistics=None)]
```


## Gemini LLM: gemini-2.0-flash

This model is used for re-ranking search results in the RAG pipeline.

### Model Details

| Property | Description |
|----------|-------------|
| Model code | `models/gemini-2.0-flash` |
| Supported data types (Input) | Audio, images, video, and text |
| Supported data types (Output) | Text |
| Input token limit | 1,048,576 |
| Output token limit | 8,192 |
| Capabilities | Structured outputs, caching, function calling, code execution, search |
| Versions | Latest: gemini-2.0-flash<br>Stable: gemini-2.0-flash-001<br>Experimental: gemini-2.0-flash-exp |
| Latest update | February 2025 |
| Knowledge cutoff | August 2024 |

### Free Tier Limitations

- **Context caching (storage)**: Free of charge, up to 1,000,000 tokens of storage per hour
- **Requests per minute**: 15
- **Requests per day**: 1,500
- **Tokens per minute**: 1,000,000

### Re-Ranking Implementation

For the RAG system, we'll use gemini-2.0-flash primarily for re-ranking search results to improve relevance:

```python
def rerank_with_gemini(query, results, top_n=5):
    """Rerank search results using Gemini's understanding of relevance.
    
    The large context window of gemini-2.0-flash (over 1M tokens) allows us to
    process many candidate documents at once for efficient re-ranking.
    """
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Prepare prompt for reranking
    system_instruction = """
    You are a financial analyst assistant. Your task is to evaluate the relevance 
    of financial news articles to a user's query. Rate each article on a scale of 0-10, 
    where 10 means the article is perfectly relevant to answering the query, and 0 means 
    it's completely irrelevant. Consider specificity, recency, and depth of information.
    
    Return a JSON array with each article's ID and its relevance score:
    [{"id": "article-id-1", "score": 8.5}, {"id": "article-id-2", "score": 6.2}, ...]
    """
    
    # Format articles for evaluation - we can include many more articles than with smaller models
    articles_text = "\n\n".join([
        f"Article ID: {result['id']}\nTitle: {result['metadata']['title']}\n"
        f"Date: {result['metadata']['published_at']}\n"
        f"Content: {result['content'][:2000]}..."  # Can include more content with large context window
        for result in results
    ])
    
    input_contents = f"""
    USER QUERY: {query}
    
    ARTICLES TO EVALUATE:
    {articles_text}
    
    Please rate the relevance of each article to the query.
    """
    
    # Generate relevance ratings
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=input_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=1024,
            temperature=0.1
        )
    )
    
    # Parse results and rerank
    try:
        scores = json.loads(response.text)
        # Create a mapping of article IDs to scores
        score_map = {item["id"]: item["score"] for item in scores}
        
        # Sort results by score
        sorted_results = sorted(
            results, 
            key=lambda x: score_map.get(x["id"], 0), 
            reverse=True
        )
        
        return sorted_results[:top_n]
    except:
        # Fallback to original ranking if parsing fails
        return results[:top_n]
```

### Example use Gemini 2.0 Flash
```python
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=gemini_api_key)

input_config=types.GenerateContentConfig(
        system_instruction="You are a wizard. You speak in riddles.",
        max_output_tokens=500,
        temperature=0.1
        )

input_contents="Explain how RAG works in a few words"

response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents=input_contents,
    config=input_config
)
print(response.text)
# From dusty tomes, knowledge I glean, then weave it with queries, a vibrant scene.
```


## Sources

- [Google Embedding Docs](https://ai.google.dev/gemini-api/docs/embeddings)
- [Google Text Generation Docs](https://ai.google.dev/gemini-api/docs/text-generation)
- [Google GenerateContentConfig Docs](https://ai.google.dev/api/generate-content)
