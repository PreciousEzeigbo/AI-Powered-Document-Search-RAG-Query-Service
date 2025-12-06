"""
Embeddings Service Module

Generates vector embeddings using OpenRouter API.
Handles API communication, rate limiting, and error handling for embedding generation.
"""

import aiohttp
import asyncio
from typing import List, Optional
import numpy as np


class EmbeddingsService:
    """
    Service for generating text embeddings via OpenRouter API.
    
    Embeddings are dense vector representations of text that capture semantic meaning.
    They enable similarity search by placing semantically similar texts close together
    in vector space.
    
    Why embeddings matter for RAG:
    - Convert text into numbers that computers can compare
    - Semantic similarity (not just keyword matching)
    - Enable fast nearest-neighbor search in vector databases
    
    Example:
        "cat" and "kitten" have similar embeddings
        "cat" and "car" have different embeddings (despite similar spelling)
    """
    
    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small"):
        """
        Initialize the embeddings service.
        
        Args:
            api_key: OpenRouter API key for authentication
            model: Embedding model to use (default: text-embedding-3-small)
            
        Model choices explained:
        - text-embedding-3-small: 1536 dimensions, cost-effective, good performance
        - text-embedding-3-large: 3072 dimensions, higher quality, more expensive
        - text-embedding-ada-002: 1536 dimensions, legacy but reliable
        
        We default to text-embedding-3-small because:
        - Balance of quality and cost
        - Fast inference
        - Sufficient for most RAG use cases
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Track API statistics for monitoring
        self.total_tokens_processed = 0
        self.total_requests = 0
    
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single text string.
        
        This method:
        1. Sends text to OpenRouter API
        2. Receives a dense vector (array of floats)
        3. Returns the vector for storage
        
        The embedding vector captures the semantic meaning of the text.
        Similar texts produce similar vectors (high cosine similarity).
        
        Args:
            text: Input text to embed (should be pre-chunked)
            
        Returns:
            List of floats representing the embedding vector
            Dimension depends on model (1536 for text-embedding-3-small)
            
        Raises:
            Exception: If API call fails or returns invalid data
            
        Example:
            text = "The quick brown fox"
            embedding = await generate_embedding(text)
            # embedding = [0.021, -0.045, 0.123, ..., 0.089]  # 1536 floats
        """
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "input": text,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    # Extract embedding from response
                    # OpenRouter returns: {"data": [{"embedding": [...]}]}
                    embedding = data["data"][0]["embedding"]
                    
                    # Update statistics
                    self.total_requests += 1
                    self.total_tokens_processed += data.get("usage", {}).get("total_tokens", 0)
                    
                    return embedding
                    
        except asyncio.TimeoutError:
            raise Exception("Embedding generation timed out")
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Batching is more efficient than individual requests because:
        - Reduces HTTP overhead
        - Some APIs offer batch discounts
        - Faster overall processing time
        
        However, we need to handle:
        - Batch size limits (typically ~100 texts)
        - Total token limits per request
        - Error handling for individual items
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors, one per input text
            
        Raises:
            Exception: If batch processing fails
        """
        if not texts:
            return []
        
        # OpenRouter supports batch embedding requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "input": texts,  # Array of strings
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)  # Longer timeout for batches
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Batch API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    # Extract all embeddings
                    # Response includes embeddings in order
                    embeddings = [item["embedding"] for item in data["data"]]
                    
                    # Update statistics
                    self.total_requests += 1
                    self.total_tokens_processed += data.get("usage", {}).get("total_tokens", 0)
                    
                    return embeddings
                    
        except Exception as e:
            raise Exception(f"Batch embedding failed: {str(e)}")
    
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Cosine similarity measures the angle between two vectors:
        - 1.0 = identical direction (very similar)
        - 0.0 = perpendicular (unrelated)
        - -1.0 = opposite direction (very dissimilar)
        
        Why cosine similarity?
        - Scale-invariant (only cares about direction, not magnitude)
        - Works well with embeddings from most models
        - Efficient to compute
        - Intuitive interpretation as a score
        
        Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
            
        Example:
            sim = calculate_similarity(embedding_cat, embedding_kitten)
            # sim ≈ 0.85 (high similarity)
            
            sim = calculate_similarity(embedding_cat, embedding_car)
            # sim ≈ 0.12 (low similarity)
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return float(similarity)
    
    
    def get_statistics(self) -> dict:
        """
        Get service usage statistics.
        
        Useful for:
        - Monitoring API usage
        - Cost estimation
        - Performance tracking
        
        Returns:
            Dictionary with usage stats
        """
        return {
            "total_requests": self.total_requests,
            "total_tokens_processed": self.total_tokens_processed,
            "model": self.model
        }