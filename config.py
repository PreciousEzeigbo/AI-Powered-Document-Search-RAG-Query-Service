"""
Configuration Module

Centralizes all configuration management using environment variables.
Uses pydantic for validation and type safety.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Using pydantic-settings provides:
    - Automatic loading from .env files
    - Type validation
    - Default values
    - Clear documentation of all settings
    
    Benefits:
    - Single source of truth for configuration
    - Type safety (catches config errors early)
    - Easy to test (just override settings)
    - Works with docker, k8s, etc.
    """
    
    # API Keys
    google_api_key: str = "your_api_key_here"
    openrouter_api_key: str = "your_api_key_here"
    use_google: bool = False
    
    # Model Configuration
    embedding_model: str = "google/text-embedding-004"
    llm_model: str = "google/gemini-1.5-flash"
    provider: str = "openrouter"  # Options: "google", "openrouter", "openai", "anthropic"
    
    vector_store_type: str = "chromadb"  # Options: "chromadb" or "pinecone"
    collection_name: str = "document_chunks"
    
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    
    database_path: str = "rag_documents.db"
    
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size_mb: int = 50
    
    api_title: str = "RAG Document Search Service"
    api_version: str = "1.0.0"
    api_description: str = "AI-Powered Document Search & RAG Query Service"
    
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # RAG Configuration
    default_max_results: int = 5
    max_max_results: int = 20
    rag_temperature: float = 0.3
    rag_max_tokens: int = 1000
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create global settings instance
# This can be imported and used throughout the application
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency function for FastAPI.
    
    Can be used to inject settings into endpoints:
    
    @app.get("/config")
    async def get_config(settings: Settings = Depends(get_settings)):
        return {"chunk_size": settings.chunk_size}
    
    Returns:
        Settings instance
    """
    return settings