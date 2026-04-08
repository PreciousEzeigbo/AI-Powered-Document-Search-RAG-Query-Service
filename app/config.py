"""
Configuration Module

Centralizes all configuration management using environment variables.
Uses pydantic for validation and type safety.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import Any, Optional
import os


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
    google_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    use_google: bool = False
    
    # Model Configuration
    embedding_model: str = "google/text-embedding-004"
    llm_model: str = "google/gemini-1.5-flash"
    provider: str = "openrouter"  # Options: "google", "openrouter", "openai", "anthropic"
    
    vector_store_type: str = "chromadb"  # Options: "chromadb" or "pinecone"
    collection_name: str = "document_chunks"
    
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    api_access_key: Optional[str] = None
    
    database_path: str = "rag_documents.db"
    
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size_mb: int = 50
    max_chunks_per_document: int = 150
    
    api_title: str = "RAG Document Search Service"
    api_version: str = "1.0.0"
    api_description: str = "AI-Powered Document Search & RAG Query Service"
    
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # RAG Configuration
    default_max_results: int = 5
    max_max_results: int = 20
    rag_temperature: float = 0.3
    rag_max_tokens: int = 1000
    llm_max_tokens: int = 1024

    # Session & Retention
    session_ttl_hours: int = 24  # Auto-purge sessions after this many hours (0 = disabled)

    # Rate Limiting
    rate_limit_upload: str = "10/minute"
    rate_limit_query: str = "30/minute"

    @model_validator(mode='before')
    @classmethod
    def map_legacy_rag_tokens(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if values.get('llm_max_tokens') is None and values.get('rag_max_tokens') is not None:
                values['llm_max_tokens'] = values['rag_max_tokens']
        return values
    
    # CORS Configuration
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    
    @model_validator(mode='after')
    def validate_settings(self):
        # Validate API keys
        if os.getenv('ENV') != 'dev':
            provider = (self.provider or '').strip().lower()
            if provider == 'google' or self.use_google:
                if not self.google_api_key or 'your_api_key_here' in self.google_api_key:
                    raise ValueError('GOOGLE_API_KEY must be set when provider=google')
            elif provider == 'openrouter':
                if not self.openrouter_api_key or 'your_api_key_here' in self.openrouter_api_key:
                    raise ValueError('OPENROUTER_API_KEY must be set when provider=openrouter')

        # Validate CORS: reject allow_credentials=True with wildcard origins
        if "*" in self.cors_origins and len(self.cors_origins) == 1:
            raise ValueError(
                "CORS misconfiguration: wildcard origins ('*') should not be used. "
                "Specify explicit allowed origins."
            )

        return self

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