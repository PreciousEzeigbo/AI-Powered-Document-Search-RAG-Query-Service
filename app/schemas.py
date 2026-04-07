from pydantic import BaseModel, Field
from typing import List, Dict, Any

from app.config import settings


class ConversationTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""
    question: str = Field(..., description="User question to answer", min_length=1)
    document_id: str | None = Field(
        default=None,
        description="Optional document scope for retrieval",
    )
    history: List[ConversationTurn] = Field(
        default_factory=list,
        description="Recent conversation turns for follow-up context",
    )
    max_results: int = Field(
        settings.default_max_results,
        description="Maximum number of relevant chunks to retrieve",
        ge=1,
        le=settings.max_max_results,
    )


class ChunkInfo(BaseModel):
    """Information about a retrieved document chunk."""
    chunk_id: str
    text: str
    similarity_score: float
    document_id: str
    chunk_index: int


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint."""
    answer: str
    chunks_used: List[ChunkInfo]
    total_chunks_retrieved: int
    processing_time_ms: float


class DocumentUploadResponse(BaseModel):
    """Response model for document upload endpoint."""
    document_id: str
    filename: str
    total_chunks: int
    total_tokens: int
    status: str
    message: str


class DocumentListItem(BaseModel):
    """Model for document list item."""
    document_id: str
    filename: str
    file_type: str
    upload_date: str
    chunk_count: int
    total_tokens: int


class DocumentDetailResponse(BaseModel):
    """Detailed response for a specific document."""
    document_id: str
    filename: str
    file_type: str
    upload_date: str
    extracted_text: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
