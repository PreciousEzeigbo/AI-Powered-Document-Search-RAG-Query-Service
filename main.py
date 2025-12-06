"""
AI-Powered Document Search & RAG Query Service

This FastAPI application provides document ingestion, embedding, vector storage,
and retrieval-augmented generation (RAG) capabilities for question answering.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import uuid
import os

# Import our custom modules
from config import settings, get_settings
from document_processor import DocumentProcessor
from embeddings_service import EmbeddingsService
from vector_store import VectorStore
from rag_service import RAGService
from database import Database, DocumentMetadata

# Initialize FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version
)

# Global service instances (initialized on startup)
document_processor: DocumentProcessor = None
embeddings_service: EmbeddingsService = None
vector_store: VectorStore = None
rag_service: RAGService = None
database: Database = None


# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    """Request model for RAG query endpoint"""
    question: str = Field(..., description="User question to answer", min_length=1)
    top_k: int = Field(
        settings.default_top_k,
        description="Number of relevant chunks to retrieve",
        ge=1,
        le=settings.max_top_k
    )
    
    
class ChunkInfo(BaseModel):
    """Information about a retrieved document chunk"""
    chunk_id: str
    text: str
    similarity_score: float
    document_id: str
    chunk_index: int


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint"""
    answer: str
    chunks_used: List[ChunkInfo]
    total_chunks_retrieved: int
    processing_time_ms: float


class DocumentUploadResponse(BaseModel):
    """Response model for document upload endpoint"""
    document_id: str
    filename: str
    total_chunks: int
    total_tokens: int
    status: str
    message: str


class DocumentListItem(BaseModel):
    """Model for document list item"""
    document_id: str
    filename: str
    file_type: str
    upload_date: str
    chunk_count: int
    total_tokens: int


class DocumentDetailResponse(BaseModel):
    """Detailed response for a specific document"""
    document_id: str
    filename: str
    file_type: str
    upload_date: str
    extracted_text: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """
    Initialize all service components when the application starts.
    
    This function sets up:
    - Database connection for metadata storage
    - Document processor for text extraction
    - Embeddings service for vector generation
    - Vector store for similarity search
    - RAG service for question answering
    """
    global document_processor, embeddings_service, vector_store, rag_service, database
    
    # Initialize database (SQLite for simplicity, can be replaced with PostgreSQL)
    database = Database(db_path=settings.database_path)
    await database.initialize()
    
    # Initialize document processor
    # This handles PDF, DOCX, and TXT file parsing
    document_processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # Initialize embeddings service
    # Uses OpenRouter API for generating vector embeddings
    embeddings_service = EmbeddingsService(
        api_key=settings.openrouter_api_key,
        model=settings.embedding_model
    )
    
    # Initialize vector store
    # Using ChromaDB for local development (switch to Pinecone for production)
    vector_store = VectorStore(
        store_type=settings.vector_store_type,
        collection_name=settings.collection_name,
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_environment
    )
    await vector_store.initialize()
    
    # Initialize RAG service
    # Orchestrates retrieval and generation
    rag_service = RAGService(
        embeddings_service=embeddings_service,
        vector_store=vector_store,
        llm_model=settings.llm_model,
        api_key=settings.openrouter_api_key
    )
    
    print("✅ All services initialized successfully")


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, DOCX, or TXT).
    
    This endpoint performs the following steps:
    1. Validates file type
    2. Extracts text from the document
    3. Splits text into overlapping chunks
    4. Generates embeddings for each chunk using OpenRouter
    5. Stores embeddings in vector database
    6. Saves metadata to SQL database
    
    Args:
        file: Uploaded file (PDF, DOCX, or TXT)
        
    Returns:
        DocumentUploadResponse with document ID and processing statistics
        
    Raises:
        HTTPException: If file type is unsupported or processing fails
    """
    try:
        # Validate file type
        allowed_extensions = [".pdf", ".docx", ".txt"]
        file_extension = "." + file.filename.split(".")[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Read file content
        content = await file.read()
        
        # Extract text from document
        # The processor handles different file formats and returns clean text
        extracted_text = document_processor.extract_text(content, file_extension)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document"
            )
        
        # Chunk the text into manageable pieces
        # Chunking is important because:
        # - LLMs have context limits
        # - Smaller chunks improve retrieval precision
        # - Overlap maintains context between chunks
        chunks = document_processor.chunk_text(extracted_text)
        
        # Generate embeddings for each chunk
        # Embeddings convert text into dense vectors that capture semantic meaning
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = await embeddings_service.generate_embedding(chunk["text"])
            chunk_embeddings.append({
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": chunk["text"],
                "embedding": embedding,
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": i,
                    "token_count": chunk["token_count"]
                }
            })
        
        # Store embeddings in vector database
        # Vector stores enable fast similarity search using cosine similarity or other metrics
        await vector_store.add_embeddings(chunk_embeddings)
        
        # Calculate total tokens
        total_tokens = sum(chunk["token_count"] for chunk in chunks)
        
        # Save document metadata to SQL database
        # This allows us to track documents and link chunks back to source files
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=file.filename,
            file_type=file_extension,
            upload_date=datetime.utcnow().isoformat(),
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            extracted_text=extracted_text[:1000]  # Store preview only
        )
        await database.save_document_metadata(metadata)
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            status="success",
            message=f"Document processed and {len(chunks)} chunks indexed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Answer a question using RAG (Retrieval-Augmented Generation).
    
    This endpoint implements the RAG pipeline:
    1. Embed the user's question into a vector
    2. Search vector database for most similar document chunks
    3. Retrieve top-K relevant chunks based on similarity
    4. Construct a prompt with retrieved context
    5. Call LLM to generate an answer grounded in the context
    6. Return answer with source chunks and similarity scores
    
    Args:
        request: QueryRequest containing question and retrieval parameters
        
    Returns:
        QueryResponse with answer, source chunks, and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        import time
        start_time = time.time()
        
        # Generate embedding for the user's question
        # This converts the question into the same vector space as document chunks
        question_embedding = await embeddings_service.generate_embedding(request.question)
        
        # Retrieve most similar chunks from vector store
        # Uses cosine similarity or Euclidean distance to find relevant context
        similar_chunks = await vector_store.search(
            query_embedding=question_embedding,
            top_k=request.top_k
        )
        
        if not similar_chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                chunks_used=[],
                total_chunks_retrieved=0,
                processing_time_ms=0
            )
        
        # Generate answer using RAG
        # The RAG service constructs a prompt with context and calls the LLM
        answer = await rag_service.generate_answer(
            question=request.question,
            context_chunks=similar_chunks
        )
        
        # Format chunk information for response
        chunks_info = [
            ChunkInfo(
                chunk_id=chunk["id"],
                text=chunk["text"],
                similarity_score=chunk["score"],
                document_id=chunk["metadata"]["document_id"],
                chunk_index=chunk["metadata"]["chunk_index"]
            )
            for chunk in similar_chunks
        ]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            chunks_used=chunks_info,
            total_chunks_retrieved=len(similar_chunks),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentListItem])
async def list_documents():
    """
    List all uploaded documents with their metadata.
    
    Returns a summary of all documents in the system, including:
    - Document ID for reference
    - Original filename
    - File type (PDF, DOCX, TXT)
    - Upload timestamp
    - Number of chunks created
    - Total token count
    
    Returns:
        List of DocumentListItem objects
        
    Raises:
        HTTPException: If database query fails
    """
    try:
        documents = await database.get_all_documents()
        
        return [
            DocumentListItem(
                document_id=doc["document_id"],
                filename=doc["filename"],
                file_type=doc["file_type"],
                upload_date=doc["upload_date"],
                chunk_count=doc["chunk_count"],
                total_tokens=doc["total_tokens"]
            )
            for doc in documents
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")


@app.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_details(document_id: str):
    """
    Get detailed information about a specific document.
    
    Retrieves comprehensive information including:
    - Full metadata (filename, type, upload date)
    - Extracted text content
    - All chunks with their text and token counts
    - Additional metadata
    
    Args:
        document_id: Unique identifier of the document
        
    Returns:
        DocumentDetailResponse with complete document information
        
    Raises:
        HTTPException: If document not found or retrieval fails
    """
    try:
        document = await database.get_document_by_id(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Retrieve chunks from vector store
        # This gets all chunks associated with this document
        chunks = await vector_store.get_chunks_by_document_id(document_id)
        
        formatted_chunks = [
            {
                "chunk_id": chunk["id"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "text": chunk["text"],
                "token_count": chunk["metadata"]["token_count"]
            }
            for chunk in chunks
        ]
        
        return DocumentDetailResponse(
            document_id=document["document_id"],
            filename=document["filename"],
            file_type=document["file_type"],
            upload_date=document["upload_date"],
            extracted_text=document["extracted_text"],
            chunks=formatted_chunks,
            metadata={
                "chunk_count": document["chunk_count"],
                "total_tokens": document["total_tokens"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns:
        JSON response indicating service health
    """
    return {"status": "healthy", "service": "RAG Document Search"}


if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )