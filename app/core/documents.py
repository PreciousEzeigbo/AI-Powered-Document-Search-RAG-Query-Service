from datetime import datetime, timezone
import asyncio
from io import BytesIO
import uuid
from typing import Any, Dict, List

import fitz  # PyMuPDF
import docx
import tiktoken
from fastapi import HTTPException, UploadFile

from app.database import Database, DocumentMetadata
from app.schemas import DocumentDetailResponse, DocumentListItem, DocumentUploadResponse
from app.config import settings
from loguru import logger

from app.core.embeddings import EmbeddingsStore


class DocumentService:
    """Document ingestion, chunking, embedding, storage, and management."""

    # MIME types accepted for each file extension
    _ALLOWED_MIME_TYPES: Dict[str, set[str]] = {
        ".pdf": {"application/pdf"},
        ".docx": {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        },
        ".txt": {"text/plain", "application/octet-stream"},
    }

    def __init__(
        self,
        database: Database,
        embeddings: EmbeddingsStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.database = database
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def extract_text(self, file_content: bytes, file_type: str) -> str:
        if file_type == ".pdf":
            return self._extract_from_pdf(file_content)
        if file_type == ".docx":
            return self._extract_from_docx(file_content)
        if file_type == ".txt":
            return self._extract_from_txt(file_content)
        raise ValueError(f"Unsupported file type: {file_type}")

    def _detect_file_type_from_content(self, content: bytes) -> str | None:
        if content.startswith(b"%PDF-"):
            return ".pdf"

        # DOCX is a ZIP container that includes Office XML parts.
        if content.startswith(b"PK") and (b"word/" in content or b"[Content_Types].xml" in content):
            return ".docx"

        return None

    def _extract_from_pdf(self, content: bytes) -> str:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text_parts = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text and page_text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"PDF extraction failed: {str(e)}")

    def _extract_from_docx(self, content: bytes) -> str:
        try:
            doc = docx.Document(BytesIO(content))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            raise ValueError(f"DOCX extraction failed: {str(e)}")

    def _extract_from_txt(self, content: bytes) -> str:
        for encoding in ["utf-8", "utf-16", "latin-1"]:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file with supported encodings")

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            end_idx = start_idx + self.chunk_size
            chunk_tokens = tokens[start_idx:end_idx]
            chunks.append(
                {
                    "text": self.tokenizer.decode(chunk_tokens).strip(),
                    "token_count": len(chunk_tokens),
                    "start_token": start_idx,
                    "end_token": end_idx,
                }
            )
            start_idx += self.chunk_size - self.chunk_overlap

        return chunks

    @staticmethod
    def _safe_processing_detail(error: Exception) -> str:
        """Return a user-safe error detail without exposing sensitive internals."""
        raw = str(error).strip()
        if not raw:
            return "Processing failed"

        lowered = raw.lower()
        blocked_terms = ["traceback", "sqlalchemy", "sqlite", "api key", "credential", "token"]
        if any(term in lowered for term in blocked_terms):
            return "Processing failed"

        if len(raw) > 200:
            return raw[:200].rstrip() + "..."

        return raw

    async def _read_file_with_limit(self, file: UploadFile, max_size_bytes: int) -> bytes:
        chunks: list[bytes] = []
        total_bytes = 0

        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break

            total_bytes += len(chunk)
            if total_bytes > max_size_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max: {settings.max_file_size_mb}MB",
                )

            chunks.append(chunk)

        return b"".join(chunks)

    async def upload_document(
        self, file: UploadFile, session_id: str
    ) -> DocumentUploadResponse:
        try:
            if not file.filename:
                raise HTTPException(status_code=400, detail="Filename is required")

            allowed_extensions = [".pdf", ".docx", ".txt"]
            file_extension = "." + file.filename.split(".")[-1].lower()
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
                )

            # Secondary MIME-type validation
            content_type = (file.content_type or "").lower()
            allowed_mimes = self._ALLOWED_MIME_TYPES.get(file_extension, set())
            if content_type and allowed_mimes and content_type not in allowed_mimes:
                logger.warning(
                    "MIME mismatch: ext={} content_type={}", file_extension, content_type
                )

            max_size_bytes = settings.max_file_size_mb * 1024 * 1024
            if hasattr(file, 'size') and file.size and file.size > max_size_bytes:
                raise HTTPException(status_code=413, detail=f"File too large. Max: {settings.max_file_size_mb}MB")

            document_id = str(uuid.uuid4())
            content = await self._read_file_with_limit(file, max_size_bytes)

            detected_extension = self._detect_file_type_from_content(content)
            effective_extension = file_extension

            # Gracefully handle mismatched extension/content for supported formats.
            if detected_extension and detected_extension != file_extension:
                if file_extension in {".pdf", ".docx"} and detected_extension in {".pdf", ".docx"}:
                    effective_extension = detected_extension
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File content does not match extension {file_extension}",
                    )

            extracted_text = self.extract_text(content, effective_extension)

            # V-02: Clear raw file content from memory immediately after extraction
            del content

            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail="No text could be extracted from the document")

            chunks = self.chunk_text(extracted_text)

            # V-02: Clear extracted text from memory after chunking
            del extracted_text
            
            if len(chunks) > settings.max_chunks_per_document:
                raise HTTPException(
                    status_code=413,
                    detail=f"Document contains too much text ({len(chunks)} chunks). The maximum allowed for this API tier is {settings.max_chunks_per_document} chunks."
                )

            texts = [chunk["text"] for chunk in chunks]
            
            embeddings = await self.embeddings.generate_embeddings_batch(
                texts, task_type="retrieval_document"
            )

            chunk_embeddings = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_embeddings.append(
                    {
                        "chunk_id": f"{document_id}_chunk_{i}",
                        "text": chunk["text"],
                        "embedding": embedding,
                        "metadata": {
                            "document_id": document_id,
                            "session_id": session_id,
                            "chunk_index": i,
                            "token_count": chunk["token_count"],
                        },
                    }
                )

            await self.embeddings.add_embeddings(chunk_embeddings)
            total_tokens = sum(chunk["token_count"] for chunk in chunks)

            # V-01: No extracted_text stored in database
            metadata = DocumentMetadata(
                document_id=document_id,
                session_id=session_id,
                filename=file.filename,
                file_type=effective_extension,
                upload_date=datetime.now(timezone.utc).isoformat(),
                chunk_count=len(chunks),
                total_tokens=total_tokens,
            )
            await self.database.save_document_metadata(metadata)

            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                total_chunks=len(chunks),
                total_tokens=total_tokens,
                status="success",
                message=f"Document processed and {len(chunks)} chunks indexed",
            )
        except HTTPException:
            raise
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning("Document processing validation error: {}", type(e).__name__)
            raise HTTPException(status_code=400, detail=self._safe_processing_detail(e))
        except Exception as e:
            logger.exception("Document processing error")
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str or "resourceexhausted" in error_str:
                raise HTTPException(
                    status_code=429, 
                    detail="API Quota Exhausted: Your daily Google Free Tier embedding limit (1,000 requests/day) has been reached. Please try again tomorrow or switch to a paid API key."
                )
            raise HTTPException(status_code=500, detail="Processing failed")

    async def list_documents(self, session_id: str) -> list[DocumentListItem]:
        documents = await self.database.get_all_documents(session_id)
        return [
            DocumentListItem(
                document_id=doc["document_id"],
                filename=doc["filename"],
                file_type=doc["file_type"],
                upload_date=doc["upload_date"],
                chunk_count=doc["chunk_count"],
                total_tokens=doc["total_tokens"],
            )
            for doc in documents
        ]

    async def get_document_details(
        self, document_id: str, session_id: str
    ) -> DocumentDetailResponse:
        document = await self.database.get_document_by_id(document_id, session_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        chunks = await self.embeddings.get_chunks_by_document_id(document_id, session_id)
        formatted_chunks = [
            {
                "chunk_id": chunk["id"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "text": chunk["text"],
                "token_count": chunk["metadata"]["token_count"],
            }
            for chunk in chunks
        ]

        return DocumentDetailResponse(
            document_id=document["document_id"],
            filename=document["filename"],
            file_type=document["file_type"],
            upload_date=document["upload_date"],
            chunks=formatted_chunks,
            metadata={
                "chunk_count": document["chunk_count"],
                "total_tokens": document["total_tokens"],
            },
        )

    async def delete_document(
        self, document_id: str, session_id: str
    ) -> dict[str, str]:
        # Verify ownership first
        document = await self.database.get_document_by_id(document_id, session_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        await self.embeddings.delete_by_document_id(document_id, session_id)
        await self.database.delete_document(document_id, session_id)
        return {"status": "success", "document_id": document_id}
