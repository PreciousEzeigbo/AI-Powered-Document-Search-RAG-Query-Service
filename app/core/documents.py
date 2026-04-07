from datetime import datetime, timezone
from io import BytesIO
import uuid
from typing import Any, Dict, List

import PyPDF2
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
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
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

    async def upload_document(self, file: UploadFile) -> DocumentUploadResponse:
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
            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail="No text could be extracted from the document")

            chunks = self.chunk_text(extracted_text)
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                embedding = await self.embeddings.generate_embedding(chunk["text"], task_type="retrieval_document")
                chunk_embeddings.append(
                    {
                        "chunk_id": f"{document_id}_chunk_{i}",
                        "text": chunk["text"],
                        "embedding": embedding,
                        "metadata": {
                            "document_id": document_id,
                            "chunk_index": i,
                            "token_count": chunk["token_count"],
                        },
                    }
                )

            await self.embeddings.add_embeddings(chunk_embeddings)
            total_tokens = sum(chunk["token_count"] for chunk in chunks)

            metadata = DocumentMetadata(
                document_id=document_id,
                filename=file.filename,
                file_type=effective_extension,
                upload_date=datetime.now(timezone.utc).isoformat(),
                chunk_count=len(chunks),
                total_tokens=total_tokens,
                extracted_text=extracted_text[:1000],
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
            logger.exception("Document processing validation error")
            raise HTTPException(status_code=400, detail=self._safe_processing_detail(e))
        except Exception as e:
            logger.exception("Document processing error")
            raise HTTPException(status_code=500, detail="Processing failed")

    async def list_documents(self) -> list[DocumentListItem]:
        documents = await self.database.get_all_documents()
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

    async def get_document_details(self, document_id: str) -> DocumentDetailResponse:
        document = await self.database.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        chunks = await self.embeddings.get_chunks_by_document_id(document_id)
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
            extracted_text=document["extracted_text"],
            chunks=formatted_chunks,
            metadata={
                "chunk_count": document["chunk_count"],
                "total_tokens": document["total_tokens"],
            },
        )

    async def delete_document(self, document_id: str) -> dict[str, str]:
        await self.embeddings.delete_by_document_id(document_id)
        await self.database.delete_document(document_id)
        return {"status": "success", "document_id": document_id}
