"""SQLAlchemy async database layer and FastAPI dependencies."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException, Request
from loguru import logger
from sqlalchemy import Integer, String, Text, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

if TYPE_CHECKING:
    from app.core.documents import DocumentService
    from app.core.embeddings import EmbeddingsStore
    from app.core.rag import RAGPipeline


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    upload_date: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@dataclass
class DocumentMetadata:
    document_id: str
    filename: str
    file_type: str
    upload_date: str
    chunk_count: int
    total_tokens: int
    extracted_text: str


class Database:
    """Database manager backed by SQLAlchemy async sessions."""

    def __init__(self, db_path: str = "rag_documents.db"):
        self.db_path = db_path
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.db_path}", future=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Database initialized at {self.db_path}")

    async def get_session(self) -> AsyncSession:
        return self.session_factory()

    async def save_document_metadata(self, metadata: DocumentMetadata) -> None:
        async with self.session_factory() as session:
            session.add(
                Document(
                    document_id=metadata.document_id,
                    filename=metadata.filename,
                    file_type=metadata.file_type,
                    upload_date=metadata.upload_date,
                    chunk_count=metadata.chunk_count,
                    total_tokens=metadata.total_tokens,
                    extracted_text=metadata.extracted_text,
                )
            )
            await session.commit()

    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            row = await session.get(Document, document_id)
            if not row:
                return None
            return {
                "document_id": row.document_id,
                "filename": row.filename,
                "file_type": row.file_type,
                "upload_date": row.upload_date,
                "chunk_count": row.chunk_count,
                "total_tokens": row.total_tokens,
                "extracted_text": row.extracted_text or "",
            }

    async def get_all_documents(self) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            result = await session.execute(select(Document).order_by(Document.upload_date.desc()))
            rows = result.scalars().all()
            return [
                {
                    "document_id": row.document_id,
                    "filename": row.filename,
                    "file_type": row.file_type,
                    "upload_date": row.upload_date,
                    "chunk_count": row.chunk_count,
                    "total_tokens": row.total_tokens,
                }
                for row in rows
            ]

    async def delete_document(self, document_id: str) -> bool:
        async with self.session_factory() as session:
            result = await session.execute(delete(Document).where(Document.document_id == document_id))
            await session.commit()
            return (result.rowcount or 0) > 0

    async def get_documents_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(Document)
                .where(Document.upload_date.between(start_date, end_date))
                .order_by(Document.upload_date.desc())
            )
            rows = result.scalars().all()
            return [
                {
                    "document_id": row.document_id,
                    "filename": row.filename,
                    "file_type": row.file_type,
                    "upload_date": row.upload_date,
                    "chunk_count": row.chunk_count,
                    "total_tokens": row.total_tokens,
                    "extracted_text": row.extracted_text or "",
                }
                for row in rows
            ]

    async def get_statistics(self) -> Dict[str, Any]:
        async with self.session_factory() as session:
            total_documents = (await session.execute(select(func.count(Document.document_id)))).scalar() or 0
            total_chunks = (await session.execute(select(func.sum(Document.chunk_count)))).scalar() or 0
            total_tokens = (await session.execute(select(func.sum(Document.total_tokens)))).scalar() or 0

            by_file_type_result = await session.execute(
                select(Document.file_type, func.count(Document.file_type)).group_by(Document.file_type)
            )
            by_file_type = {row[0]: row[1] for row in by_file_type_result.fetchall()}

            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "by_file_type": by_file_type,
            }

    async def close(self) -> None:
        await self.engine.dispose()
        logger.debug("Database connection closed")


@dataclass
class AppState:
    database: Database
    embeddings: "EmbeddingsStore"
    documents: "DocumentService"
    rag: "RAGPipeline"


def get_app_state(request: Request) -> AppState:
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Services are not initialized")
    return state


async def get_db_session(request: Request) -> AsyncSession:
    app_state = get_app_state(request)
    async with app_state.database.session_factory() as session:
        yield session