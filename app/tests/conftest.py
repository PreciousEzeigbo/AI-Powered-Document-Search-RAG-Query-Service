import asyncio
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import AsyncMock
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI

from app.config import settings
from app.database import AppState, get_app_state
from app.health import router as health_router
from app.routes import router as api_router
from app.schemas import ChunkInfo, QueryResponse
from app.security import SessionContext, require_session


# A fixed session ID used across all tests
TEST_SESSION_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


class SyncASGIClient:
    def __init__(self, app: FastAPI, headers: dict[str, str] | None = None):
        self.app = app
        self.headers = headers or {}

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=self.headers,
        ) as client:
            return await client.request(method, url, **kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        return asyncio.run(self._request(method, url, **kwargs))

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


@pytest.fixture
def mock_app_state() -> Any:
    documents: Any = SimpleNamespace()
    documents.upload_document = AsyncMock(
        return_value={
            "document_id": "doc-1",
            "filename": "sample.txt",
            "total_chunks": 1,
            "total_tokens": 12,
            "status": "success",
            "message": "Document processed",
        }
    )
    documents.list_documents = AsyncMock(
        return_value=[
            {
                "document_id": "doc-1",
                "filename": "sample.txt",
                "file_type": ".txt",
                "upload_date": "2026-04-07T00:00:00Z",
                "chunk_count": 1,
                "total_tokens": 12,
            }
        ]
    )
    documents.get_document_details = AsyncMock(
        return_value={
            "document_id": "doc-1",
            "filename": "sample.txt",
            "file_type": ".txt",
            "upload_date": "2026-04-07T00:00:00Z",
            "chunks": [
                {
                    "chunk_id": "doc-1_chunk_0",
                    "chunk_index": 0,
                    "text": "hello world",
                    "token_count": 2,
                }
            ],
            "metadata": {
                "chunk_count": 1,
                "total_tokens": 12,
            },
        }
    )
    documents.delete_document = AsyncMock(
        return_value={
            "status": "success",
            "document_id": "doc-1",
        }
    )

    rag: Any = SimpleNamespace()
    response = QueryResponse(
        answer="This is a mocked answer.",
        chunks_used=[
            ChunkInfo(
                chunk_id="doc-1_chunk_0",
                text="hello world",
                similarity_score=0.98,
                document_id="doc-1",
                chunk_index=0,
            )
        ],
        total_chunks_retrieved=1,
        processing_time_ms=3.1,
    )
    rag.query_documents = AsyncMock(return_value=response.model_dump())

    return AppState(
        database=cast(Any, None),
        embeddings=cast(Any, None),
        documents=documents,
        rag=rag,
    )


def _build_app(
    mock_state: Any | None = None,
    session_override: SessionContext | None = None,
) -> FastAPI:
    """Helper to build a test FastAPI app with optional overrides."""
    test_app = FastAPI()
    test_app.include_router(api_router)
    test_app.include_router(health_router)

    if mock_state is not None:
        test_app.dependency_overrides[get_app_state] = lambda: mock_state

    # Always override session dependency in tests
    ctx = session_override or SessionContext(session_id=TEST_SESSION_ID)
    test_app.dependency_overrides[require_session] = lambda: ctx

    return test_app


@pytest.fixture
def client(mock_app_state: Any) -> Generator[SyncASGIClient, None, None]:
    test_app = _build_app(mock_state=mock_app_state)
    yield SyncASGIClient(test_app, headers={"X-Session-Id": TEST_SESSION_ID})


@pytest.fixture
def app_without_state() -> Generator[SyncASGIClient, None, None]:
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[require_session] = lambda: SessionContext(
        session_id=TEST_SESSION_ID
    )

    yield SyncASGIClient(app, headers={"X-Session-Id": TEST_SESSION_ID})


@pytest.fixture
def protected_client(mock_app_state: Any, monkeypatch: pytest.MonkeyPatch) -> SyncASGIClient:
    monkeypatch.setattr(settings, "api_access_key", "test-secret")

    test_app = _build_app(mock_state=mock_app_state)

    return SyncASGIClient(test_app, headers={
        "X-API-Key": "test-secret",
        "X-Session-Id": TEST_SESSION_ID,
    })
