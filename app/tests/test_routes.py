from typing import Any

import pytest

from app.tests.conftest import SyncASGIClient


def test_upload_document_success(client: SyncASGIClient, mock_app_state: Any):
    response = client.post(
        "/documents/upload",
        files={"file": ("sample.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-1"
    assert body["status"] == "success"
    mock_app_state.documents.upload_document.assert_awaited_once()


def test_list_documents_success(client: SyncASGIClient, mock_app_state: Any):
    response = client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert body[0]["document_id"] == "doc-1"
    mock_app_state.documents.list_documents.assert_awaited_once()


def test_get_document_details_success(client: SyncASGIClient, mock_app_state: Any):
    response = client.get("/documents/doc-1")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-1"
    assert body["metadata"]["chunk_count"] == 1
    mock_app_state.documents.get_document_details.assert_awaited_once_with("doc-1")


def test_delete_document_success(client: SyncASGIClient, mock_app_state: Any):
    response = client.delete("/documents/doc-1")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_app_state.documents.delete_document.assert_awaited_once_with("doc-1")


def test_query_documents_success(client: SyncASGIClient, mock_app_state: Any):
    payload: dict[str, object] = {
        "question": "What is in the file?",
        "document_id": "doc-1",
        "history": [{"role": "user", "content": "Hi"}],
        "max_results": 3,
    }

    response = client.post("/query", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "This is a mocked answer."
    assert body["total_chunks_retrieved"] == 1
    mock_app_state.rag.query_documents.assert_awaited_once()


def test_query_documents_validation_error_for_large_max_results(client: SyncASGIClient):
    payload: dict[str, object] = {
        "question": "What is in the file?",
        "max_results": 999,
    }

    response = client.post("/query", json=payload)

    assert response.status_code == 422


def test_routes_return_503_when_state_not_initialized(app_without_state: SyncASGIClient):
    response = app_without_state.get("/documents")

    assert response.status_code == 503
    assert response.json()["detail"] == "Services are not initialized"


def test_routes_require_api_key_when_configured(
    protected_client: SyncASGIClient,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.config.settings.api_access_key", "test-secret")

    response = protected_client.get("/documents")

    assert response.status_code == 200


def test_routes_reject_missing_api_key_when_configured(
    mock_app_state: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import FastAPI

    from app.database import get_app_state
    from app.health import router as health_router
    from app.routes import router as api_router
    monkeypatch.setattr("app.config.settings.api_access_key", "test-secret")

    app = FastAPI()
    app.include_router(api_router)
    app.include_router(health_router)
    app.dependency_overrides[get_app_state] = lambda: mock_app_state

    response = SyncASGIClient(app).get("/documents")

    assert response.status_code == 401
