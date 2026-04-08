from fastapi import APIRouter, Depends, File, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.database import AppState, get_app_state
from app.schemas import (
    DocumentDetailResponse,
    DocumentListItem,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)
from app.security import SessionContext, require_api_key, require_session

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(tags=["documents", "query"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
@limiter.limit(settings.rate_limit_upload)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
    session: SessionContext = Depends(require_session),
    state: AppState = Depends(get_app_state),
) -> DocumentUploadResponse:
    return await state.documents.upload_document(file, session.session_id)


@router.get("/documents", response_model=list[DocumentListItem])
async def list_documents(
    _: None = Depends(require_api_key),
    session: SessionContext = Depends(require_session),
    state: AppState = Depends(get_app_state),
) -> list[DocumentListItem]:
    return await state.documents.list_documents(session.session_id)


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_details(
    document_id: str,
    _: None = Depends(require_api_key),
    session: SessionContext = Depends(require_session),
    state: AppState = Depends(get_app_state),
) -> DocumentDetailResponse:
    return await state.documents.get_document_details(document_id, session.session_id)


@router.delete("/documents/{document_id}", response_model=dict[str, str])
async def delete_document(
    document_id: str,
    _: None = Depends(require_api_key),
    session: SessionContext = Depends(require_session),
    state: AppState = Depends(get_app_state),
) -> dict[str, str]:
    return await state.documents.delete_document(document_id, session.session_id)


@router.post("/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit_query)
async def query_documents(
    request: Request,
    body: QueryRequest,
    _: None = Depends(require_api_key),
    session: SessionContext = Depends(require_session),
    state: AppState = Depends(get_app_state),
) -> QueryResponse:
    return await state.rag.query_documents(body, session.session_id)
