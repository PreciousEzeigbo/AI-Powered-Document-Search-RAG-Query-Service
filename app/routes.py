from fastapi import APIRouter, Depends, File, UploadFile

from app.database import AppState, get_app_state
from app.schemas import DocumentDetailResponse, DocumentListItem, DocumentUploadResponse, QueryRequest, QueryResponse
from app.security import require_api_key

router = APIRouter(tags=["documents", "query"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
    state: AppState = Depends(get_app_state),
) -> DocumentUploadResponse:
    return await state.documents.upload_document(file)


@router.get("/documents", response_model=list[DocumentListItem])
async def list_documents(
    _: None = Depends(require_api_key),
    state: AppState = Depends(get_app_state),
) -> list[DocumentListItem]:
    return await state.documents.list_documents()


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_details(
    document_id: str,
    _: None = Depends(require_api_key),
    state: AppState = Depends(get_app_state),
) -> DocumentDetailResponse:
    return await state.documents.get_document_details(document_id)


@router.delete("/documents/{document_id}", response_model=dict[str, str])
async def delete_document(
    document_id: str,
    _: None = Depends(require_api_key),
    state: AppState = Depends(get_app_state),
) -> dict[str, str]:
    return await state.documents.delete_document(document_id)


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    _: None = Depends(require_api_key),
    state: AppState = Depends(get_app_state),
) -> QueryResponse:
    return await state.rag.query_documents(request)
