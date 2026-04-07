"""Application entrypoint: startup wiring and route registration."""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.documents import DocumentService
from app.core.embeddings import EmbeddingsStore
from app.core.rag import RAGPipeline
from app.database import AppState, Database
from app.health import router as health_router
from app.routes import router as api_router
from loguru import logger

# Chroma 0.4.x can emit noisy PostHog telemetry logger errors even when telemetry is disabled.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.on_event("startup")
async def startup_event() -> None:
    database = Database(db_path=settings.database_path)
    await database.initialize()

    api_key = (
        settings.google_api_key if settings.provider == "google" else settings.openrouter_api_key
    ) or ""
    embeddings = EmbeddingsStore(
        api_key=api_key,
        embedding_model=settings.embedding_model,
        provider=settings.provider,
        store_type=settings.vector_store_type,
        collection_name=settings.collection_name,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_environment=settings.pinecone_environment,
    )
    await embeddings.initialize()

    documents = DocumentService(
        database=database,
        embeddings=embeddings,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    rag = RAGPipeline(
        embeddings=embeddings,
        llm_model=settings.llm_model,
        api_key=api_key,
        provider=settings.provider,
        llm_max_tokens=settings.llm_max_tokens,
    )

    app.state.app_state = AppState(
        database=database,
        embeddings=embeddings,
        documents=documents,
        rag=rag,
    )

    logger.info("All services initialized successfully")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    app_state = getattr(app.state, "app_state", None)
    if app_state:
        await app_state.database.close()


app.include_router(api_router)
app.include_router(health_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
