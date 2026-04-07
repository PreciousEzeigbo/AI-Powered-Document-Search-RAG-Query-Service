import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
import chromadb
from chromadb.config import Settings
import google.generativeai as genai


class EmbeddingsStore:
    """Embedding generation and vector storage operations in one place."""

    def __init__(
        self,
        api_key: str,
        embedding_model: str,
        provider: str,
        store_type: str,
        collection_name: str,
        pinecone_api_key: Optional[str] = None,
        pinecone_environment: Optional[str] = None,
    ):
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.provider = provider.lower()
        self.base_url = "https://openrouter.ai/api/v1"

        self.store_type = store_type
        self.collection_name = collection_name
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment

        self.client = None
        self.collection = None

        if self.provider == "google":
            genai.configure(api_key=self.api_key)

    def _normalize_google_model_name(self, model_name: str) -> str:
        normalized = (model_name or "").strip()
        if normalized.startswith("google/"):
            normalized = normalized[len("google/") :]
        if normalized.startswith("models/"):
            normalized = normalized[len("models/") :]
        return normalized

    def _google_candidate_models(self) -> List[str]:
        configured = self._normalize_google_model_name(self.embedding_model)

        # Keep candidates ordered by preference; remove duplicates while preserving order.
        seeds = [
            configured,
            "gemini-embedding-001",
            "gemini-embedding-2-preview",
            "text-embedding-004",
            "embedding-001",
        ]

        candidates: List[str] = []
        for model in seeds:
            if model and model not in candidates:
                candidates.append(model)

        try:
            for model in genai.list_models():
                methods = getattr(model, "supported_generation_methods", []) or []
                if "embedContent" in methods:
                    name = self._normalize_google_model_name(getattr(model, "name", ""))
                    if name and name not in candidates:
                        candidates.append(name)
        except Exception:
            # Discovery is best-effort; static candidates are still attempted.
            pass

        return candidates

    async def initialize(self) -> None:
        if self.store_type == "chromadb":
            self.client = chromadb.PersistentClient(
                path="./chroma_data",
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for RAG"},
            )
            print(f"ChromaDB initialized with collection: {self.collection_name}")
            return

        if self.store_type == "pinecone":
            import pinecone

            pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
            if self.collection_name not in pinecone.list_indexes():
                pinecone.create_index(name=self.collection_name, dimension=1536, metric="cosine")
            self.collection = pinecone.Index(self.collection_name)
            print(f"Pinecone initialized with index: {self.collection_name}")
            return

        raise ValueError(f"Unsupported store type: {self.store_type}")

    async def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        if self.provider == "google":
            candidates = self._google_candidate_models()

            loop = asyncio.get_event_loop()
            last_error: Optional[Exception] = None
            for candidate in candidates:
                try:
                    def _embed_with_task(m: str) -> Any:
                        try:
                            return genai.embed_content(
                                model=f"models/{m}",
                                content=text,
                                task_type=task_type,
                            )
                        except Exception as e:
                            msg = str(e).lower()
                            if (
                                "task_type" in msg
                                or "unknown field" in msg
                                or "unexpected keyword" in msg
                                or "invalid argument" in msg
                            ):
                                return genai.embed_content(
                                    model=f"models/{m}",
                                    content=text,
                                )
                            raise

                    result = await loop.run_in_executor(None, lambda m=candidate: _embed_with_task(m))
                    return result["embedding"]
                except Exception as model_error:
                    last_error = model_error
                    lowered = str(model_error).lower()
                    if "not found" in lowered or "not supported" in lowered:
                        continue
                    raise Exception(f"Failed to generate embedding: {str(model_error)}") from model_error

            raise Exception(f"Failed to generate embedding: {str(last_error)}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.embedding_model, "input": text}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to generate embedding: API error {response.status} - {error_text}")
                data = await response.json()
                return data["data"][0]["embedding"]

    async def add_embeddings(self, embeddings: List[Dict[str, Any]]) -> None:
        if self.store_type == "chromadb":
            self.collection.add(
                ids=[emb["chunk_id"] for emb in embeddings],
                embeddings=[emb["embedding"] for emb in embeddings],
                documents=[emb["text"] for emb in embeddings],
                metadatas=[emb["metadata"] for emb in embeddings],
            )
            return

        vectors_to_upsert = []
        for emb in embeddings:
            metadata = emb["metadata"].copy()
            metadata["text"] = emb["text"]
            vectors_to_upsert.append((emb["chunk_id"], emb["embedding"], metadata))

        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            self.collection.upsert(vectors=vectors_to_upsert[i : i + batch_size])

    async def search(
        self,
        query_embedding: List[float],
        max_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.store_type == "chromadb":
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=filter_metadata if filter_metadata else None,
                include=["documents", "metadatas", "distances"],
            )

            formatted_results: List[Dict[str, Any]] = []
            if results["ids"]:
                for i in range(len(results["ids"][0])):
                    distance = results["distances"][0][i]
                    similarity = 1 / (1 + distance)
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "text": results["documents"][0][i],
                            "score": similarity,
                            "metadata": results["metadatas"][0][i],
                        }
                    )
            return formatted_results

        results = self.collection.query(
            vector=query_embedding,
            top_k=max_results,
            filter=filter_metadata,
            include_metadata=True,
        )
        return [
            {
                "id": match["id"],
                "text": match["metadata"].get("text", ""),
                "score": match["score"],
                "metadata": match["metadata"],
            }
            for match in results["matches"]
        ]

    async def get_chunks_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        if self.store_type == "chromadb":
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
            return [
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
                for i in range(len(results["ids"]))
            ]
        return []

    async def delete_by_document_id(self, document_id: str) -> None:
        if self.store_type == "chromadb":
            self.collection.delete(where={"document_id": document_id})
            return

        # Pinecone requires an id list for deletion. Left as a no-op until id mapping is persisted.
        return
