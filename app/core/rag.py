import time
import traceback
import re
from typing import Any, Dict, List

import aiohttp
import google.generativeai as genai
from fastapi import HTTPException

from app.core.embeddings import EmbeddingsStore
from app.schemas import ChunkInfo, ConversationTurn, QueryRequest, QueryResponse


class RAGPipeline:
    """Retrieval + generation pipeline and query orchestration."""

    def __init__(
        self,
        embeddings: EmbeddingsStore,
        llm_model: str,
        api_key: str,
        provider: str,
        llm_max_tokens: int = 1024,
    ):
        self.embeddings = embeddings
        self.llm_model = llm_model
        self.api_key = api_key
        self.provider = provider.lower()
        self.llm_max_tokens = llm_max_tokens
        self.base_url = "https://openrouter.ai/api/v1"

        if self.provider == "google":
            genai.configure(api_key=self.api_key)

    async def query_documents(self, request: QueryRequest) -> QueryResponse:
        try:
            start_time = time.time()
            question_embedding = await self.embeddings.generate_embedding(
                request.question, task_type="retrieval_query"
            )
            similar_chunks = await self.embeddings.search(
                query_embedding=question_embedding,
                max_results=request.max_results,
                filter_metadata={"document_id": request.document_id} if request.document_id else None,
            )

            if not similar_chunks:
                return QueryResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    chunks_used=[],
                    total_chunks_retrieved=0,
                    processing_time_ms=0,
                )

            answer = await self.generate_answer(
                question=request.question,
                context_chunks=similar_chunks,
                history=request.history,
            )

            chunks_info = [
                ChunkInfo(
                    chunk_id=chunk["id"],
                    text=chunk["text"],
                    similarity_score=chunk["score"],
                    document_id=chunk["metadata"]["document_id"],
                    chunk_index=chunk["metadata"]["chunk_index"],
                )
                for chunk in similar_chunks
            ]

            processing_time = (time.time() - start_time) * 1000
            return QueryResponse(
                answer=answer,
                chunks_used=chunks_info,
                total_chunks_retrieved=len(similar_chunks),
                processing_time_ms=round(processing_time, 2),
            )
        except Exception as e:
            print("❌ ERROR in query pipeline")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Query failed")

    async def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        history: List[ConversationTurn] | None = None,
        max_context_chunks: int = 5,
    ) -> str:
        context_text = self._build_context(context_chunks[:max_context_chunks])
        prompt = self._build_rag_prompt(question, context_text, history or [])
        return await self._call_llm(prompt)

    def _build_history_block(self, history: List[ConversationTurn]) -> str:
        if not history:
            return "(none)"

        lines: List[str] = []
        for turn in history[-4:]:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content.strip()}")
        return "\n".join(lines)

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for chunk in chunks:
            clean_text = self._clean_chunk_text(chunk.get("text", ""))
            if clean_text:
                parts.append(clean_text)
        return "\n\n".join(parts)

    def _clean_chunk_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = text
        # Remove injected chunk/source markers if present in retrieved text.
        cleaned = re.sub(r"\[\s*chunk\s*\d+\s*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\(\s*document\s*:[^)]+\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\(\s*relevance\s*:[^)]+\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _build_rag_prompt(self, question: str, context: str, history: List[ConversationTurn]) -> str:
        history_block = self._build_history_block(history)
        return f"""You are a helpful AI assistant that answers questions based on provided context.

PRIOR CONVERSATION:
{history_block}

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise and direct in your answer
- If multiple chunks provide relevant information, synthesize them into a coherent answer
- Do not reference chunk numbers, source labels, or metadata in your response
- Answer as a coherent paragraph
- Use prior conversation only to resolve references (for example: "that", "it", "elaborate")

QUESTION:
{question}

ANSWER:"""

    async def _call_llm(self, prompt: str) -> str:
        if self.provider == "google":
            model_name = self.llm_model.replace("google/", "").replace("models/", "")
            if not model_name.startswith("gemini"):
                model_name = "gemini-2.5-flash"
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=self.llm_max_tokens,
                        temperature=0.3,
                    ),
                )
                return response.text
            except Exception as e:
                if "not found" in str(e).lower() and model_name != "gemini-2.5-flash":
                    model = genai.GenerativeModel("gemini-2.5-flash")
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            max_output_tokens=self.llm_max_tokens,
                            temperature=0.3,
                        ),
                    )
                    return response.text
                raise Exception(f"LLM generation failed: {str(e)}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.llm_max_tokens,
            "temperature": 0.3,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"LLM generation failed: API error {response.status} - {error_text}")
                data = await response.json()
                return data["choices"][0]["message"]["content"].strip()
