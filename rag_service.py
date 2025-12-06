"""
RAG (Retrieval-Augmented Generation) Service Module

Orchestrates the RAG pipeline: retrieval of relevant context and generation of answers.
Combines vector search results with LLM capabilities to produce grounded responses.
"""

import aiohttp
from typing import List, Dict, Any
import google.generativeai as genai


class RAGService:
    """
    Service for Retrieval-Augmented Generation (RAG).
    
    RAG is a technique that enhances LLM responses by:
    1. Retrieving relevant documents from a knowledge base
    2. Including those documents as context in the prompt
    3. Having the LLM generate answers grounded in the context
    
    Why RAG?
    - Reduces hallucinations (LLM making up information)
    - Enables answers based on your specific documents
    - Keeps information up-to-date without retraining
    - Provides source attribution for answers
    
    The RAG Pipeline:
    User Question → Embed Question → Vector Search → Retrieve Chunks
    → Build Prompt → LLM Generation → Answer + Sources
    """
    
    def __init__(
        self,
        embeddings_service,
        vector_store,
        llm_model: str = "gemini-1.5-flash",
        api_key: str = None,
        provider: str = "google"
    ):
        """
        Initialize the RAG service.
        
        Args:
            embeddings_service: Service for generating embeddings
            vector_store: Vector database for similarity search
            llm_model: LLM model for answer generation
            api_key: API key (Google or OpenRouter)
            provider: "google" or "openrouter"
            
        LLM Model Selection:
        Google:
        - gemini-1.5-flash: Fast and efficient (default)
        - gemini-1.5-pro: More capable, slower
        
        OpenRouter:
        - Claude 3.5 Sonnet: Excellent reasoning, context handling
        - GPT-4: Strong performance, good for complex queries
        - GPT-3.5 Turbo: Fast and cost-effective for simple queries
        """
        self.embeddings_service = embeddings_service
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.api_key = api_key
        self.provider = provider.lower()
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Initialize Google Gemini if using Google provider
        if self.provider == "google":
            genai.configure(api_key=self.api_key)
    
    
    async def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        max_context_chunks: int = 5
    ) -> str:
        """
        Generate an answer to a question using retrieved context.
        
        This is the core RAG method that:
        1. Takes retrieved chunks from vector search
        2. Constructs a prompt with question and context
        3. Calls the LLM to generate an answer
        4. Returns the grounded response
        
        Prompt Engineering for RAG:
        - Clear instructions to use only provided context
        - Structured context presentation
        - Instructions to cite sources
        - Handling of insufficient context
        
        Args:
            question: User's question
            context_chunks: Retrieved chunks from vector search
            max_context_chunks: Maximum chunks to include in prompt
            
        Returns:
            Generated answer string
            
        Raises:
            Exception: If LLM generation fails
            
        Example flow:
            Question: "What is the refund policy?"
            Context: [chunk1: "Refunds within 30 days...", chunk2: "..."]
            Prompt: "Given context..., answer: What is the refund policy?"
            Answer: "Based on the provided policy, refunds are available..."
        """
        chunks_to_use = context_chunks[:max_context_chunks]
        
        context_text = self._build_context(chunks_to_use)
        
        prompt = self._build_rag_prompt(question, context_text)
        
        answer = await self._call_llm(prompt)
        
        return answer
    
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build a formatted context string from retrieved chunks.
        
        Format each chunk with:
        - Chunk number (for citation)
        - Source document information
        - The actual text content
        
        This structured format helps the LLM:
        - Track which information came from where
        - Cite sources in its answer
        - Distinguish between different chunks
        
        Args:
            chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted context string
            
        Example output:
            ```
            [Chunk 1] (Document: sales_report.pdf, Relevance: 0.89)
            Q3 sales increased by 25% compared to Q2...
            
            [Chunk 2] (Document: sales_report.pdf, Relevance: 0.85)
            The top performing product was Widget X...
            ```
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            doc_id = chunk["metadata"].get("document_id", "unknown")
            score = chunk.get("score", 0.0)
            
            chunk_text = f"""[Chunk {i}] (Document: {doc_id}, Relevance: {score:.2f})
{chunk['text']}
"""
            context_parts.append(chunk_text)
        
        # Join all chunks with clear separation
        return "\n\n".join(context_parts)
    
    
    def _build_rag_prompt(self, question: str, context: str) -> str:
        """
        Construct the RAG prompt for the LLM.
        
        A good RAG prompt includes:
        1. Role definition (you are a helpful assistant)
        2. Task description (answer based on context)
        3. Context provision (the retrieved chunks)
        4. Specific instructions (cite sources, be concise)
        5. The user's question
        6. Output format guidance
        
        Prompt engineering principles:
        - Be explicit about using only provided context
        - Request citations to prevent hallucination
        - Handle cases where context doesn't answer question
        - Encourage concise, direct answers
        
        Args:
            question: User's question
            context: Formatted context from retrieved chunks
            
        Returns:
            Complete prompt string for the LLM
        """
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so clearly
- Cite the chunk numbers (e.g., [Chunk 1]) when referencing specific information
- Be concise and direct in your answer
- If multiple chunks provide relevant information, synthesize them into a coherent answer

QUESTION:
{question}

ANSWER:"""
        
        return prompt
    
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM API to generate an answer.
        
        This method:
        1. Formats the request for Google Gemini or OpenRouter
        2. Sends the prompt to the LLM
        3. Extracts and returns the generated text
        
        API Parameters explained:
        - model: Which LLM to use
        - messages: Conversation format (user/assistant)
        - max_tokens: Maximum length of response
        - temperature: Randomness (0=deterministic, 1=creative)
        
        We use low temperature (0.3) for RAG because:
        - We want factual, consistent answers
        - Less creativity, more accuracy
        - Reduces hallucination risk
        
        Args:
            prompt: Complete prompt including context and question
            
        Returns:
            Generated answer text
            
        Raises:
            Exception: If API call fails
        """
        try:
            if self.provider == "google":
                # Use Google Gemini API
                # The Google SDK expects model names like "gemini-1.5-flash" (without prefixes)
                # Remove any provider prefix (e.g., "google/gemini-1.5-flash" -> "gemini-1.5-flash")
                model_name = self.llm_model.replace("google/", "").replace("models/", "")
                
                # Ensure we have a valid gemini model name
                if not model_name.startswith("gemini"):
                    model_name = "gemini-1.5-flash"  # Default to stable model
                
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            max_output_tokens=1000,
                            temperature=0.3,
                        )
                    )
                    return response.text
                except Exception as e:
                    # If model not found, try with "gemini-pro" as fallback
                    if "not found" in str(e).lower() and model_name != "gemini-pro":
                        print(f"Model {model_name} not available, trying gemini-pro...")
                        model = genai.GenerativeModel("gemini-pro")
                        response = model.generate_content(
                            prompt,
                            generation_config=genai.GenerationConfig(
                                max_output_tokens=1000,
                                temperature=0.3,
                            )
                        )
                        return response.text
                    raise
            else:
                # Use OpenRouter API
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "model": self.llm_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.3,
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"LLM API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    answer = data["choices"][0]["message"]["content"]
                    
                    return answer.strip()
                    
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    
    async def generate_answer_with_sources(
        self,
        question: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate in one method.
        
        This is a convenience method that:
        1. Embeds the question
        2. Searches for relevant chunks
        3. Generates an answer
        4. Returns answer with sources
        
        Useful for simple integrations where you want
        the entire RAG process in one call.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
            
        Example return:
        {
            "answer": "The refund policy allows...",
            "sources": [
                {"chunk_id": "doc1_chunk_0", "score": 0.89, ...},
                ...
            ],
            "question": "What is the refund policy?",
            "chunks_used": 3
        }
        """
        # Step 1: Embed question
        question_embedding = await self.embeddings_service.generate_embedding(question)
        
        # Step 2: Retrieve relevant chunks
        context_chunks = await self.vector_store.search(
            query_embedding=question_embedding,
            top_k=top_k
        )
        
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "question": question,
                "chunks_used": 0
            }
        
        # Step 3: Generate answer
        answer = await self.generate_answer(question, context_chunks)
        
        # Step 4: Return complete response
        return {
            "answer": answer,
            "sources": context_chunks,
            "question": question,
            "chunks_used": len(context_chunks)
        }