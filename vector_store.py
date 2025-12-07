"""
Vector Store Module

Manages vector storage and similarity search using ChromaDB or Pinecone.
Provides a unified interface for storing embeddings and performing nearest-neighbor search.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np


class VectorStore:
    """
    Unified interface for vector database operations.
    
    Vector databases are specialized for:
    - Storing high-dimensional vectors (embeddings)
    - Fast similarity search (nearest neighbors)
    - Metadata filtering
    - Scalability to millions of vectors
    
    We support two backends:
    1. ChromaDB - Local, embedded database (good for development/small scale)
    2. Pinecone - Cloud-based (good for production/large scale)
    
    Why vector databases instead of traditional databases?
    - Optimized for similarity search with approximate nearest neighbors (ANN)
    - Traditional DB: "Find exact match" → O(n)
    - Vector DB: "Find similar items" → O(log n) with ANN algorithms
    - Handles high-dimensional data efficiently (1536+ dimensions)
    """
    
    def __init__(
        self,
        store_type: str = "chromadb",
        collection_name: str = "document_chunks",
        api_key: Optional[str] = None,
        environment: Optional[str] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            store_type: Type of vector store ("chromadb" or "pinecone")
            collection_name: Name of the collection/index to use
            api_key: API key for Pinecone (not needed for ChromaDB)
            environment: Pinecone environment (e.g., "us-west1-gcp")
            
        ChromaDB vs Pinecone:
        
        ChromaDB:
        - Embedded database (runs in your process)
        - No external dependencies
        - Free and open source
        - Good for: development, testing, small deployments
        - Limitations: single-node, limited scale
        
        Pinecone:
        - Managed cloud service
        - Requires API key and subscription
        - Highly scalable (millions of vectors)
        - Good for: production, large scale, distributed systems
        - Features: replication, backups, monitoring
        """
        self.store_type = store_type
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        self.pinecone_api_key = api_key
        self.pinecone_environment = environment
    
    
    async def initialize(self):
        """
        Initialize the vector store connection.
        
        This sets up the database client and creates/connects to the collection.
        Called once during application startup.
        
        For ChromaDB:
        - Creates a persistent client (data saved to disk)
        - Creates or gets the collection
        
        For Pinecone:
        - Connects to cloud service
        - Creates index if it doesn't exist
        """
        if self.store_type == "chromadb":
            await self._initialize_chromadb()
        elif self.store_type == "pinecone":
            await self._initialize_pinecone()
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")
    
    
    async def _initialize_chromadb(self):
        """
        Initialize ChromaDB client and collection.
        
        ChromaDB stores data in a local directory, providing:
        - Persistence across restarts
        - Fast local queries
        - No network latency
        
        Collection creation is idempotent (safe to call multiple times).
        If collection exists, it just returns it.
        """

        self.client = chromadb.PersistentClient(
            path="./chroma_data",
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document chunks for RAG"}
        )
        
        print(f"ChromaDB initialized with collection: {self.collection_name}")
    
    
    async def _initialize_pinecone(self):
        """
        Initialize Pinecone client and index.
        
        Pinecone setup requires:
        1. Import pinecone library
        2. Initialize with API key
        3. Create index with correct dimension and metric
        4. Connect to index
        
        Note: This is a placeholder. In production, you'd:
        - Import pinecone library
        - Handle index creation with proper dimensions
        - Set up proper error handling
        """
        try:
            import pinecone
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.pinecone_api_key,
                environment=self.pinecone_environment
            )

            if self.collection_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.collection_name,
                    dimension=1536,
                    metric="cosine"
                )
            
            # Connect to index
            self.collection = pinecone.Index(self.collection_name)
            
            print(f"Pinecone initialized with index: {self.collection_name}")
            
        except ImportError:
            raise Exception("Pinecone library not installed. Run: pip install pinecone-client")
        except Exception as e:
            raise Exception(f"Pinecone initialization failed: {str(e)}")
    
    
    async def add_embeddings(self, embeddings: List[Dict[str, Any]]):
        """
        Add embeddings to the vector store.
        
        This is the core ingestion method that stores:
        - Vector embeddings (the dense arrays)
        - Original text (for retrieval)
        - Metadata (document_id, chunk_index, etc.)
        
        Each embedding dictionary should contain:
        - chunk_id: unique identifier
        - text: original text of the chunk
        - embedding: vector array
        - metadata: additional information
        
        Args:
            embeddings: List of embedding dictionaries
            
        Storage format varies by backend:
        - ChromaDB: stores vectors, text, and metadata together
        - Pinecone: stores vectors and metadata (text as metadata)
        """
        if self.store_type == "chromadb":
            await self._add_to_chromadb(embeddings)
        elif self.store_type == "pinecone":
            await self._add_to_pinecone(embeddings)
    
    
    async def _add_to_chromadb(self, embeddings: List[Dict[str, Any]]):
        """
        Add embeddings to ChromaDB collection.
        
        ChromaDB's add() method expects:
        - ids: list of unique identifiers
        - embeddings: list of vector arrays
        - documents: list of text strings (optional but recommended)
        - metadatas: list of metadata dictionaries
        
        We batch insert for efficiency.
        """
        ids = [emb["chunk_id"] for emb in embeddings]
        vectors = [emb["embedding"] for emb in embeddings]
        texts = [emb["text"] for emb in embeddings]
        metadatas = [emb["metadata"] for emb in embeddings]
        
        # Add to collection
        # ChromaDB automatically indexes vectors for fast search
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Added {len(embeddings)} embeddings to ChromaDB")
    
    
    async def _add_to_pinecone(self, embeddings: List[Dict[str, Any]]):
        """
        Add embeddings to Pinecone index.
        
        Pinecone's upsert() method expects tuples:
        (id, vector, metadata)
        
        Text is stored in metadata since Pinecone doesn't have
        a separate document field.
        """
        vectors_to_upsert = []
        
        for emb in embeddings:
            # Combine text with metadata
            metadata = emb["metadata"].copy()
            metadata["text"] = emb["text"]
            
            # Create tuple for upsert
            vectors_to_upsert.append(
                (emb["chunk_id"], emb["embedding"], metadata)
            )
        
        # Upsert to Pinecone (upsert = insert or update)
        # Batch size of 100 is recommended by Pinecone
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.collection.upsert(vectors=batch)
        
        print(f"Added {len(embeddings)} embeddings to Pinecone")
    
    
    async def search(
        self,
        query_embedding: List[float],
        max_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using nearest neighbor search.
        
        This is the core retrieval operation for RAG:
        1. Takes a query embedding (question vector)
        2. Finds most similar document chunks
        3. Returns chunks with similarity scores
        
        Similarity search algorithms:
        - Exact search: O(n) - compares to all vectors
        - ANN search: O(log n) - uses indexing (HNSW, IVF)
        
        Most vector DBs use ANN for speed at scale.
        
        Args:
            query_embedding: Query vector to search for
            max_results: Maximum number of results to return
            filter_metadata: Optional filters (e.g., document_id)
            
        Returns:
            List of matching chunks with scores and metadata
            
        Example result:
        [
            {
                "id": "doc1_chunk_0",
                "text": "The quick brown fox...",
                "score": 0.89,
                "metadata": {"document_id": "doc1", "chunk_index": 0}
            },
            ...
        ]
        """
        if self.store_type == "chromadb":
            return await self._search_chromadb(query_embedding, max_results, filter_metadata)
        elif self.store_type == "pinecone":
            return await self._search_pinecone(query_embedding, max_results, filter_metadata)
    
    
    async def _search_chromadb(
        self,
        query_embedding: List[float],
        max_results: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in ChromaDB.
        
        ChromaDB's query() method:
        - Finds nearest neighbors using cosine similarity
        - Supports metadata filtering
        - Returns results with distances and metadata
        """
        # Build where clause for filtering
        where_clause = filter_metadata if filter_metadata else None
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        # ChromaDB returns nested lists, so we flatten them
        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                # Convert distance to similarity score
                # ChromaDB returns L2 distance, we convert to similarity (1 / (1 + distance))
                distance = results["distances"][0][i]
                similarity = 1 / (1 + distance)
                
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "score": similarity,
                    "metadata": results["metadatas"][0][i]
                })
        
        return formatted_results
    
    
    async def _search_pinecone(
        self,
        query_embedding: List[float],
        max_results: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in Pinecone.
        
        Pinecone's query() method:
        - Uses HNSW (Hierarchical Navigable Small World) algorithm
        - Extremely fast even with millions of vectors
        - Returns scores (cosine similarity by default)
        """
        # Query Pinecone
        results = self.collection.query(
            vector=query_embedding,
            top_k=max_results,
            filter=filter_metadata,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results["matches"]:
            formatted_results.append({
                "id": match["id"],
                "text": match["metadata"].get("text", ""),
                "score": match["score"],
                "metadata": match["metadata"]
            })
        
        return formatted_results
    
    
    async def get_chunks_by_document_id(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks belonging to a specific document.
        
        Useful for:
        - Displaying document contents
        - Deleting a document's chunks
        - Document-level operations
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of all chunks from that document
        """
        if self.store_type == "chromadb":
            # Query with metadata filter
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            for i in range(len(results["ids"])):
                formatted_results.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
            
            return formatted_results
            
        elif self.store_type == "pinecone":
            # Pinecone doesn't have a direct "get by metadata" method
            # We'd need to query with a dummy vector and high max_results
            # Or maintain a separate mapping
            # This is a simplified version
            return []
    
    
    async def delete_by_document_id(self, document_id: str):
        """
        Delete all chunks belonging to a document.
        
        Important for:
        - Removing outdated documents
        - GDPR compliance (data deletion)
        - Storage management
        
        Args:
            document_id: ID of document to delete
        """
        if self.store_type == "chromadb":
            # Delete by metadata filter
            self.collection.delete(
                where={"document_id": document_id}
            )
            print(f"Deleted chunks for document: {document_id}")
            
        elif self.store_type == "pinecone":
            # Pinecone requires individual IDs for deletion
            # In production, you'd maintain a document_id -> chunk_ids mapping
            pass