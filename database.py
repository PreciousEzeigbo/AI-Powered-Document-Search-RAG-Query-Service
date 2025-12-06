"""
Database Module

Handles SQL database operations for document metadata storage.
Uses SQLite for simplicity, but can be easily adapted for PostgreSQL.
"""

import aiosqlite
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DocumentMetadata:
    """
    Data class representing document metadata.
    
    Using dataclasses provides:
    - Type hints for all fields
    - Automatic __init__, __repr__, __eq__ methods
    - Easy conversion to/from dictionaries
    - Clear structure for document information
    """
    document_id: str
    filename: str
    file_type: str
    upload_date: str
    chunk_count: int
    total_tokens: int
    extracted_text: str


class Database:
    """
    Database manager for document metadata.
    
    Why separate metadata storage from vector storage?
    1. Different access patterns
       - Vector DB: similarity search (frequent, read-heavy)
       - SQL DB: metadata queries (less frequent, structured)
    
    2. Data types
       - Vector DB: optimized for high-dimensional vectors
       - SQL DB: optimized for structured, relational data
    
    3. Query capabilities
       - Vector DB: nearest neighbor search
       - SQL DB: complex joins, aggregations, transactions
    
    4. Best tool for each job
       - Use vector DB for what it's good at (similarity)
       - Use SQL DB for what it's good at (structure)
    
    We use SQLite here for simplicity:
    - Embedded (no separate server)
    - File-based (easy deployment)
    - ACID compliant
    - Good for small to medium scale
    
    For production, consider PostgreSQL with pgvector extension,
    which can handle both metadata AND vectors in one database.
    """
    
    def __init__(self, db_path: str = "rag_documents.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
    
    
    async def initialize(self):
        """
        Initialize database connection and create tables.
        
        This method:
        1. Establishes async connection to SQLite
        2. Creates documents table if it doesn't exist
        3. Ensures proper schema is in place
        
        Called once during application startup.
        
        Table schema:
        - document_id: Primary key (UUID)
        - filename: Original filename
        - file_type: Extension (.pdf, .docx, .txt)
        - upload_date: ISO timestamp
        - chunk_count: Number of chunks created
        - total_tokens: Sum of tokens across all chunks
        - extracted_text: Preview of extracted text
        """
        self.connection = await aiosqlite.connect(self.db_path)
        
        await self.connection.execute("PRAGMA foreign_keys = ON")
        
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                extracted_text TEXT
            )
        """)

        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_upload_date 
            ON documents(upload_date)
        """)
        
        await self.connection.commit()
        
        print(f"✅ Database initialized at {self.db_path}")
    
    
    async def save_document_metadata(self, metadata: DocumentMetadata):
        """
        Save document metadata to database.
        
        Uses INSERT OR REPLACE to handle duplicates gracefully.
        If document_id already exists, it updates the record.
        
        Args:
            metadata: DocumentMetadata object to save
            
        Raises:
            Exception: If database operation fails
        """
        try:
            
            data = asdict(metadata)
            
            query = """
                INSERT OR REPLACE INTO documents 
                (document_id, filename, file_type, upload_date, 
                 chunk_count, total_tokens, extracted_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                data["document_id"],
                data["filename"],
                data["file_type"],
                data["upload_date"],
                data["chunk_count"],
                data["total_tokens"],
                data["extracted_text"]
            )
            
            await self.connection.execute(query, values)
            await self.connection.commit()
            
        except Exception as e:
            await self.connection.rollback()
            raise Exception(f"Failed to save document metadata: {str(e)}")
    
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document metadata by ID.
        
        Args:
            document_id: Unique identifier of the document
            
        Returns:
            Dictionary with document metadata, or None if not found
        """
        query = """
            SELECT * FROM documents 
            WHERE document_id = ?
        """
        
        async with self.connection.execute(query, (document_id,)) as cursor:
            row = await cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            
            return None
    
    
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieve all documents, sorted by upload date (newest first).
        
        Returns:
            List of document metadata dictionaries
        """
        query = """
            SELECT document_id, filename, file_type, upload_date,
                   chunk_count, total_tokens
            FROM documents 
            ORDER BY upload_date DESC
        """
        
        documents = []
        
        async with self.connection.execute(query) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            for row in rows:
                documents.append(dict(zip(columns, row)))
        
        return documents
    
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document metadata from database.
        
        Note: This only deletes metadata.
        You should also delete vectors from the vector store.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        query = """
            DELETE FROM documents 
            WHERE document_id = ?
        """
        
        cursor = await self.connection.execute(query, (document_id,))
        await self.connection.commit()
        
        # Check if any rows were affected
        return cursor.rowcount > 0
    
    
    async def get_documents_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents uploaded within a date range.
        
        Useful for:
        - Analytics (documents uploaded per month)
        - Cleanup (delete old documents)
        - Reporting
        
        Args:
            start_date: Start date (ISO format: YYYY-MM-DD)
            end_date: End date (ISO format: YYYY-MM-DD)
            
        Returns:
            List of matching documents
        """
        query = """
            SELECT * FROM documents 
            WHERE upload_date BETWEEN ? AND ?
            ORDER BY upload_date DESC
        """
        
        documents = []
        
        async with self.connection.execute(query, (start_date, end_date)) as cursor:
            rows = await cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            for row in rows:
                documents.append(dict(zip(columns, row)))
        
        return documents
    
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Provides insights such as:
        - Total number of documents
        - Total chunks across all documents
        - Total tokens processed
        - Distribution by file type
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Total documents
        query = "SELECT COUNT(*) FROM documents"
        async with self.connection.execute(query) as cursor:
            row = await cursor.fetchone()
            stats["total_documents"] = row[0]
        
        # Total chunks
        query = "SELECT SUM(chunk_count) FROM documents"
        async with self.connection.execute(query) as cursor:
            row = await cursor.fetchone()
            stats["total_chunks"] = row[0] or 0
        
        # Total tokens
        query = "SELECT SUM(total_tokens) FROM documents"
        async with self.connection.execute(query) as cursor:
            row = await cursor.fetchone()
            stats["total_tokens"] = row[0] or 0
        
        # Distribution by file type
        query = """
            SELECT file_type, COUNT(*) as count 
            FROM documents 
            GROUP BY file_type
        """
        async with self.connection.execute(query) as cursor:
            rows = await cursor.fetchall()
            stats["by_file_type"] = {row[0]: row[1] for row in rows}
        
        return stats
    
    
    async def close(self):
        """
        Close database connection.
        
        Should be called during application shutdown
        to ensure proper cleanup.
        """
        if self.connection:
            await self.connection.close()
            print("Database connection closed")