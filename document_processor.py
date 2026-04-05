"""
Document Processing Module

Handles text extraction from multiple file formats and intelligent text chunking.
Supports PDF, DOCX, and TXT files with configurable chunking strategies.
"""

from typing import List, Dict, Any
import PyPDF2
import docx
import tiktoken
from io import BytesIO


class DocumentProcessor:
    """
    Processes documents by extracting text and chunking it for embedding.
    
    This class handles:
    - Text extraction from PDF, DOCX, and TXT files
    - Intelligent text chunking with token counting
    - Overlap between chunks to maintain context
    
    The chunking strategy is important because:
    - It ensures chunks fit within embedding model limits
    - Overlapping chunks preserve context across boundaries
    - Token-based chunking is more accurate than character-based
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target number of tokens per chunk (default: 500)
                       500 tokens ≈ 375 words, good balance for context and precision
            chunk_overlap: Number of overlapping tokens between chunks (default: 50)
                          Overlap helps maintain context continuity
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    
    def extract_text(self, file_content: bytes, file_type: str) -> str:
        """
        Extract text content from various file formats.
        
        This method routes to specific extractors based on file type.
        Each extractor is optimized for its format to preserve text quality.
        
        Args:
            file_content: Raw bytes of the uploaded file
            file_type: File extension (.pdf, .docx, or .txt)
            
        Returns:
            Extracted text as a string
            
        Raises:
            ValueError: If file type is unsupported
            Exception: If extraction fails
        """
        if file_type == ".pdf":
            return self._extract_from_pdf(file_content)
        elif file_type == ".docx":
            return self._extract_from_docx(file_content)
        elif file_type == ".txt":
            return self._extract_from_txt(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    
    def _extract_from_pdf(self, content: bytes) -> str:
        """
        Extract text from PDF files using PyPDF2.
        
        PDFs can be complex with:
        - Multiple pages
        - Various encodings
        - Embedded images (which we ignore)
        - Tables and formatting
        
        We extract pure text and preserve paragraph breaks.
        
        Args:
            content: PDF file bytes
            
        Returns:
            Extracted text with page breaks
        """
        text_parts = []
        
        try:
            # Create a PDF reader from bytes
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                
                if page_text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    
    def _extract_from_docx(self, content: bytes) -> str:
        """
        Extract text from DOCX files using python-docx.
        
        DOCX files are structured as XML with:
        - Paragraphs
        - Tables
        - Headers/footers
        - Styles and formatting
        
        We extract paragraphs which preserves natural document structure.
        
        Args:
            content: DOCX file bytes
            
        Returns:
            Extracted text with paragraph breaks
        """
        try:
            # Load DOCX from bytes
            docx_file = BytesIO(content)
            doc = docx.Document(docx_file)
            
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            return "\n\n".join(paragraphs)
            
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")
    
    
    def _extract_from_txt(self, content: bytes) -> str:
        """
        Extract text from plain text files.
        
        Handles various encodings:
        - UTF-8 (most common)
        - UTF-16 (Windows)
        - Latin-1 (fallback)
        
        Args:
            content: TXT file bytes
            
        Returns:
            Decoded text string
        """
        encodings = ['utf-8', 'utf-16', 'latin-1']
        
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        raise Exception("Could not decode text file with supported encodings")
    
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks based on token count.
        
        Chunking strategy:
        1. Tokenize the entire text
        2. Create chunks of target size
        3. Add overlap between chunks for context continuity
        4. Track token counts for each chunk
        
        Why token-based chunking?
        - Embedding models have token limits (typically 8192)
        - Token counting is more accurate than character counting
        - Ensures consistent chunk sizes for embeddings
        
        Why overlapping chunks?
        - Prevents information loss at boundaries
        - Important context might span chunk boundaries
        - Improves retrieval quality
        
        Args:
            text: Full text to be chunked
            
        Returns:
            List of dictionaries containing chunk text and metadata
            
        Example:
            Input: "The quick brown fox..." (1000 tokens)
            chunk_size: 500, overlap: 50
            Output: [
                {text: tokens[0:500], token_count: 500},
                {text: tokens[450:950], token_count: 500},
                {text: tokens[900:1000], token_count: 100}
            ]
        """
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            end_idx = start_idx + self.chunk_size
            
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text.strip(),
                "token_count": len(chunk_tokens),
                "start_token": start_idx,
                "end_token": end_idx
            })

            start_idx += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Useful for:
        - Validating text before embedding
        - Calculating costs (embeddings are priced per token)
        - Ensuring chunks fit within limits
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))