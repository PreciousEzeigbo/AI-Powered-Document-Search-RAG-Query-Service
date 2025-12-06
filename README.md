# AI-Powered Document Search & RAG Query Service

A FastAPI-based Retrieval-Augmented Generation (RAG) service that enables intelligent document search and question-answering capabilities using AI embeddings and vector similarity search.

## 🌟 Features

- **Document Ingestion**: Upload and process PDF, DOCX, and TXT files
- **Smart Chunking**: Automatic text splitting with configurable chunk size and overlap
- **Vector Embeddings**: Convert text chunks into embeddings using Google's text-embedding-004 model
- **Vector Storage**: Local ChromaDB for fast similarity search
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate, context-aware answers
- **FastAPI Backend**: Modern, async API with automatic documentation
- **Metadata Storage**: SQLite database for document metadata tracking

## 🏗️ Architecture

```
┌─────────────┐
│   Upload    │
│  Document   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   Document      │
│   Processor     │
│ (Text Extract)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Text Chunker  │
│ (500 chars/50   │
│   overlap)      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌──────────────┐
│   Embeddings    │─────▶│  ChromaDB    │
│   Service       │      │ (Vector DB)  │
│ (Google Gemini) │      └──────────────┘
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   SQLite DB     │
│  (Metadata)     │
└─────────────────┘

Query Flow:
Question → Embedding → Vector Search → Top-K Chunks → LLM → Answer
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Google Gemini API Key ([Get one free](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/PreciousEzeigbo/AI-Powered-Document-Search-RAG-Query-Service.git
cd AI-Powered-Document-Search-RAG-Query-Service
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install pydantic-settings google-generativeai
```

4. **Configure environment variables**
```bash
cp .env.sample .env
# Edit .env and add your Google API key
```

Required `.env` configuration:
```env
GOOGLE_API_KEY=your_google_api_key_here
USE_GOOGLE=true
EMBEDDING_MODEL=text-embedding-004
LLM_MODEL=gemini-1.5-flash
PROVIDER=google
VECTOR_STORE_TYPE=chromadb
DATABASE_PATH=rag_documents.db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
HOST=0.0.0.0
PORT=8000
```

5. **Run the service**
```bash
python3 main.py
```

The service will start at `http://localhost:8000`

## 📖 API Documentation

Once running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### 1. Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

#### 2. Query Documents
```bash
POST /query
Content-Type: application/json

{
  "question": "What is the refund policy?",
  "top_k": 5
}
```

#### 3. List Documents
```bash
GET /documents
```

#### 4. Get Document Details
```bash
GET /documents/{document_id}
```

#### 5. Health Check
```bash
GET /health
```

## 🛠️ Technology Stack

- **Framework**: FastAPI 0.104.1
- **LLM Provider**: Google Gemini (gemini-1.5-flash)
- **Embeddings**: Google text-embedding-004
- **Vector Database**: ChromaDB 0.4.18 (local)
- **Metadata Storage**: SQLite (aiosqlite)
- **Document Processing**: PyPDF2, python-docx
- **Server**: Uvicorn

## 📁 Project Structure

```
AI-RAG-Query-Service/
├── main.py                  # FastAPI application & endpoints
├── config.py                # Configuration management
├── embeddings_service.py    # Embedding generation
├── vector_store.py          # ChromaDB interface
├── rag_service.py           # RAG pipeline orchestration
├── database.py              # SQLite metadata storage
├── document_processor.py    # Document parsing & chunking
├── requirements.txt         # Python dependencies
├── .env.sample             # Environment variables template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Configuration

### Chunking Parameters
- `CHUNK_SIZE`: Number of characters per chunk (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)

### Vector Store
- **ChromaDB** (default): Local, embedded, no external dependencies
- **Pinecone** (optional): Cloud-based, requires API key

### LLM Models
Currently configured for Google Gemini, but supports:
- Google Gemini (gemini-1.5-flash, gemini-1.5-pro)
- OpenAI (gpt-4o, gpt-3.5-turbo)
- Anthropic Claude (claude-3-5-sonnet)

## 🧪 Testing

Upload a test document:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test_document.txt"
```

Query the document:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 3}'
```

## 📊 How RAG Works

1. **Document Upload**: User uploads PDF/DOCX/TXT files
2. **Text Extraction**: Content is extracted from documents
3. **Chunking**: Text is split into overlapping chunks
4. **Embedding**: Each chunk is converted to a vector embedding
5. **Storage**: Vectors stored in ChromaDB, metadata in SQLite
6. **Query Processing**:
   - User question → converted to embedding
   - Vector search finds top-K similar chunks
   - Chunks + question sent to LLM
   - LLM generates grounded answer with sources

## 🔒 Security Notes

- Never commit `.env` file (already in `.gitignore`)
- Keep API keys secure
- The `.env.sample` file shows required variables without sensitive data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Precious Ezeigbo**
- GitHub: [@PreciousEzeigbo](https://github.com/PreciousEzeigbo)

## 🙏 Acknowledgments

- Google Gemini for embeddings and LLM capabilities
- ChromaDB for vector storage
- FastAPI for the modern web framework
