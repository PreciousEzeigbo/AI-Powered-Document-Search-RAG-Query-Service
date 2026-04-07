// API client functions for RAG backend integration
// Falls back to demo mode when NEXT_PUBLIC_API_URL is not configured

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
const DEMO_MODE = !API_BASE_URL; // Enable demo mode when no API URL is configured
export const MAX_UPLOAD_SIZE_MB = 50;
export const MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024;
const API_ACCESS_KEY = process.env.NEXT_PUBLIC_API_ACCESS_KEY;

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
}

export interface QueryResponse {
  answer: string;
  sources: string[];
}

export type QueryHistoryTurn = {
  role: 'user' | 'assistant';
  content: string;
};

export interface QueryRequestPayload {
  question: string;
  documentId: string;
  history: QueryHistoryTurn[];
}

type BackendChunk = {
  chunk_id: string;
  text: string;
};

type BackendQueryResponse = {
  answer: string;
  chunks_used?: BackendChunk[];
};

// Demo responses for testing without a backend
const DEMO_RESPONSES: Record<string, QueryResponse> = {
  default: {
    answer: 'Based on the uploaded documents, I can help answer questions about their content. Since this is demo mode without a connected RAG backend, please configure NEXT_PUBLIC_API_URL in your environment variables to connect to your actual RAG system.',
    sources: ['Demo Mode - Connect Real Backend'],
  },
};

export async function uploadDocument(file: File): Promise<UploadResponse> {
  if (DEMO_MODE) {
    // Demo mode: simulate successful upload with a generated ID
    await new Promise(resolve => setTimeout(resolve, 800));
    return {
      document_id: `doc_${Date.now()}`,
      filename: file.name,
      status: 'success',
    };
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: 'POST',
      headers: API_ACCESS_KEY ? { 'X-API-Key': API_ACCESS_KEY } : undefined,
      body: formData,
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const errorBody = await response.json();
        detail = errorBody?.detail ?? detail;
      } catch {
        // Ignore JSON parsing errors and keep status text fallback.
      }
      throw new Error(`Upload failed: ${detail}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
}

export async function deleteDocument(documentId: string): Promise<{ status: string }> {
  if (DEMO_MODE) {
    // Demo mode: simulate successful deletion
    await new Promise(resolve => setTimeout(resolve, 300));
    return { status: 'success' };
  }

  try {
    const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
      method: 'DELETE',
      headers: API_ACCESS_KEY ? { 'X-API-Key': API_ACCESS_KEY } : undefined,
    });

    if (response.status === 404) {
      // Stale client state: document may already be removed on backend.
      return { status: 'success' };
    }

    if (!response.ok) {
      throw new Error(`Delete failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Delete error:', error);
    throw error;
  }
}

export async function queryDocuments(payload: QueryRequestPayload): Promise<QueryResponse> {
  if (DEMO_MODE) {
    // Demo mode: return demo response after a realistic delay
    await new Promise(resolve => setTimeout(resolve, 1200));
    return DEMO_RESPONSES.default;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(API_ACCESS_KEY ? { 'X-API-Key': API_ACCESS_KEY } : {}),
      },
      body: JSON.stringify({
        question: payload.question,
        document_id: payload.documentId,
        history: payload.history,
      }),
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const errorBody = await response.json();
        detail = errorBody?.detail ?? detail;
      } catch {
        // Ignore JSON parsing errors and keep status text fallback.
      }
      throw new Error(`Query failed: ${detail}`);
    }

    const data: BackendQueryResponse = await response.json();
    return {
      answer: data.answer,
      sources: (data.chunks_used ?? []).map((chunk) => chunk.chunk_id),
    };
  } catch (error) {
    console.error('Query error:', error);
    throw error;
  }
}
