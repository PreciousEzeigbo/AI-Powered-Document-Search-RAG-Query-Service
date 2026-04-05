// API client functions for RAG backend integration
// Falls back to demo mode when NEXT_PUBLIC_API_URL is not configured

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
const DEMO_MODE = !API_BASE_URL; // Enable demo mode when no API URL is configured

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
}

export interface QueryResponse {
  answer: string;
  sources: string[];
}

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
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
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
    });

    if (!response.ok) {
      throw new Error(`Delete failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Delete error:', error);
    throw error;
  }
}

export async function queryDocuments(question: string): Promise<QueryResponse> {
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
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      throw new Error(`Query failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Query error:', error);
    throw error;
  }
}
