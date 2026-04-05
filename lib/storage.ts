// localStorage helper functions for RAG document management

export interface StoredDocument {
  document_id: string;
  filename: string;
  file_type: string;
  uploaded_at: string;
}

const STORAGE_KEY = 'rag_documents';

export function loadDocuments(): StoredDocument[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.error('Failed to load documents from localStorage:', error);
    return [];
  }
}

export function saveDocuments(documents: StoredDocument[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(documents));
  } catch (error) {
    console.error('Failed to save documents to localStorage:', error);
  }
}

export function addDocument(doc: StoredDocument): StoredDocument[] {
  const docs = loadDocuments();
  docs.push(doc);
  saveDocuments(docs);
  return docs;
}

export function removeDocument(documentId: string): StoredDocument[] {
  const docs = loadDocuments();
  const filtered = docs.filter(doc => doc.document_id !== documentId);
  saveDocuments(filtered);
  return filtered;
}

export function clearAllDocuments(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.error('Failed to clear documents:', error);
  }
}
