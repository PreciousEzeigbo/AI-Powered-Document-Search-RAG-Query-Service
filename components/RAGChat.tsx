'use client';

import { useEffect, useState } from 'react';
import { StoredDocument, loadDocuments, addDocument, removeDocument } from '@/lib/storage';
import { uploadDocument, deleteDocument, queryDocuments } from '@/lib/api';
import { ChatMessage } from './ChatThread';
import ChatThread from './ChatThread';
import InputBar from './InputBar';
import DocumentPanel from './DocumentPanel';
import EmptyState from './EmptyState';

export default function RAGChat() {
  const [documents, setDocuments] = useState<StoredDocument[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadingId, setUploadingId] = useState<string>();
  const [deletingId, setDeletingId] = useState<string>();
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [error, setError] = useState<string>();

  // Load documents from localStorage on mount
  useEffect(() => {
    const savedDocs = loadDocuments();
    setDocuments(savedDocs);
  }, []);

  const handleUploadFiles = async (files: File[]) => {
    setIsUploading(true);
    setError(undefined);

    try {
      for (const file of files) {
        const response = await uploadDocument(file);
        const newDoc: StoredDocument = {
          document_id: response.document_id,
          filename: response.filename,
          file_type: file.type,
          uploaded_at: new Date().toISOString(),
        };
        const updated = addDocument(newDoc);
        setDocuments(updated);
      }
    } catch (err) {
      setError(`Upload failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDocument = async (documentId: string) => {
    setDeletingId(documentId);
    setError(undefined);

    try {
      await deleteDocument(documentId);
      const updated = removeDocument(documentId);
      setDocuments(updated);
    } catch (err) {
      setError(`Delete failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Delete error:', err);
    } finally {
      setDeletingId(undefined);
    }
  };

  const handleQuerySubmit = async (question: string) => {
    if (!question.trim()) return;

    setIsLoading(true);
    setError(undefined);

    // Add user message
    const userMessageId = Date.now().toString();
    const userMessage: ChatMessage = {
      id: userMessageId,
      role: 'user',
      content: question,
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await queryDocuments(question);

      // Add assistant message
      const assistantMessageId = (Date.now() + 1).toString();
      const assistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(`Query failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Query error:', err);

      // Add error message to chat
      const errorMessageId = (Date.now() + 1).toString();
      const errorMessage: ChatMessage = {
        id: errorMessageId,
        role: 'assistant',
        content: `Sorry, I couldn&apos;t process your question. Please try again.`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const hasDocuments = documents.length > 0;

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Document Panel - Only one instance, responsive via CSS */}
      <DocumentPanel
        documents={documents}
        onUpload={handleUploadFiles}
        onDelete={handleDeleteDocument}
        isOpen={isPanelOpen}
        onClose={() => setIsPanelOpen(false)}
        uploadingId={uploadingId}
        deletingId={deletingId}
        isUploading={isUploading}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header with Documents Toggle Button */}
        <div className="border-b border-border px-4 py-3 flex items-center justify-between">
          <h1 className="font-serif text-lg text-foreground">RAG Chat</h1>
          <button
            onClick={() => setIsPanelOpen(!isPanelOpen)}
            className="font-mono-ui text-sm text-accent hover:text-accent/80 transition-colors"
            title={isPanelOpen ? 'Hide documents' : 'Show documents'}
          >
            📁 Documents ({documents.length})
          </button>
        </div>

        {/* Chat Content or Empty State */}
        {!hasDocuments ? (
          <EmptyState />
        ) : (
          <>
            <ChatThread messages={messages} />
            {error && (
              <div className="mx-auto max-w-3xl w-full px-4 py-2 mb-2 bg-destructive/10 border border-destructive text-destructive text-xs rounded-sm">
                {error}
              </div>
            )}
          </>
        )}

        {/* Input Bar - Only show if documents exist */}
        {hasDocuments && (
          <InputBar
            onSubmit={handleQuerySubmit}
            isLoading={isLoading}
            disabled={!hasDocuments || isUploading}
          />
        )}
      </div>
    </div>
  );
}
