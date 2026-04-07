'use client';

import { useEffect, useRef, useState } from 'react';
import { StoredDocument, addDocument, loadDocuments, removeDocument } from '@/lib/storage';
import { MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB, deleteDocument, queryDocuments, uploadDocument } from '@/lib/api';
import ChatThread, { ChatMessage } from './ChatThread';
import DocumentPanel from './DocumentPanel';
import EmptyState from './EmptyState';
import InputBar from './InputBar';

const SESSION_STORAGE_TOKEN_KEY = 'rag_session_token';
const CHAT_STORAGE_KEY = 'rag_session';
const SESSION_MAX_AGE_MS = 24 * 60 * 60 * 1000;

type StoredSession = {
  sessionToken: string;
  documentId: string;
  messages: ChatMessage[];
  savedAt: number;
};

const createId = () =>
  typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;

export default function RAGChat() {
  const [documents, setDocuments] = useState<StoredDocument[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string>();
  const [sessionToken, setSessionToken] = useState<string>('');
  const [restoreBanner, setRestoreBanner] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [deletingId, setDeletingId] = useState<string>();
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    phase: 'idle' | 'uploading' | 'success' | 'error';
    message?: string;
    completed?: number;
    total?: number;
  }>({ phase: 'idle' });

  const inputAnchorRef = useRef<HTMLDivElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const hasInitializedSessionRef = useRef(false);
  const previousDocumentIdRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    const docs = loadDocuments();
    setDocuments(docs);
    if (docs.length > 0) {
      setSelectedDocumentId(docs[docs.length - 1].document_id);
    }
  }, []);

  useEffect(() => {
    try {
      const existing = sessionStorage.getItem(SESSION_STORAGE_TOKEN_KEY);
      const token = existing || createId();
      if (!existing) {
        sessionStorage.setItem(SESSION_STORAGE_TOKEN_KEY, token);
      }
      setSessionToken(token);
    } catch {
      setSessionToken(createId());
    }
  }, []);

  useEffect(() => {
    if (!restoreBanner) return;
    const timer = window.setTimeout(() => setRestoreBanner(undefined), 4000);
    return () => window.clearTimeout(timer);
  }, [restoreBanner]);

  const scrollChatToBottom = (behavior: ScrollBehavior = 'auto') => {
    chatBottomRef.current?.scrollIntoView({ behavior, block: 'end' });
  };

  const makeSystemMessage = (text: string): ChatMessage => ({
    id: createId(),
    role: 'assistant',
    content: text,
    timestamp: Date.now(),
    isSystem: true,
  });

  const clearStoredSession = () => {
    try {
      localStorage.removeItem(CHAT_STORAGE_KEY);
    } catch {
      // Fail silently for storage issues.
    }
  };

  useEffect(() => {
    if (!sessionToken) return;

    const currentDocumentId = selectedDocumentId;
    const previousDocumentId = previousDocumentIdRef.current;

    if (!currentDocumentId) {
      setMessages([]);
      clearStoredSession();
      previousDocumentIdRef.current = undefined;
      hasInitializedSessionRef.current = true;
      return;
    }

    if (!hasInitializedSessionRef.current) {
      hasInitializedSessionRef.current = true;
      previousDocumentIdRef.current = currentDocumentId;

      let restored = false;
      try {
        const raw = localStorage.getItem(CHAT_STORAGE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw) as StoredSession;
          const age = Date.now() - (parsed.savedAt ?? 0);
          if (
            parsed.sessionToken === sessionToken &&
            parsed.documentId === currentDocumentId &&
            Array.isArray(parsed.messages) &&
            age <= SESSION_MAX_AGE_MS
          ) {
            const hydratedMessages = parsed.messages.map((msg) => ({
              ...msg,
              timestamp: msg.timestamp ?? Date.now(),
            }));
            setMessages(hydratedMessages);
            setRestoreBanner('session restored - open the drawer to switch documents.');
            setTimeout(() => scrollChatToBottom('auto'), 0);
            restored = true;
          } else {
            localStorage.removeItem(CHAT_STORAGE_KEY);
          }
        }
      } catch {
        // Fail silently for storage issues.
      }

      if (!restored) {
        setMessages([]);
      }
      return;
    }

    if (previousDocumentId !== currentDocumentId) {
      previousDocumentIdRef.current = currentDocumentId;
      const selectedDoc = documents.find((doc) => doc.document_id === currentDocumentId);
      const filename = selectedDoc?.filename ?? 'document';
      const switchMessage = makeSystemMessage(`switched to ${filename} - starting a new session.`);
      setMessages([switchMessage]);
      clearStoredSession();
      setRestoreBanner(undefined);
      setTimeout(() => scrollChatToBottom('smooth'), 0);
    }
  }, [selectedDocumentId, sessionToken, documents]);

  useEffect(() => {
    if (!sessionToken || !selectedDocumentId) return;
    try {
      const payload: StoredSession = {
        sessionToken,
        documentId: selectedDocumentId,
        messages,
        savedAt: Date.now(),
      };
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Fail silently for storage issues.
    }
  }, [messages, selectedDocumentId, sessionToken]);

  const toFriendlyError = (fallback: string, err: unknown): string => {
    if (!(err instanceof Error)) return fallback;

    const clean = err.message
      .replace(/^Upload failed:\s*/i, '')
      .replace(/^Query failed:\s*/i, '')
      .replace(/^Delete failed:\s*/i, '')
      .trim();

    if (!clean) return fallback;
    if (/traceback|exception|stack|internal server error/i.test(clean)) return fallback;
    return clean;
  };

  const handleUploadFiles = async (files: File[]) => {
    if (files.length === 0) return;

    const oversizedFile = files.find((file) => file.size > MAX_UPLOAD_SIZE_BYTES);
    if (oversizedFile) {
      setUploadStatus({
        phase: 'error',
        message: `${oversizedFile.name} is too large. Maximum file size is ${MAX_UPLOAD_SIZE_MB}MB.`,
      });
      return;
    }

    setIsUploading(true);
    setUploadStatus({
      phase: 'uploading',
      message: `Uploading ${files.length} file${files.length > 1 ? 's' : ''}...`,
      completed: 0,
      total: files.length,
    });

    try {
      let completed = 0;

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
        if (!selectedDocumentId) {
          setSelectedDocumentId(newDoc.document_id);
        }

        completed += 1;
        setUploadStatus({
          phase: 'uploading',
          message: `Uploaded ${completed} of ${files.length}`,
          completed,
          total: files.length,
        });
      }

      setUploadStatus({
        phase: 'success',
        message: `Upload complete. Ask a question about your document${files.length > 1 ? 's' : ''}.`,
        completed: files.length,
        total: files.length,
      });

      setIsPanelOpen(false);

      setTimeout(() => {
        inputAnchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        chatBottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }, 120);
    } catch (err) {
      const message = toFriendlyError('Upload failed. Please try again.', err);
      setUploadStatus({ phase: 'error', message });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteDocument = async (documentId: string) => {
    setDeletingId(documentId);

    try {
      await deleteDocument(documentId);
      const updated = removeDocument(documentId);
      setDocuments(updated);

      if (selectedDocumentId === documentId) {
        const fallbackDoc = updated[updated.length - 1];
        setSelectedDocumentId(fallbackDoc?.document_id);
      }

      if (updated.length === 0) {
        setMessages([]);
        clearStoredSession();
      }
    } finally {
      setDeletingId(undefined);
    }
  };

  const handleQuerySubmit = async (question: string) => {
    if (!question.trim() || !selectedDocumentId) return;

    setIsLoading(true);

    const userMessageId = createId();
    const pendingAssistantId = `${userMessageId}_assistant_pending`;
    const now = Date.now();

    const userMessage: ChatMessage = {
      id: userMessageId,
      role: 'user',
      content: question,
      timestamp: now,
    };

    const pendingAssistant: ChatMessage = {
      id: pendingAssistantId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      timestamp: now,
    };

    const history = messages
      .filter((msg) => !msg.isSystem)
      .slice(-4)
      .map((msg) => ({ role: msg.role, content: msg.content }));

    setMessages((prev) => [...prev, userMessage, pendingAssistant]);

    try {
      const response = await queryDocuments({
        question,
        documentId: selectedDocumentId,
        history,
      });
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === pendingAssistantId
            ? {
                ...msg,
                content: response.answer,
                sources: response.sources,
                isStreaming: false,
                isError: false,
                timestamp: Date.now(),
              }
            : msg
        )
      );
    } catch (err) {
      const message = toFriendlyError(
        'I could not answer that right now. Try again in a moment.',
        err
      );

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === pendingAssistantId
            ? {
                ...msg,
                content: message,
                isStreaming: false,
                isError: true,
                timestamp: Date.now(),
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const hasDocuments = documents.length > 0;
  const activeDocument = selectedDocumentId
    ? documents.find((doc) => doc.document_id === selectedDocumentId)
    : undefined;
  const hasActiveDocument = Boolean(activeDocument);

  return (
    <div className="h-dvh w-full overflow-hidden bg-background text-foreground">
      <DocumentPanel
        documents={documents}
        selectedDocumentId={selectedDocumentId}
        onSelect={(documentId) => {
          setSelectedDocumentId(documentId);
          setIsPanelOpen(false);
        }}
        onUpload={handleUploadFiles}
        onDelete={handleDeleteDocument}
        isOpen={isPanelOpen}
        onClose={() => setIsPanelOpen(false)}
        deletingId={deletingId}
        isUploading={isUploading}
        uploadStatus={uploadStatus}
      />

      <header className="fixed inset-x-0 top-0 z-20 border-b border-zinc-300 bg-background/95 backdrop-blur dark:border-zinc-800">
        <div className="mx-auto grid h-14 w-full max-w-2xl grid-cols-[3rem_1fr_3rem] items-center px-4">
          <button
            onClick={() => setIsPanelOpen(true)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-zinc-300 font-mono text-sm text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-900"
            aria-label="Open documents drawer"
          >
            ☰
          </button>
          <h1 className="justify-self-center text-center font-mono text-sm tracking-[0.3em] text-zinc-900 dark:text-zinc-50">
            DOCUMENT Q&amp;A<span className="animate-blink">_</span>
          </h1>
          <div aria-hidden="true" />
        </div>
      </header>

      <main className="mx-auto h-dvh w-full max-w-2xl px-4 pt-14">
        <section className="flex h-full flex-col">
          {restoreBanner && (
            <div className="mb-2 rounded-md border border-zinc-300 bg-zinc-100 px-3 py-2 font-mono text-xs text-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400">
              {restoreBanner}
            </div>
          )}
          <div className="min-h-0 flex-1 overflow-y-auto pb-40">
            {!hasDocuments ? (
              <EmptyState onOpenDocuments={() => setIsPanelOpen(true)} />
            ) : (
              <ChatThread messages={messages} />
            )}
            <div ref={chatBottomRef} />
          </div>

          <div
            ref={inputAnchorRef}
            className="fixed inset-x-0 bottom-0 z-20 border-t border-zinc-300 bg-background/95 backdrop-blur dark:border-zinc-800"
          >
            <div className="mx-auto w-full max-w-2xl px-4 py-3">
              {activeDocument && (
                <div className="mb-3 flex min-h-[44px] items-center justify-between gap-3 rounded-md border border-zinc-300 bg-transparent px-4 py-2 dark:border-zinc-700">
                  <p className="min-w-0 flex-1 break-all font-mono text-xs text-zinc-500 dark:text-zinc-400">
                    [{activeDocument.filename}]
                  </p>
                  <button
                    onClick={() => setIsPanelOpen(true)}
                    className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-md border border-zinc-300 px-3 font-mono text-xs text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-900"
                  >
                    change
                  </button>
                </div>
              )}

              <InputBar
                onSubmit={handleQuerySubmit}
                isLoading={isLoading}
                disabled={!hasActiveDocument || isUploading}
                disabledMessage="upload a document first"
              />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
