'use client';

import { useEffect, useRef } from 'react';
import { StoredDocument } from '@/lib/storage';
import DocumentUploadZone from './DocumentUploadZone';
import DocumentList from './DocumentList';

interface DocumentPanelProps {
  documents: StoredDocument[];
  selectedDocumentId?: string;
  onSelect: (id: string) => void;
  onUpload: (files: File[]) => void;
  onDelete: (id: string) => void;
  isOpen: boolean;
  onClose: () => void;
  deletingId?: string;
  isUploading?: boolean;
  uploadStatus?: {
    phase: 'idle' | 'uploading' | 'success' | 'error';
    message?: string;
    completed?: number;
    total?: number;
  };
}

export default function DocumentPanel({
  documents,
  selectedDocumentId,
  onSelect,
  onUpload,
  onDelete,
  isOpen,
  onClose,
  deletingId,
  isUploading = false,
  uploadStatus,
}: DocumentPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  // Close panel on escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  return (
    <>
      {isOpen && (
        <button
          type="button"
          className="fixed inset-0 z-30 cursor-default bg-black/50"
          aria-label="Close documents drawer"
          onClick={onClose}
        />
      )}

      <aside
        ref={panelRef}
        className={`fixed inset-y-0 left-0 z-40 flex h-full w-full flex-col border-r border-zinc-300 bg-zinc-50 transition-transform duration-300 ease-out dark:border-zinc-800 dark:bg-zinc-950 sm:w-[22rem] ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        aria-hidden={!isOpen}
      >
        <div className="flex items-center justify-between border-b border-zinc-300 px-4 py-3 dark:border-zinc-800">
          <h2 className="font-mono text-sm text-zinc-900 dark:text-zinc-50">
            &gt; documents_
          </h2>
          <button
            onClick={onClose}
            className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-md border border-zinc-300 font-mono text-sm text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-900"
            aria-label="Close documents drawer"
          >
            x
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4 pb-6">
          <DocumentUploadZone
            onFiles={onUpload}
            isLoading={isUploading}
            status={uploadStatus}
          />

          <div className="mt-6">
            <p className="mb-3 font-mono text-xs text-zinc-500 dark:text-zinc-400">
              uploaded ({documents.length})
            </p>
            <DocumentList
              documents={documents}
              selectedDocumentId={selectedDocumentId}
              onSelect={onSelect}
              onDelete={onDelete}
              deletingId={deletingId}
            />
          </div>
        </div>
      </aside>
    </>
  );
}
