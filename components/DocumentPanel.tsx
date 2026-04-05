'use client';

import { useEffect, useRef } from 'react';
import { StoredDocument } from '@/lib/storage';
import DocumentUploadZone from './DocumentUploadZone';
import DocumentList from './DocumentList';

interface DocumentPanelProps {
  documents: StoredDocument[];
  onUpload: (files: File[]) => void;
  onDelete: (id: string) => void;
  isOpen: boolean;
  onClose: () => void;
  uploadingId?: string;
  deletingId?: string;
  isUploading?: boolean;
}

export default function DocumentPanel({
  documents,
  onUpload,
  onDelete,
  isOpen,
  onClose,
  uploadingId,
  deletingId,
  isUploading = false,
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

  // Mobile: bottom sheet
  // Desktop: left sidebar
  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-30 md:hidden"
          onClick={onClose}
        />
      )}

      {/* Panel Container */}
      <div
        ref={panelRef}
        className={`fixed md:static flex flex-col bg-background border-border z-40
          md:w-80 md:border-r md:translate-x-0
          bottom-0 left-0 right-0 max-h-[80vh] md:max-h-full
          transition-transform duration-250 ease-out
          ${isOpen ? 'translate-y-0' : 'translate-y-full md:translate-y-0'}`}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border md:hidden">
          <h2 className="font-mono-ui text-sm font-semibold text-foreground">
            Documents
          </h2>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            ✕
          </button>
        </div>

        <h2 className="hidden md:block font-mono-ui text-sm font-semibold text-foreground px-4 py-3 border-b border-border">
          Documents
        </h2>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          <DocumentUploadZone onFiles={onUpload} isLoading={isUploading} />

          <div className="mt-6">
            <p className="font-mono-ui text-xs text-muted-foreground mb-3 uppercase tracking-wider">
              Uploaded ({documents.length})
            </p>
            <DocumentList
              documents={documents}
              onDelete={onDelete}
              deletingId={deletingId}
            />
          </div>
        </div>
      </div>
    </>
  );
}
