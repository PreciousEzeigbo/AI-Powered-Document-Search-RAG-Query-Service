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
  onToggle: () => void;
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
  onToggle,
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

  // Mobile: bottom sheet (slides up when open)
  // Desktop: left sidebar (always visible, toggleable)
  return (
    <>
      {/* Mobile Backdrop - only shows on mobile when panel is open */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-30 md:hidden"
          onClick={onClose}
        />
      )}

      <div className="flex">
        {/* Panel Container - Desktop sidebar + Mobile bottom sheet */}
        <div
          ref={panelRef}
          className={`
            flex flex-col bg-background border-border
            transition-all duration-250 ease-out
            
            /* Mobile: bottom sheet behavior */
            fixed md:static
            bottom-0 left-0 right-0 max-h-[80vh] z-40
            md:w-80 md:border-r md:max-h-full
            
            ${isOpen 
              ? 'translate-y-0 md:translate-x-0' 
              : 'translate-y-full md:translate-x-0 md:w-0'
            }
            
            ${!isOpen ? 'md:hidden' : 'md:flex'}
          `}
        >
        {/* Mobile Header - close button for bottom sheet */}
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

        {/* Desktop Header - always visible */}
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

        {/* Desktop Toggle Button - Only visible on desktop */}
        <button
          onClick={onToggle}
          className="hidden md:flex items-center justify-center w-8 border-l border-border bg-background hover:bg-secondary transition-colors"
          title={isOpen ? 'Collapse documents' : 'Expand documents'}
        >
          <span className="font-mono-ui text-sm text-accent">
            {isOpen ? '<<' : '>>'}
          </span>
        </button>
      </div>
    </>
  );
}
