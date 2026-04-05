'use client';

import { StoredDocument } from '@/lib/storage';

interface DocumentRowProps {
  document: StoredDocument;
  onDelete: (id: string) => void;
  isDeleting?: boolean;
}

export default function DocumentRow({
  document,
  onDelete,
  isDeleting = false,
}: DocumentRowProps) {
  const uploadedDate = new Date(document.uploaded_at).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });

  return (
    <div className="flex items-center justify-between gap-3 py-3 px-3 border-b border-border hover:bg-muted transition-colors animate-fade-in">
      <div className="flex-1 min-w-0">
        <p className="font-mono-ui text-sm text-foreground truncate">
          {document.filename}
        </p>
        <p className="font-mono-ui text-xs text-muted-foreground mt-1">
          {uploadedDate}
        </p>
      </div>
      <button
        onClick={() => onDelete(document.document_id)}
        disabled={isDeleting}
        className="font-mono-ui text-xs text-muted-foreground hover:text-destructive transition-colors disabled:opacity-50"
      >
        {isDeleting ? 'Removing...' : 'Remove'}
      </button>
    </div>
  );
}
