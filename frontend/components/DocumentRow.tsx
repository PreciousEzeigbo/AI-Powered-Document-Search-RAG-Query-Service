'use client';

import { StoredDocument } from '@/lib/storage';

interface DocumentRowProps {
  document: StoredDocument;
  isActive?: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  isDeleting?: boolean;
}

export default function DocumentRow({
  document,
  isActive = false,
  onSelect,
  onDelete,
  isDeleting = false,
}: DocumentRowProps) {
  const uploadedDate = new Date(document.uploaded_at).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });

  return (
    <div
      className={`animate-fade-in flex items-start justify-between gap-3 px-3 py-3 transition-colors ${
        isActive
          ? 'bg-zinc-100 dark:bg-zinc-900'
          : 'hover:bg-zinc-100 dark:hover:bg-zinc-900/70'
      }`}
    >
      <button
        type="button"
        onClick={() => onSelect(document.document_id)}
        className="min-w-0 flex-1 text-left"
      >
        <p className="break-words font-mono text-sm text-zinc-900 dark:text-zinc-50">
          {document.filename}
        </p>
        <p className="mt-1 font-mono text-xs text-zinc-500 dark:text-zinc-400">
          {uploadedDate}
        </p>
      </button>
      <button
        onClick={() => onDelete(document.document_id)}
        disabled={isDeleting}
        className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-md border border-zinc-300 px-3 font-mono text-xs text-zinc-500 transition-colors hover:bg-zinc-200 hover:text-zinc-900 disabled:opacity-50 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-50"
      >
        {isDeleting ? 'removing...' : 'remove'}
      </button>
    </div>
  );
}
