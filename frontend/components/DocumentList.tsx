'use client';

import { StoredDocument } from '@/lib/storage';
import DocumentRow from './DocumentRow';

interface DocumentListProps {
  documents: StoredDocument[];
  selectedDocumentId?: string;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  deletingId?: string;
}

export default function DocumentList({
  documents,
  selectedDocumentId,
  onSelect,
  onDelete,
  deletingId,
}: DocumentListProps) {
  if (documents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-md border border-dashed border-zinc-300 px-4 py-8 text-center dark:border-zinc-700">
        <p className="font-mono text-sm text-zinc-500 dark:text-zinc-400">
          no documents uploaded yet
        </p>
      </div>
    );
  }

  return (
    <div className="divide-y divide-zinc-300 rounded-md border border-zinc-300 dark:divide-zinc-800 dark:border-zinc-800">
      {documents.map((doc) => (
        <DocumentRow
          key={doc.document_id}
          document={doc}
          isActive={selectedDocumentId === doc.document_id}
          onSelect={onSelect}
          onDelete={onDelete}
          isDeleting={deletingId === doc.document_id}
        />
      ))}
    </div>
  );
}
