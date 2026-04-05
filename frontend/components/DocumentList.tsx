'use client';

import { StoredDocument } from '@/lib/storage';
import DocumentRow from './DocumentRow';

interface DocumentListProps {
  documents: StoredDocument[];
  onDelete: (id: string) => void;
  deletingId?: string;
}

export default function DocumentList({
  documents,
  onDelete,
  deletingId,
}: DocumentListProps) {
  if (documents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <p className="font-mono-ui text-sm text-muted-foreground">
          No documents uploaded yet
        </p>
      </div>
    );
  }

  return (
    <div className="divide-y divide-border">
      {documents.map((doc) => (
        <DocumentRow
          key={doc.document_id}
          document={doc}
          onDelete={onDelete}
          isDeleting={deletingId === doc.document_id}
        />
      ))}
    </div>
  );
}
