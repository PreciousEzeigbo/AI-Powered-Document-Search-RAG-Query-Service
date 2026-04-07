'use client';

import { useRef, useState } from 'react';
import { MAX_UPLOAD_SIZE_MB } from '@/lib/api';

interface DocumentUploadZoneProps {
  onFiles: (files: File[]) => void;
  isLoading?: boolean;
  status?: {
    phase: 'idle' | 'uploading' | 'success' | 'error';
    message?: string;
    completed?: number;
    total?: number;
  };
}

export default function DocumentUploadZone({
  onFiles,
  isLoading = false,
  status,
}: DocumentUploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onFiles(files);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.currentTarget.files || []);
    if (files.length > 0) {
      onFiles(files);
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`min-h-[160px] cursor-pointer rounded-md border-2 border-dashed border-zinc-300 p-6 transition-colors hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800/40 ${
        isDragOver ? 'bg-zinc-100 dark:bg-zinc-800/60' : ''
      } ${isLoading ? 'cursor-not-allowed opacity-50' : ''}`}
      onClick={() => !isLoading && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        onChange={handleInputChange}
        disabled={isLoading}
        className="hidden"
        accept=".pdf,.txt,.docx,.doc"
      />
      <div className="flex h-full flex-col items-center justify-center text-center">
        <p className="font-mono text-sm text-zinc-900 dark:text-zinc-50">
          {isLoading ? 'uploading document...' : 'drop files here or click to browse_'}
        </p>
        <p className="mt-2 font-mono text-xs text-zinc-500 dark:text-zinc-400">
          supported formats: pdf, txt, docx | max size: {MAX_UPLOAD_SIZE_MB} MB
        </p>

        {status?.phase === 'uploading' && (
          <div className="mt-3 w-full max-w-xs">
            <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
              <div
                className="h-full bg-zinc-900 transition-all dark:bg-zinc-50"
                style={{
                  width: `${status.total ? Math.max(8, Math.round(((status.completed ?? 0) / status.total) * 100)) : 20}%`,
                }}
              />
            </div>
            <p className="mt-2 font-mono text-xs text-zinc-500 dark:text-zinc-400">{status.message}</p>
          </div>
        )}

        {status?.phase === 'success' && status.message && (
          <p className="mt-3 w-full rounded-md border border-zinc-300 bg-zinc-100 px-3 py-2 font-mono text-xs text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
            {status.message}
          </p>
        )}

        {status?.phase === 'error' && status.message && (
          <p className="mt-3 w-full rounded-md border border-zinc-300 bg-zinc-100 px-3 py-2 font-mono text-xs text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
            {status.message}
          </p>
        )}
      </div>
    </div>
  );
}
