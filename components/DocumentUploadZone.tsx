'use client';

import { useRef, useState } from 'react';

interface DocumentUploadZoneProps {
  onFiles: (files: File[]) => void;
  isLoading?: boolean;
}

export default function DocumentUploadZone({
  onFiles,
  isLoading = false,
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
      className={`border-2 border-dashed rounded-sm p-6 cursor-pointer transition-colors ${
        isDragOver
          ? 'border-accent bg-accent/5'
          : 'border-border hover:border-accent/50'
      } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
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
      <div className="text-center">
        <p className="font-mono-ui text-sm text-foreground mb-2">
          {isLoading ? 'Uploading...' : 'Drop files here or click to browse'}
        </p>
        <p className="font-mono-ui text-xs text-muted-foreground">
          Supported formats: PDF, TXT, DOCX
        </p>
      </div>
    </div>
  );
}
