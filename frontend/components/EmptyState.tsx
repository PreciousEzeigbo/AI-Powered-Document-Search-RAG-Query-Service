'use client';

export default function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 text-center">
      <h1 className="font-serif text-2xl text-foreground mb-4">
        Upload documents to get started
      </h1>
      <p className="font-mono-ui text-sm text-muted-foreground max-w-sm mb-8">
        Add PDFs, text files, or documents to your library, then ask questions about their content.
      </p>
      <div className="text-xs font-mono-ui text-muted-foreground space-y-1">
        <p>✓ Drag and drop files or click to browse</p>
        <p>✓ Ask natural language questions</p>
        <p>✓ Get answers with source citations</p>
      </div>
    </div>
  );
}
