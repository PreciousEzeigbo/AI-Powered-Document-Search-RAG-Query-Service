'use client';

interface EmptyStateProps {
  onOpenDocuments?: () => void;
}

export default function EmptyState({ onOpenDocuments }: EmptyStateProps) {
  return (
    <div className="flex h-full items-center justify-center py-8">
      <div className="w-full rounded-md border border-dashed border-zinc-300 bg-transparent px-8 py-8 text-center dark:border-zinc-700">
        <h2 className="font-mono text-sm text-zinc-900 dark:text-zinc-50">&gt; no document loaded_</h2>
        <p className="mx-auto mt-6 max-w-md font-mono text-sm leading-relaxed text-zinc-500 dark:text-zinc-400">
          upload a PDF, TXT or DOCX to start asking questions
        </p>

        <button
          type="button"
          onClick={onOpenDocuments}
          className="mt-6 inline-flex min-h-[44px] min-w-[220px] items-center justify-center rounded-md bg-zinc-900 px-4 py-2 font-mono text-sm text-zinc-50 transition-colors hover:bg-zinc-700"
        >
          [ upload a document ]
        </button>
      </div>
    </div>
  );
}
