'use client';

import { useState } from 'react';

interface SourcesFootnoteProps {
  sources: string[];
}

export default function SourcesFootnote({ sources }: SourcesFootnoteProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-4 border-t border-zinc-300 pt-3 dark:border-zinc-700">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="inline-flex min-h-[44px] min-w-[44px] items-center gap-2 rounded-md px-2 font-mono text-xs text-zinc-500 transition-colors hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
      >
        <span className="text-[10px]">{isExpanded ? 'v' : '>'}</span>
        <span>sources ({sources.length})</span>
      </button>

      <div
        className="overflow-hidden transition-all duration-200 ease-out"
        style={{
          maxHeight: isExpanded ? '500px' : '0px',
          opacity: isExpanded ? 1 : 0,
        }}
      >
        <ul className="mt-2 ml-5 space-y-1 font-mono text-xs text-zinc-500 dark:text-zinc-400">
          {sources.map((source, idx) => (
            <li key={idx} className="list-disc">{source}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
