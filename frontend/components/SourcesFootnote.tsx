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
    <div className="mt-4 border-t border-border pt-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm font-mono-ui text-muted-foreground hover:text-foreground transition-colors"
      >
        <span className="text-xs">►</span>
        <span>Sources ({sources.length})</span>
      </button>

      <div
        className="overflow-hidden transition-all duration-200 ease-out"
        style={{
          maxHeight: isExpanded ? '500px' : '0px',
          opacity: isExpanded ? 1 : 0,
        }}
      >
        <ul className="mt-2 ml-5 space-y-1 text-xs font-mono-ui text-muted-foreground">
          {sources.map((source, idx) => (
            <li key={idx} className="list-disc">{source}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
