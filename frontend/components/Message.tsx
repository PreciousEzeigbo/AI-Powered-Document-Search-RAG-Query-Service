'use client';

import StreamingText from './StreamingText';
import SourcesFootnote from './SourcesFootnote';
import TypingIndicator from './TypingIndicator';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  isStreaming?: boolean;
  isError?: boolean;
  isSystem?: boolean;
}

export default function Message({
  role,
  content,
  sources,
  isStreaming = false,
  isError = false,
  isSystem = false,
}: MessageProps) {
  const isUser = role === 'user';

  if (isSystem) {
    return (
      <div className="flex w-full justify-center">
        <div className="max-w-[90%] rounded-md border border-zinc-300 bg-zinc-100 px-4 py-2 text-center font-mono text-xs text-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className={`flex w-full animate-fade-in ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`w-full max-w-full rounded-md border px-4 py-4 ${
          isUser
            ? 'max-w-[90%] border-zinc-900 bg-zinc-900 text-zinc-50 dark:border-zinc-200 dark:bg-zinc-200 dark:text-zinc-950'
            : isError
              ? 'border-zinc-300 bg-zinc-100 text-zinc-900 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100'
              : 'border-zinc-300 bg-zinc-50 text-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100'
        }`}
      >
        {isUser ? (
          <div className="font-mono text-sm leading-relaxed break-words">
            {content}
          </div>
        ) : (
          <div className="font-mono text-sm leading-relaxed break-words">
            {isStreaming ? (
              <StreamingText text={content} charDelay={30} />
            ) : (
              content
            )}
          </div>
        )}
        {!isUser && isStreaming && (
          <div className="mt-2">
            <TypingIndicator />
          </div>
        )}
        {!isUser && sources && <SourcesFootnote sources={sources} />}
      </div>
    </div>
  );
}
