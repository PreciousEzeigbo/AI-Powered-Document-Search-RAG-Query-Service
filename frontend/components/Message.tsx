'use client';

import StreamingText from './StreamingText';
import SourcesFootnote from './SourcesFootnote';
import TypingIndicator from './TypingIndicator';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  isStreaming?: boolean;
}

export default function Message({
  role,
  content,
  sources,
  isStreaming = false,
}: MessageProps) {
  const isUser = role === 'user';

  return (
    <div
      className={`flex gap-4 animate-fade-in ${
        isUser ? 'justify-end' : 'justify-start'
      }`}
    >
      <div
        className={`max-w-[80%] ${
          isUser
            ? 'bg-accent text-accent-foreground rounded-none'
            : 'bg-transparent text-foreground'
        }`}
      >
        {isUser ? (
          <div className="font-mono-ui text-sm leading-relaxed break-words">
            {content}
          </div>
        ) : (
          <div className="font-serif text-base leading-relaxed break-words">
            {isStreaming ? (
              <StreamingText text={content} charDelay={30} />
            ) : (
              content
            )}
          </div>
        )}
        {!isUser && sources && <SourcesFootnote sources={sources} />}
      </div>

      {isStreaming && isUser === false && (
        <div className="absolute right-4">
          <TypingIndicator />
        </div>
      )}
    </div>
  );
}
