'use client';

import { useEffect, useRef } from 'react';
import Message from './Message';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  sources?: string[];
  isStreaming?: boolean;
  isError?: boolean;
  isSystem?: boolean;
}

interface ChatThreadProps {
  messages: ChatMessage[];
}

export default function ChatThread({ messages }: ChatThreadProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="w-full overflow-x-hidden">
      <div className="w-full space-y-4 py-4">
        {messages.map((msg) => (
          <Message
            key={msg.id}
            role={msg.role}
            content={msg.content}
            sources={msg.sources}
            isStreaming={msg.isStreaming}
            isError={msg.isError}
            isSystem={msg.isSystem}
          />
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}
