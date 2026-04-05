'use client';

import { useEffect, useRef } from 'react';
import Message from './Message';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  isStreaming?: boolean;
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
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-8">
      {messages.map((msg) => (
        <Message
          key={msg.id}
          role={msg.role}
          content={msg.content}
          sources={msg.sources}
          isStreaming={msg.isStreaming}
        />
      ))}
      <div ref={endRef} />
    </div>
  );
}
