'use client';

import { useState, useRef } from 'react';

interface InputBarProps {
  onSubmit: (text: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export default function InputBar({
  onSubmit,
  isLoading = false,
  disabled = false,
}: InputBarProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    if (input.trim() && !isLoading && !disabled) {
      onSubmit(input.trim());
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(
        textareaRef.current.scrollHeight,
        120
      ) + 'px';
    }
  };

  return (
    <div className="border-t border-border bg-background p-4">
      <div className="mx-auto max-w-3xl flex gap-3">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents..."
          disabled={disabled || isLoading}
          rows={1}
          className="flex-1 font-mono-ui text-sm bg-input text-foreground border border-border rounded-sm px-3 py-2 resize-none outline-none focus:ring-1 focus:ring-accent disabled:opacity-50"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || isLoading || !input.trim()}
          className="bg-accent text-accent-foreground font-mono-ui text-sm px-4 py-2 rounded-sm hover:opacity-90 disabled:opacity-50 transition-opacity whitespace-nowrap"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
