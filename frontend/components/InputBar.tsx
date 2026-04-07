'use client';

import { useState, useRef } from 'react';

interface InputBarProps {
  onSubmit: (text: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
  disabledMessage?: string;
}

export default function InputBar({
  onSubmit,
  isLoading = false,
  disabled = false,
  disabledMessage,
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
    <div className="w-full">
      {disabled && disabledMessage && (
        <p className="mb-2 font-mono text-sm text-zinc-500 dark:text-zinc-400">{disabledMessage}</p>
      )}
      <div className="flex w-full items-end gap-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="type your question..."
          disabled={disabled || isLoading}
          rows={1}
          className="min-h-[44px] flex-1 resize-none rounded-md border border-zinc-300 bg-transparent px-3 py-2 font-mono text-sm text-zinc-900 outline-none placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-0 disabled:cursor-not-allowed disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-50 dark:placeholder:text-zinc-500"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || isLoading || !input.trim()}
          className="inline-flex min-h-[44px] min-w-[72px] items-center justify-center whitespace-nowrap rounded-md border border-zinc-300 px-4 py-2 font-mono text-sm text-zinc-900 transition-colors hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-50 dark:hover:bg-zinc-800"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
