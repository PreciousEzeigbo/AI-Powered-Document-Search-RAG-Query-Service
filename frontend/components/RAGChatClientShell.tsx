'use client';

import dynamic from 'next/dynamic';

const RAGChat = dynamic(() => import('@/components/RAGChat'), {
  ssr: false,
  loading: () => (
    <main className="flex h-dvh w-full flex-col">
      <div className="flex h-14 items-center justify-center border-b border-zinc-300 px-4 dark:border-zinc-800">
        <div className="grid w-full max-w-2xl grid-cols-[3rem_1fr_3rem] items-center">
          <div className="h-8 w-8 rounded-md border border-zinc-300 dark:border-zinc-700" />
          <span className="justify-self-center font-mono text-sm tracking-[0.3em] text-zinc-900 dark:text-zinc-50">
            DOCUMENT Q&amp;A<span className="animate-blink">_</span>
          </span>
          <div aria-hidden="true" />
        </div>
      </div>
      <div className="flex flex-1 items-center justify-center px-4">
        <p className="font-mono text-sm text-zinc-500 dark:text-zinc-400">loading workspace...</p>
      </div>
    </main>
  ),
});

export default function RAGChatClientShell() {
  return <RAGChat />;
}