import * as React from 'react'

import { cn } from '@/lib/utils'

function Input({ className, type, ...props }: React.ComponentProps<'input'>) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        'file:text-zinc-900 placeholder:text-zinc-400 selection:bg-zinc-900 selection:text-zinc-50 dark:file:text-zinc-50 dark:placeholder:text-zinc-500 border-zinc-300 h-9 w-full min-w-0 rounded-md border bg-transparent px-3 py-1 font-mono text-sm text-zinc-900 shadow-xs transition-[color,box-shadow] outline-none file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 dark:text-zinc-50 dark:border-zinc-700 dark:bg-zinc-950 md:text-sm',
        'focus-visible:border-zinc-500 focus-visible:ring-0',
        'aria-invalid:border-zinc-500 aria-invalid:ring-0',
        className,
      )}
      {...props}
    />
  )
}

export { Input }
