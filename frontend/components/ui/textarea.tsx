import * as React from 'react'

import { cn } from '@/lib/utils'

function Textarea({ className, ...props }: React.ComponentProps<'textarea'>) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        'border-zinc-300 placeholder:text-zinc-400 focus-visible:border-zinc-500 focus-visible:ring-0 aria-invalid:border-zinc-500 aria-invalid:ring-0 dark:border-zinc-700 dark:bg-zinc-950 flex field-sizing-content min-h-16 w-full rounded-md border bg-transparent px-3 py-2 font-mono text-sm text-zinc-900 shadow-xs transition-[color,box-shadow] outline-none disabled:cursor-not-allowed disabled:opacity-50 dark:text-zinc-50 md:text-sm',
        className,
      )}
      {...props}
    />
  )
}

export { Textarea }
