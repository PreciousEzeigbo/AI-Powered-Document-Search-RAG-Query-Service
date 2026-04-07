'use client'

import * as React from 'react'
import * as TogglePrimitive from '@radix-ui/react-toggle'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const toggleVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-md border border-zinc-300 bg-transparent font-mono text-xs font-medium text-zinc-700 hover:bg-zinc-100 hover:text-zinc-900 disabled:pointer-events-none disabled:opacity-50 data-[state=on]:bg-zinc-900 data-[state=on]:text-zinc-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 [&_svg]:shrink-0 focus-visible:outline-none transition-[color,box-shadow] aria-invalid:border-zinc-500 whitespace-nowrap dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-900 dark:hover:text-zinc-50 dark:data-[state=on]:bg-zinc-200 dark:data-[state=on]:text-zinc-950",
  {
    variants: {
      variant: {
        default: 'bg-transparent',
        outline:
          'border border-zinc-300 bg-transparent shadow-none hover:bg-zinc-100 hover:text-zinc-900 dark:border-zinc-700 dark:hover:bg-zinc-900 dark:hover:text-zinc-50',
      },
      size: {
        default: 'h-9 px-2 min-w-9',
        sm: 'h-8 px-1.5 min-w-8',
        lg: 'h-10 px-2.5 min-w-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  },
)

function Toggle({
  className,
  variant,
  size,
  ...props
}: React.ComponentProps<typeof TogglePrimitive.Root> &
  VariantProps<typeof toggleVariants>) {
  return (
    <TogglePrimitive.Root
      data-slot="toggle"
      className={cn(toggleVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Toggle, toggleVariants }
