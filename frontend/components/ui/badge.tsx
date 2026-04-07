import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center justify-center rounded-md border px-2 py-0.5 text-xs font-medium w-fit whitespace-nowrap shrink-0 [&>svg]:size-3 gap-1 [&>svg]:pointer-events-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive transition-[color,box-shadow] overflow-hidden',
  {
    variants: {
      variant: {
        default:
          'border-zinc-300 bg-zinc-100 text-zinc-900 [a&]:hover:bg-zinc-200 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50 dark:[a&]:hover:bg-zinc-800',
        secondary:
          'border-zinc-300 bg-transparent text-zinc-700 [a&]:hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:[a&]:hover:bg-zinc-900',
        destructive:
          'border-zinc-700 bg-zinc-900 text-zinc-50 [a&]:hover:bg-zinc-800 dark:border-zinc-300 dark:bg-zinc-200 dark:text-zinc-950 dark:[a&]:hover:bg-zinc-300',
        outline:
          'border-zinc-300 text-zinc-700 [a&]:hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:[a&]:hover:bg-zinc-900',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

function Badge({
  className,
  variant,
  asChild = false,
  ...props
}: React.ComponentProps<'span'> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot : 'span'

  return (
    <Comp
      data-slot="badge"
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }
