import { cn } from '@/lib/utils'

function Kbd({ className, ...props }: React.ComponentProps<'kbd'>) {
  return (
    <kbd
      data-slot="kbd"
      className={cn(
        'pointer-events-none inline-flex h-5 min-w-5 w-fit items-center justify-center gap-1 rounded-sm border border-zinc-300 bg-zinc-100 px-1 font-mono text-[10px] font-medium text-zinc-600 select-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400',
        "[&_svg:not([class*='size-'])]:size-3",
        '[[data-slot=tooltip-content]_&]:bg-zinc-900/20 [[data-slot=tooltip-content]_&]:text-zinc-50 dark:[[data-slot=tooltip-content]_&]:bg-zinc-50/10',
        className,
      )}
      {...props}
    />
  )
}

function KbdGroup({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <kbd
      data-slot="kbd-group"
      className={cn('inline-flex items-center gap-1', className)}
      {...props}
    />
  )
}

export { Kbd, KbdGroup }
