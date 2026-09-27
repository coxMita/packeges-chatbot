/** A few line icons in the SF Symbols spirit, drawn at 24px and scaled. */
import type { ReactNode } from 'react'

interface P { size?: number }

const svg = (size: number, children: ReactNode, sw = 1.8) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor"
       strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {children}
  </svg>
)

export const Shield = ({ size = 16 }: P) =>
  svg(size, <><path d="M12 3l7.5 3v5.5c0 4.6-3.2 8.3-7.5 9.5-4.3-1.2-7.5-4.9-7.5-9.5V6z" /><path d="M9 12l2.2 2.2L15.5 10" /></>)

export const Chevron = ({ size = 14, open = false }: P & { open?: boolean }) => (
  <span className={`chevron${open ? ' open' : ''}`}>
    {svg(size, <path d="M9 6l6 6-6 6" />, 2.2)}
  </span>
)

export const Code = ({ size = 15 }: P) =>
  svg(size, <><path d="M8 8l-4 4 4 4" /><path d="M16 8l4 4-4 4" /><path d="M13.5 5l-3 14" /></>)

export const Chart = ({ size = 15 }: P) =>
  svg(size, <><path d="M4 20h16" /><path d="M7 16v-5" /><path d="M12 16V7" /><path d="M17 16v-3" /></>)

export const Nodes = ({ size = 15 }: P) =>
  svg(size, <><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="8" r="2.5" /><circle cx="10" cy="18" r="2.5" /><path d="M8.2 7.2l7.4.6M7 8.3l2.2 7.3M16.3 10l-4.6 6.3" /></>)

export const Sparkle = ({ size = 13 }: P) =>
  svg(size, <path d="M12 3c.6 4.2 2.8 6.4 7 7-4.2.6-6.4 2.8-7 7-.6-4.2-2.8-6.4-7-7 4.2-.6 6.4-2.8 7-7z" />)

export const Compose = ({ size = 15 }: P) =>
  svg(size, <><path d="M12 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-6" /><path d="M17.5 3.5l3 3L12 15l-4 1 1-4z" /></>)

export const ArrowUp = ({ size = 17 }: P) => svg(size, <path d="M12 19V5M5.5 11.5L12 5l6.5 6.5" />, 2.4)

export const Box = ({ size = 18 }: P) =>
  svg(size, <><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" /><path d="M4 7.5l8 4.5 8-4.5M12 12v9" /></>)

export const Brain = ({ size = 18 }: P) =>
  svg(size, <><path d="M9 4a3 3 0 00-3 3 3 3 0 00-2 5 3 3 0 002 5 3 3 0 006 0V4.5A2.5 2.5 0 009 4z" /><path d="M15 4a3 3 0 013 3 3 3 0 012 5 3 3 0 01-2 5 3 3 0 01-6 0" /></>)

export const Message = ({ size = 18 }: P) =>
  svg(size, <path d="M4 5h16v11H9l-5 4z" />)
