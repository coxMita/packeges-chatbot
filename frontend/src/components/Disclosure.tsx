import { useState, type ReactNode } from 'react'
import { Chevron } from './Icons'

interface Props {
  icon: ReactNode
  title: string
  meta?: ReactNode
  defaultOpen?: boolean
  children: ReactNode
}

/** An inset-grouped row that expands in place, like a Settings list. */
export function Disclosure({ icon, title, meta, defaultOpen = false, children }: Props) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="disclosure">
      <button className="disclosure-head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className="disclosure-icon">{icon}</span>
        <span className="disclosure-title">{title}</span>
        {meta !== undefined && <span className="disclosure-meta">{meta}</span>}
        <Chevron open={open} />
      </button>
      {open && <div className="disclosure-body">{children}</div>}
    </div>
  )
}
