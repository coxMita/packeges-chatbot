import type { ReactNode } from 'react'

interface Props {
  text: string
  streaming: boolean
}

/**
 * The local LLM's prose, streamed token by token.
 *
 * The model is given only the verdict and the evidence list -- never the
 * package source -- so it can explain the classifier's reasoning but cannot
 * form an opinion of its own.
 */
export function Explanation({ text, streaming }: Props) {
  const paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim())

  return (
    <div className="prose">
      {paragraphs.length === 0 && streaming && (
        <div className="thinking">
          <span className="spinner" />
          Writing…
        </div>
      )}

      {paragraphs.map((p, i) => (
        <p key={i}>
          {inline(p)}
          {streaming && i === paragraphs.length - 1 && <span className="cursor" />}
        </p>
      ))}
    </div>
  )
}

/** Render the little markdown a small model still emits: `code` and **bold**. */
function inline(text: string): ReactNode[] {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, i) => {
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return <code key={i}>{part.slice(1, -1)}</code>
    }
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>
    }
    return part
  })
}
