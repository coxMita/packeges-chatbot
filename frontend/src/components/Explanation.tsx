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
    <div className="explanation">
      <div className="explanation-head">Explanation · local model</div>

      {paragraphs.length === 0 && streaming && (
        <div className="thinking">
          <span className="spinner" />
          Generating explanation…
        </div>
      )}

      {paragraphs.map((p, i) => (
        <p key={i}>
          {p}
          {streaming && i === paragraphs.length - 1 && <span className="cursor" />}
        </p>
      ))}
    </div>
  )
}
