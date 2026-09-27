interface Props {
  /** 0-100 */
  value: number
  label: string
  tone: 'malicious' | 'suspicious' | 'clean'
}

const COLOR = { malicious: 'var(--red)', suspicious: 'var(--orange)', clean: 'var(--green)' }

/** A single activity-style ring with the number in the middle. */
export function Gauge({ value, label, tone }: Props) {
  const r = 42
  const c = 2 * Math.PI * r
  const v = Math.max(0, Math.min(100, value))
  return (
    <div className="gauge" role="img" aria-label={`${label} ${v.toFixed(0)}%`}>
      <svg viewBox="0 0 100 100">
        <circle className="gauge-track" cx="50" cy="50" r={r} strokeWidth="9" fill="none" />
        <circle
          className="gauge-value"
          cx="50" cy="50" r={r} strokeWidth="9" fill="none" strokeLinecap="round"
          stroke={COLOR[tone]}
          strokeDasharray={c}
          strokeDashoffset={c * (1 - Math.max(v, 1.5) / 100)}
        />
      </svg>
      <div className="gauge-label">
        <b>{v.toFixed(0)}%</b>
        <span>{label}</span>
      </div>
    </div>
  )
}
