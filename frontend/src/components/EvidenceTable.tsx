import type { AnalysisResult, Evidence } from '../types'

/**
 * The feature attributions behind a verdict, largest effect first, each set
 * against the training data.
 *
 * Contributions are SHAP values from the gradient-boosted model: positive
 * pushed the score toward "malicious", negative toward "benign". The bars show
 * how common the observed value was among the malicious and the benign
 * packages the model learned from -- which is *why* the feature pushed the way
 * it did. This list is the ground truth the LLM explanation is written from.
 */
export function EvidenceTable({ result }: { result: AnalysisResult }) {
  if (result.evidence.length === 0) {
    return <p className="note">No individual signal had a meaningful effect on this score.</p>
  }

  return (
    <div>
      {result.evidence.map((e) => (
        <div className="evidence-row" key={e.feature}>
          <div className="evidence-top">
            <div className="evidence-desc">{e.description}</div>
            <span className={`push ${e.direction}`}>
              {e.direction === 'malicious' ? '↑ malicious' : '↓ benign'} {Math.abs(e.contribution).toFixed(2)}
            </span>
          </div>
          <div className="evidence-meta">
            {e.feature} = {observed(e)}
          </div>
          {e.training && <TrainingBars e={e} />}
        </div>
      ))}
      <p className="note">
        Bars: share of the {result.evidence[0].training?.n_malicious.toLocaleString() ?? '—'} malicious
        and {result.evidence[0].training?.n_benign.toLocaleString() ?? '—'} benign training packages
        with the same value or beyond. Arrows: SHAP contribution toward each verdict.
      </p>
    </div>
  )
}

function TrainingBars({ e }: { e: Evidence }) {
  const t = e.training!
  const pct = (x: number) => `${x < 0.01 && x > 0 ? '<1' : Math.round(x * 100)}%`
  const lean =
    t.ratio >= 1.05 ? <><b>{fmtRatio(t.ratio)}× more common in malware</b></>
    : t.ratio <= 0.95 ? <><b>{fmtRatio(1 / Math.max(t.ratio, 1e-3))}× more common in benign packages</b></>
    : <>about as common in both</>
  return (
    <div className="train">
      <span className="train-label">Malware in training</span>
      <span className="train-bar"><i className="mal" style={{ width: `${t.malicious_share * 100}%` }} /></span>
      <span className="train-pct">{pct(t.malicious_share)}</span>
      <span className="train-label">Benign in training</span>
      <span className="train-bar"><i className="ben" style={{ width: `${t.benign_share * 100}%` }} /></span>
      <span className="train-pct">{pct(t.benign_share)}</span>
      <span className="train-caption">
        Packages that {condition(e)}: {lean}.
      </span>
    </div>
  )
}

const isBinary = (f: string) =>
  f.startsWith('pkg_has_') || f.startsWith('install_has_') || f.endsWith('_cmdclass') || f.endsWith('_subclass')

function observed(e: Evidence): string {
  if (isBinary(e.feature)) return e.value ? 'yes' : 'no'
  return fmt(e.value)
}

/** 23.2558 -> 23.26; integers stay whole. */
function fmt(v: number): string {
  return String(Number(v.toPrecision(4)))
}

function condition(e: Evidence): string {
  if (isBinary(e.feature)) return e.value ? 'have this' : 'lack this'
  if (e.value === 0 && e.training?.tail === 'le') return 'have none'
  return `have ${e.training?.tail === 'le' ? 'at most' : 'at least'} ${fmt(e.value)}`
}

function fmtRatio(r: number): string {
  return r >= 10 ? Math.round(r).toString() : r.toFixed(1)
}
