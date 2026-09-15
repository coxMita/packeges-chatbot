import type { AnalysisResult } from '../types'

/**
 * The feature attributions behind a verdict, largest effect first.
 *
 * These are SHAP values from the gradient-boosted model: a positive
 * contribution pushed the score toward "malicious", a negative one toward
 * "benign". This table is the ground truth the LLM explanation is written
 * from -- if the prose ever cites something that is not here, it is unsupported.
 */
export function EvidenceTable({ result }: { result: AnalysisResult }) {
  if (result.evidence.length === 0) {
    return (
      <div className="evidence">
        <p className="evidence-note">
          No individual feature had a meaningful effect on this score.
        </p>
      </div>
    )
  }

  return (
    <div className="evidence">
      {result.evidence.map((e) => (
        <div className="evidence-row" key={e.feature}>
          <div>
            <div className="evidence-desc">{e.description}</div>
            <div className="evidence-meta">
              {e.feature} = {e.value}
            </div>
          </div>
          <div className={`contribution ${e.direction}`}>
            {e.contribution > 0 ? '+' : ''}
            {e.contribution.toFixed(3)}
          </div>
        </div>
      ))}

      <p className="evidence-note">
        SHAP contributions. Positive values pushed the score toward malicious,
        negative toward benign. Raw probability {result.malicious_probability.toFixed(4)},
        decision threshold {result.threshold}.
      </p>
    </div>
  )
}
