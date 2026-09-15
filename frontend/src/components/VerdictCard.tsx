import { useState } from 'react'
import type { AnalysisResult } from '../types'
import { EvidenceTable } from './EvidenceTable'
import { Explanation } from './Explanation'

interface Props {
  result: AnalysisResult
  explanation: string
  streaming: boolean
}

export function VerdictCard({ result, explanation, streaming }: Props) {
  const [showEvidence, setShowEvidence] = useState(false)
  const malicious = result.verdict === 'malicious'

  return (
    <div className={`card ${result.verdict}`}>
      <div className="card-head">
        <div className="verdict-row">
          <span className={`badge ${result.verdict}`}>
            {malicious ? 'Malicious' : 'No threat found'}
          </span>
          <span className="pkg-name">
            {result.package} <span className="pkg-version">{result.version}</span>
          </span>
        </div>

        {result.metadata.summary && <p className="summary">{result.metadata.summary}</p>}

        <div className="confidence">
          <div className="confidence-label">
            <span>Model confidence</span>
            <b>{result.confidence.toFixed(1)}%</b>
          </div>
          <div className="bar">
            <div
              className={`bar-fill ${result.verdict}`}
              style={{ width: `${Math.max(result.confidence, 2)}%` }}
            />
          </div>
        </div>
      </div>

      <div className="section">
        <button className="section-toggle" onClick={() => setShowEvidence((v) => !v)}>
          <span className="chevron">{showEvidence ? '▼' : '▶'}</span>
          Evidence the model used ({result.evidence.length} features)
        </button>
        {showEvidence && <EvidenceTable result={result} />}
      </div>

      <div className="section">
        <Explanation text={explanation} streaming={streaming} />
      </div>
    </div>
  )
}
