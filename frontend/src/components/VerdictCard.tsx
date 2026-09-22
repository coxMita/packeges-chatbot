import { useState } from 'react'
import type { AnalysisResult, Assessment } from '../types'
import { EvidenceTable } from './EvidenceTable'
import { Explanation } from './Explanation'
import { SimilarityPanel } from './SimilarityPanel'

interface Props {
  result: AnalysisResult
  explanation: string
  streaming: boolean
}

export function VerdictCard({ result, explanation, streaming }: Props) {
  const [showEvidence, setShowEvidence] = useState(false)
  const assessment = result.assessment
  // Fall back to the classifier alone if the server has no calibration.
  const tier = assessment?.tier ?? (result.verdict === 'malicious' ? 'malicious' : 'clean')
  const tone = tier === 'clean' ? 'benign' : tier

  return (
    <div className={`card ${tone}`}>
      <div className="card-head">
        <div className="verdict-row">
          <span className={`badge ${tone}`}>{TIER_LABEL[tier]}</span>
          <span className="pkg-name">
            {result.package} <span className="pkg-version">{result.version}</span>
          </span>
        </div>

        {result.metadata.summary && <p className="summary">{result.metadata.summary}</p>}

        {assessment && <Chance assessment={assessment} />}

        <div className="confidence">
          <div className="confidence-label">
            <span>Classifier confidence ({result.verdict})</span>
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

      {result.similarity && <SimilarityPanel similarity={result.similarity} />}

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

const TIER_LABEL = {
  malicious: 'Malicious',
  suspicious: 'Suspicious — review the code',
  clean: 'No threat found',
} as const

function Chance({ assessment: a }: { assessment: Assessment }) {
  const pct = (p: number) => `${p < 1 ? p.toFixed(1) : Math.round(p)}%`
  const range = a.chance_low === a.chance_high ? pct(a.chance_low)
    : `${pct(a.chance_low)} – ${pct(a.chance_high)}`
  const who = (flag: boolean | null) => (flag === null ? 'n/a' : flag ? 'flags it' : 'clear')

  return (
    <div className="chance">
      <div className="chance-main">
        <span>Chance it is really malware</span>
        <b className={a.tier === 'clean' ? 'benign' : a.tier}>{range}</b>
      </div>
      <p className="chance-note">
        Classifier {who(a.classifier_flags)} · code similarity {who(a.similarity_flags)}.{' '}
        Low end: a package picked at random from PyPI ({a.prior_low * 100}% malware). High
        end: one you already had reason to doubt ({a.prior_high * 100}%). Calibrated on
        known malware styles only — of recent malware unlike the training data, this
        detector caught {a.novel_malware_caught}.
      </p>
    </div>
  )
}
