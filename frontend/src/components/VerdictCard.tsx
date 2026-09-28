import type { AnalysisResult, Assessment } from '../types'
import { Disclosure } from './Disclosure'
import { EvidenceTable } from './EvidenceTable'
import { FileScanPanel } from './FileScanPanel'
import { Gauge } from './Gauge'
import { Bolt, Chart, Code, Nodes } from './Icons'
import { SimilarityPanel } from './SimilarityPanel'

type Tier = 'malicious' | 'suspicious' | 'clean'

const TIER_LABEL: Record<Tier, string> = {
  malicious: 'Malicious',
  suspicious: 'Suspicious — review the code',
  clean: 'No threat found',
}

export function VerdictCard({ result }: { result: AnalysisResult }) {
  const a = result.assessment
  // Fall back to the classifier alone if the server has no calibration.
  const tier: Tier = a?.tier ?? (result.verdict === 'malicious' ? 'malicious' : 'clean')
  const scan = result.file_scan
  const flagged = scan?.files.length ?? 0
  const caps = result.capabilities ?? []

  return (
    <div className="card">
      <div className="card-head">
        <Gauge value={result.confidence} label="confidence" tone={tier} />
        <div className="head-text">
          <div className={`tier ${tier}`}>{TIER_LABEL[tier]}</div>
          <div className="pkg-line">
            {result.package} <span className="pkg-version">{result.version}</span>
          </div>
          {result.metadata.summary && <p className="summary">{result.metadata.summary}</p>}
        </div>
      </div>

      <div className="stats">
        <Stat
          label="Chance it is malware"
          value={a ? chanceRange(a) : '—'}
          sub={a ? 'random pick → already suspected' : 'no calibration loaded'}
          tone={tier}
        />
        <Stat
          label="Classifier"
          value={result.verdict === 'malicious' ? 'Flags it' : 'Clear'}
          sub={`p = ${result.malicious_probability.toFixed(3)} · threshold ${result.threshold}`}
          tone={result.verdict === 'malicious' ? 'malicious' : 'clean'}
        />
        <Stat
          label="Code similarity"
          value={result.similarity ? `${result.similarity.malicious_percent.toFixed(0)}% malware` : 'n/a'}
          sub={result.similarity
            ? `${result.similarity.n_malicious} of ${result.similarity.k} closest known packages`
            : 'no Python code to compare'}
          tone={a?.similarity_flags ? 'malicious' : undefined}
        />
      </div>

      {a && (
        <p className="card-note">
          Confidence is how far the classifier's score sits from its decision threshold.
          The chance is calibrated for malware resembling the training data; of recent
          malware unlike it, this detector caught {a.novel_malware_caught}. Based on{' '}
          {a.basis}.
        </p>
      )}

      {caps.length > 0 && (
        <Disclosure
          icon={<Bolt />}
          title="What the code can do"
          meta={`${caps.length} ${caps.length === 1 ? 'capability' : 'capabilities'}`}
          defaultOpen
        >
          <div className="caps">
            {caps.map((c) => (
              <div className="cap" key={c.feature}>
                <span className={`cap-dot ${tier}`} />
                <span className="cap-desc">{c.description}</span>
                {c.count > 1 && <span className="cap-count">×{c.count}</span>}
              </div>
            ))}
          </div>
          <p className="note">
            Concrete behaviour found by reading the code: what runs by itself on install or
            import, where decoded or downloaded data flows, and payloads outside Python. A
            package is only called malicious when it can do at least one of these.
          </p>
        </Disclosure>
      )}

      {scan && flagged > 0 && (
        <Disclosure
          icon={<Code />}
          title="Suspect code"
          meta={`${scan.n_files_flagged} of ${scan.n_files_scanned} files`}
          defaultOpen={tier !== 'clean'}
        >
          <FileScanPanel scan={scan} />
        </Disclosure>
      )}

      <Disclosure
        icon={<Chart />}
        title="Why the model decided this"
        meta={`${result.evidence.length} signals`}
        defaultOpen
      >
        <EvidenceTable result={result} />
      </Disclosure>

      {result.similarity && (
        <Disclosure
          icon={<Nodes />}
          title="Similar known packages"
          meta={`${result.similarity.malicious_percent.toFixed(0)}% malware`}
        >
          <SimilarityPanel similarity={result.similarity} />
        </Disclosure>
      )}
    </div>
  )
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub: string; tone?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value${tone ? ` tone-${tone}` : ''}`}>{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  )
}

function chanceRange(a: Assessment): string {
  const pct = (p: number) => `${p < 1 ? p.toFixed(1) : Math.round(p)}%`
  return a.chance_low === a.chance_high ? pct(a.chance_low) : `${pct(a.chance_low)} – ${pct(a.chance_high)}`
}
