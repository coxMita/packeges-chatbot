import { useState } from 'react'
import type { CodeMatch, Similarity } from '../types'

/**
 * Nearest known packages by code, and the file pairs that look most alike.
 *
 * This is a second opinion alongside the classifier, computed by comparing the
 * package's code embeddings against every package in the training data. It
 * shows likeness to known samples, not proof -- the note under the list says so.
 */
export function SimilarityPanel({ similarity }: { similarity: Similarity }) {
  const [showNeighbours, setShowNeighbours] = useState(false)
  const [showCode, setShowCode] = useState(true)
  const pct = similarity.malicious_percent
  const tone = pct >= 50 ? 'malicious' : 'benign'

  return (
    <>
      <div className="section similarity-head">
        <div className="confidence-label">
          <span>Similarity to known malware</span>
          <b className={tone}>{pct.toFixed(1)}%</b>
        </div>
        <div className="bar">
          <div className={`bar-fill ${tone}`} style={{ width: `${Math.max(pct, 2)}%` }} />
        </div>
        <p className="similarity-note">
          {similarity.n_malicious} of the {similarity.k} known packages whose code is
          closest to this one are malware ({similarity.n_files_compared} files compared).
        </p>
      </div>

      <div className="section">
        <button className="section-toggle" onClick={() => setShowNeighbours((v) => !v)}>
          <span className="chevron">{showNeighbours ? '▼' : '▶'}</span>
          Closest known packages ({similarity.neighbours.length})
        </button>
        {showNeighbours && (
          <div className="evidence">
            {similarity.neighbours.map((n) => (
              <div className="evidence-row" key={`${n.package}@${n.version}`}>
                <div>
                  <div className="evidence-desc">
                    {n.package} <span className="pkg-version">{n.version}</span>
                  </div>
                  <div className="evidence-meta">
                    {n.label === 'malicious' ? 'known malware' : `benign (${n.pool})`}
                  </div>
                </div>
                <div className={`contribution ${n.label}`}>{n.similarity.toFixed(3)}</div>
              </div>
            ))}
            <p className="evidence-note">
              Cosine similarity of code embeddings (1.0 = identical). Likeness to
              known samples, not proof of behaviour.
            </p>
          </div>
        )}
      </div>

      {similarity.matches.length > 0 && (
        <div className="section">
          <button className="section-toggle" onClick={() => setShowCode((v) => !v)}>
            <span className="chevron">{showCode ? '▼' : '▶'}</span>
            Most similar code
          </button>
          {showCode && (
            <div className="matches">
              {similarity.matches.map((m) => (
                <MatchView key={m.known_label} match={m} />
              ))}
            </div>
          )}
        </div>
      )}
    </>
  )
}

function MatchView({ match }: { match: CodeMatch }) {
  const label = match.known_label === 'malicious' ? 'closest known malware' : 'closest known benign'
  return (
    <div className="match">
      <div className="match-title">
        <span className={`contribution ${match.known_label}`}>{label}</span>
        <span className="match-sim">similarity {match.similarity.toFixed(3)}</span>
      </div>
      <div className="code-pair">
        <figure>
          <figcaption>this package · {match.query_file}</figcaption>
          <pre>{match.query_code}</pre>
        </figure>
        <figure>
          <figcaption>
            {match.known_package} {match.known_version} · {shortPath(match.known_file)}
          </figcaption>
          <pre>{match.known_code}</pre>
        </figure>
      </div>
    </div>
  )
}

/** DataDog samples sit under dated wrapper folders; keep the meaningful tail. */
function shortPath(path: string): string {
  const parts = path.split('/')
  return parts.length > 3 ? '…/' + parts.slice(-3).join('/') : path
}
