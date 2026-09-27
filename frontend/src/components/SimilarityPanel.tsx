import type { CodeMatch, Similarity } from '../types'

/**
 * Nearest known packages by code, and the file pairs that look most alike.
 *
 * A second opinion alongside the classifier, computed by comparing the
 * package's code embeddings against every package in the training data. It
 * shows likeness to known samples, not proof.
 */
export function SimilarityPanel({ similarity }: { similarity: Similarity }) {
  const pct = similarity.malicious_percent
  return (
    <div>
      <div className="stat-sub">
        {similarity.n_malicious} of the {similarity.k} known packages whose code is closest are
        malware, weighted by closeness ({similarity.n_files_compared} files compared).
      </div>
      <div className="sim-bar"><i style={{ width: `${Math.max(pct, 1.5)}%` }} /></div>

      <div className="list">
        {similarity.neighbours.map((n) => (
          <div className="list-row" key={`${n.package}@${n.version}`}>
            <span className="name">
              {n.package} <span className="pkg-version">{n.version}</span>
            </span>
            <span className={`tag${n.label === 'malicious' ? ' danger' : ''}`}>
              {n.label === 'malicious' ? 'known malware' : `benign · ${n.pool}`}
            </span>
            <span className="num">{n.similarity.toFixed(3)}</span>
          </div>
        ))}
      </div>

      {similarity.matches.map((m) => (
        <MatchView key={m.known_label} match={m} />
      ))}
      <p className="note">
        Cosine similarity of code embeddings (1.0 = identical). Likeness to known samples,
        not proof of behaviour.
      </p>
    </div>
  )
}

function MatchView({ match }: { match: CodeMatch }) {
  return (
    <>
      <div className="subhead">
        Closest known {match.known_label === 'malicious' ? 'malware' : 'benign'} file ·{' '}
        {match.similarity.toFixed(3)}
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
    </>
  )
}

/** DataDog samples sit under dated wrapper folders; keep the meaningful tail. */
function shortPath(path: string): string {
  const parts = path.split('/')
  return parts.length > 3 ? '…/' + parts.slice(-3).join('/') : path
}
