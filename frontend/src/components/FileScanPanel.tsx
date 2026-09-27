import { useState } from 'react'
import type { FileScan, ScannedFile } from '../types'
import { Chevron } from './Icons'

/**
 * Where the suspicious code sits: files ranked by how many different kinds of
 * suspicious operation they hold, with the exact lines. Guidance for a human
 * reviewer, not a verdict -- plenty of benign files use one or two of these.
 */
export function FileScanPanel({ scan }: { scan: FileScan }) {
  return (
    <div className="scan">
      {scan.files.map((f, i) => (
        <FileView key={f.path} file={f} defaultOpen={i === 0} />
      ))}
      <p className="note">
        Ranked by how many different kinds of suspicious operation sit in one file.
        Legitimate libraries spread these across modules; malware hidden in a real
        library tends to pack them together. Guidance for review, not a verdict.
      </p>
    </div>
  )
}

function FileView({ file, defaultOpen }: { file: ScannedFile; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="scan-file">
      <button className="scan-file-head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <Chevron open={open} size={12} />
        <span className="scan-path">{shortPath(file.path)}</span>
        {file.runs_at_install && <span className="tag danger">runs at install</span>}
        {file.is_test && <span className="tag">test file</span>}
        <span className="scan-loc">{file.loc} lines</span>
      </button>
      <div className="scan-cats">
        {file.categories.map((c) => (
          <span key={c.id} className={`tag cat-${c.id}`}>
            {c.label}
          </span>
        ))}
      </div>
      {open && (
        <pre className="scan-lines">
          {file.hits.map((h) => (
            <div key={`${h.category}-${h.line}`} className="scan-line" title={h.label}>
              <span className="ln">{h.line}</span>
              <span className="code">{h.code}</span>
            </div>
          ))}
        </pre>
      )}
    </div>
  )
}

/** Drop the sdist's top-level `name-version/` folder. */
function shortPath(path: string): string {
  const parts = path.split('/')
  return parts.length > 1 ? parts.slice(1).join('/') : path
}
