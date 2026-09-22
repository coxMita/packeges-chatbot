import type { AnalysisResult, Health } from './types'

export async function getHealth(): Promise<Health> {
  const r = await fetch('/api/health')
  if (!r.ok) throw new Error(`health check failed (${r.status})`)
  return r.json()
}

export async function analyze(pkg: string): Promise<AnalysisResult> {
  const r = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ package: pkg }),
  })
  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: r.statusText }))
    throw new Error(body.detail ?? `request failed (${r.status})`)
  }
  return r.json()
}

/**
 * Stream the LLM explanation of an already-computed verdict.
 *
 * The full verdict is sent back to the server rather than re-analysed, so the
 * prose is guaranteed to describe the exact result on screen. `onChunk` fires
 * per token; the returned promise resolves when the stream closes.
 */
export function streamExplanation(
  result: AnalysisResult,
  onChunk: (text: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const params = new URLSearchParams({
    package: result.package,
    version: result.version,
    verdict: JSON.stringify(withoutCode(result)),
  })

  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/explain?${params}`)

    const close = () => {
      source.close()
      resolve()
    }

    source.onmessage = (event) => {
      if (event.data === '[DONE]') return close()
      try {
        const payload = JSON.parse(event.data)
        if (payload.error) {
          source.close()
          return reject(new Error(payload.error))
        }
        if (payload.text) onChunk(payload.text)
      } catch {
        /* keep-alive or partial frame; ignore */
      }
    }

    // EventSource reconnects on error by default, which would restart
    // generation from scratch. Close on the first failure instead.
    source.onerror = () => close()
    signal?.addEventListener('abort', close)
  })
}

/**
 * The explainer is only ever given summaries, never package source -- and the
 * code snippets would also blow past URL length limits on this GET request.
 */
function withoutCode(result: AnalysisResult): AnalysisResult {
  if (!result.similarity) return result
  return {
    ...result,
    similarity: {
      ...result.similarity,
      matches: result.similarity.matches.map((m) => ({ ...m, query_code: '', known_code: '' })),
    },
  }
}
