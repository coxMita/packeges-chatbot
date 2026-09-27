import type { AnalysisResult, ChatTurn, Health } from './types'

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
): Promise<void> {
  return postStream(
    '/api/explain',
    { package: result.package, version: result.version, verdict: withoutCode(result) },
    onChunk,
  )
}

/** Stream the answer to a follow-up question about earlier analyses. */
export function askQuestion(
  question: string,
  analyses: AnalysisResult[],
  history: ChatTurn[],
  onChunk: (text: string) => void,
): Promise<void> {
  return postStream(
    '/api/chat',
    { question, analyses: analyses.map(withoutCode), history },
    onChunk,
  )
}

/**
 * POST a JSON body and read the server-sent events off the response.
 *
 * EventSource only does GET, and the payloads are too large for a URL, so the
 * SSE frames are parsed by hand.
 */
async function postStream(url: string, body: unknown, onChunk: (text: string) => void) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok || !r.body) {
    const err = await r.json().catch(() => ({ detail: r.statusText }))
    const detail = typeof err.detail === 'string' ? err.detail : `request failed (${r.status})`
    throw new Error(detail)
  }

  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buffer += value
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const data = frame.replace(/^data: /, '')
      if (data === '[DONE]') return
      let payload: { text?: string; error?: string }
      try {
        payload = JSON.parse(data)
      } catch {
        continue // keep-alive or partial frame
      }
      if (payload.error) throw new Error(payload.error)
      if (payload.text) onChunk(payload.text)
    }
  }
}

/**
 * The explainer is only ever given summaries, never package source: code
 * snippets and scanned source lines are blanked before anything is sent back.
 */
function withoutCode(result: AnalysisResult): AnalysisResult {
  return {
    ...result,
    similarity: result.similarity && {
      ...result.similarity,
      matches: result.similarity.matches.map((m) => ({ ...m, query_code: '', known_code: '' })),
    },
    file_scan: result.file_scan && {
      ...result.file_scan,
      // Keep where and what (line, call, category); drop the source line itself.
      files: result.file_scan.files.map((f) => ({ ...f, hits: f.hits.map((h) => ({ ...h, code: '' })) })),
    },
  }
}
