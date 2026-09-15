import { useEffect, useRef, useState } from 'react'
import { analyze, getHealth, streamExplanation } from './api'
import { VerdictCard } from './components/VerdictCard'
import type { Health, Message } from './types'

const EXAMPLES = ['requests', 'flask', 'python-dateutil', 'urllib3==2.2.0']

export function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function submit(pkg: string) {
    const name = pkg.trim()
    if (!name || busy) return

    setInput('')
    setBusy(true)
    const userId = crypto.randomUUID()
    setMessages((m) => [...m, { kind: 'user', id: userId, text: name }])

    try {
      const result = await analyze(name)
      const id = crypto.randomUUID()

      setMessages((m) => [
        ...m,
        { kind: 'result', id, result, explanation: '', streaming: true },
      ])

      // The verdict is already on screen; the explanation fills in behind it.
      await streamExplanation(result, (chunk) => {
        setMessages((m) =>
          m.map((msg) =>
            msg.kind === 'result' && msg.id === id
              ? { ...msg, explanation: msg.explanation + chunk }
              : msg,
          ),
        )
      })

      setMessages((m) =>
        m.map((msg) =>
          msg.kind === 'result' && msg.id === id ? { ...msg, streaming: false } : msg,
        ),
      )
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          kind: 'error',
          id: crypto.randomUUID(),
          text: err instanceof Error ? err.message : String(err),
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>PyPI Package Analyser</h1>
        <span className="sub">static ML classifier + local LLM</span>
        <div className="status">
          <span>
            <i className={`dot ${health?.model_loaded ? 'on' : 'off'}`} />
            {health?.model_loaded ? `model · ${health.n_features} features` : 'model offline'}
          </span>
          <span>
            <i className={`dot ${health?.llm_available ? 'on' : 'off'}`} />
            {health?.llm_available ? health.llm_model : 'llm offline'}
          </span>
        </div>
      </header>

      <div className="messages">
        {messages.length === 0 && (
          <div className="intro">
            <p>
              Name a PyPI package and it will be downloaded, parsed and scored by a
              gradient-boosted model trained on ~2,500 real malicious packages.
            </p>
            <p>
              The package is never installed or executed — only read. The classifier
              decides; the local LLM only explains which features drove the decision.
            </p>
            <div className="examples">
              {EXAMPLES.map((e) => (
                <button key={e} onClick={() => submit(e)} disabled={busy}>
                  {e}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => {
          if (msg.kind === 'user') {
            return (
              <div className="msg-user" key={msg.id}>
                {msg.text}
              </div>
            )
          }
          if (msg.kind === 'error') {
            return (
              <div className="msg-error" key={msg.id}>
                {msg.text}
              </div>
            )
          }
          return (
            <VerdictCard
              key={msg.id}
              result={msg.result}
              explanation={msg.explanation}
              streaming={msg.streaming}
            />
          )
        })}

        {busy && (
          <div className="thinking">
            <span className="spinner" />
            Downloading and analysing…
          </div>
        )}

        <div ref={endRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          submit(input)
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="package name, e.g. requests or requests==2.31.0"
          disabled={busy}
          autoFocus
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Analyse
        </button>
      </form>
    </div>
  )
}
