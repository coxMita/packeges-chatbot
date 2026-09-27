import { useEffect, useRef, useState } from 'react'
import { analyze, askQuestion, getHealth, streamExplanation } from './api'
import { Explanation } from './components/Explanation'
import { ArrowUp, Box, Brain, Compose, Message as MessageIcon, Shield, Sparkle } from './components/Icons'
import { VerdictCard } from './components/VerdictCard'
import { route } from './route'
import type { AnalysisResult, ChatTurn, Health, Message } from './types'

const EXAMPLES = ['requests', 'flask', 'python-dateutil', 'urllib3==2.2.0']

export function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [pending, setPending] = useState<'analyze' | 'question' | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    let alive = true
    const poll = () => getHealth().then((h) => alive && setHealth(h)).catch(() => alive && setHealth(null))
    poll()
    const t = setInterval(poll, 30_000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [])

  const busy = pending !== null
  const lastResult = [...messages].reverse().find((m) => m.kind === 'result')
  const current = lastResult?.kind === 'result' ? lastResult.result : null

  async function submit(raw: string) {
    const text = raw.trim()
    if (!text || busy) return

    setInput('')
    if (inputRef.current) inputRef.current.style.height = 'auto'
    setMessages((m) => [...m, { kind: 'user', id: crypto.randomUUID(), text }])

    const r = route(text)
    try {
      if (r.kind === 'analyze') {
        try {
          await runAnalysis(r.spec)
        } catch (err) {
          // A lone word that isn't on PyPI, typed after a result, was a reply.
          const notFound = err instanceof Error && err.message.endsWith('is not on PyPI')
          if (!(notFound && current && /^[A-Za-z][\w-]*[?.!]?$/.test(text))) throw err
          await runQuestion(text)
        }
      } else {
        await runQuestion(r.text)
      }
    } catch (err) {
      pushError(err)
    } finally {
      setPending(null)
      inputRef.current?.focus()
    }
  }

  function pushError(err: unknown) {
    setMessages((m) => [
      ...m,
      { kind: 'error', id: crypto.randomUUID(), text: err instanceof Error ? err.message : String(err) },
    ])
  }

  /** Append streamed text to one message. */
  function appendTo(id: string, chunk: string) {
    setMessages((m) =>
      m.map((msg) => {
        if (msg.id !== id) return msg
        if (msg.kind === 'result') return { ...msg, explanation: msg.explanation + chunk }
        if (msg.kind === 'answer') return { ...msg, text: msg.text + chunk }
        return msg
      }),
    )
  }

  function finish(id: string) {
    setMessages((m) =>
      m.map((msg) =>
        msg.id === id && (msg.kind === 'result' || msg.kind === 'answer') ? { ...msg, streaming: false } : msg,
      ),
    )
  }

  async function runAnalysis(spec: string) {
    setPending('analyze')
    const result = await analyze(spec)
    const id = crypto.randomUUID()
    setMessages((m) => [...m, { kind: 'result', id, result, explanation: '', streaming: true }])

    // The verdict is already on screen; the explanation fills in behind it.
    try {
      await streamExplanation(result, (chunk) => appendTo(id, chunk))
    } finally {
      finish(id)
    }
  }

  async function runQuestion(question: string) {
    setPending('question')
    const analyses: AnalysisResult[] = []
    const history: ChatTurn[] = []
    // `messages` still holds the conversation as it was before this question.
    for (const msg of messages) {
      if (msg.kind === 'user') history.push({ role: 'user', content: msg.text })
      else if (msg.kind === 'result') {
        analyses.push(msg.result)
        if (msg.explanation) history.push({ role: 'assistant', content: msg.explanation })
      } else if (msg.kind === 'answer' && msg.text) {
        history.push({ role: 'assistant', content: msg.text })
      }
    }

    const id = crypto.randomUUID()
    const about = analyses.at(-1)
    setMessages((m) => [
      ...m,
      { kind: 'answer', id, text: '', streaming: true, about: about ? `${about.package} ${about.version}` : undefined },
    ])
    try {
      await askQuestion(question, analyses, history, (chunk) => appendTo(id, chunk))
    } finally {
      finish(id)
    }
  }

  const composer = (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault()
        submit(input)
      }}
    >
      <textarea
        ref={inputRef}
        value={input}
        rows={1}
        onChange={(e) => {
          setInput(e.target.value)
          // Grow with the text, up to the CSS max-height.
          e.target.style.height = 'auto'
          e.target.style.height = `${e.target.scrollHeight}px`
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault()
            submit(input)
          }
        }}
        placeholder={
          current
            ? `Ask about ${current.package}, or name another package`
            : 'Package name, e.g. requests or requests==2.31.0'
        }
        aria-label="Package name or question"
        disabled={busy}
        autoFocus
      />
      <button className="send" type="submit" disabled={busy || !input.trim()} aria-label="Send">
        <ArrowUp />
      </button>
    </form>
  )

  const navbar = (
    <header className="navbar">
      <div className="brand">
        <span className="brand-mark"><Shield size={15} /></span>
        Package Inspector
      </div>
      <div className="navbar-spacer" />
      <div className="status" aria-live="polite">
        <span><i className={`dot ${health?.model_loaded ? 'ok' : 'off'}`} />Classifier</span>
        <span><i className={`dot ${health?.similarity_ready ? 'ok' : 'off'}`} />Similarity</span>
        <span>
          <i className={`dot ${health?.llm_available ? 'ok' : 'off'}`} />
          {health?.llm_model ?? 'LLM'}
        </span>
      </div>
      {messages.length > 0 && (
        <button className="nav-button" onClick={() => setMessages([])} disabled={busy}>
          <Compose size={14} /> New
        </button>
      )}
    </header>
  )

  if (messages.length === 0) {
    return (
      <div className="app">
        {navbar}
        <main className="welcome">
          <div className="hero-mark"><Shield size={32} /></div>
          <h1>Is this package safe?</h1>
          <p className="lede">
            Name any PyPI package. A classifier trained on thousands of real malware samples
            scores it, and a local model explains why.
          </p>
          {composer}
          <div className="examples">
            {EXAMPLES.map((e) => (
              <button className="chip" key={e} onClick={() => submit(e)} disabled={busy}>
                {e}
              </button>
            ))}
          </div>
          <div className="tiles">
            <div className="tile">
              <div className="tile-icon"><Box /></div>
              <b>Never installed</b>
              <p>Downloaded and parsed as text. No package code ever runs.</p>
            </div>
            <div className="tile">
              <div className="tile-icon"><Brain /></div>
              <b>Measured, not guessed</b>
              <p>Every signal is compared with the malware and benign packages it learned from.</p>
            </div>
            <div className="tile">
              <div className="tile-icon"><MessageIcon /></div>
              <b>Ask follow-ups</b>
              <p>After a result, ask anything about it. Name another package to check that one.</p>
            </div>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="app">
      {navbar}

      <div className="messages">
        <div className="thread">
          {messages.map((msg) => {
            if (msg.kind === 'user') {
              return <div className="msg-user" key={msg.id}>{msg.text}</div>
            }
            if (msg.kind === 'error') {
              return <div className="msg-error" key={msg.id}>{msg.text}</div>
            }
            if (msg.kind === 'answer') {
              return (
                <div className="msg-assistant" key={msg.id}>
                  <div className="reply">
                    <div className="reply-label">
                      <Sparkle /> Answer
                      {msg.about && <span className="about">· about {msg.about}</span>}
                    </div>
                    <Explanation text={msg.text} streaming={msg.streaming} />
                  </div>
                </div>
              )
            }
            return (
              <div className="msg-assistant" key={msg.id}>
                <VerdictCard result={msg.result} />
                <div className="reply">
                  <div className="reply-label"><Sparkle /> Explanation</div>
                  <Explanation text={msg.explanation} streaming={msg.streaming} />
                </div>
              </div>
            )
          })}

          {pending === 'analyze' && !messages.some((m) => m.kind === 'result' && m.streaming) && (
            <div className="thinking">
              <span className="spinner" />
              Downloading and analysing…
            </div>
          )}

          <div ref={endRef} />
        </div>
      </div>

      <div className="dock">
        {current && (
          <div className="context-chip">
            Follow-up questions are about <code>{current.package} {current.version}</code>
          </div>
        )}
        {composer}
        <p className="disclaimer">
          Verdicts come from the classifier. The local model explains them and can word things badly.
        </p>
      </div>
    </div>
  )
}
