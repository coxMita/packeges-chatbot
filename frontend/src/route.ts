/**
 * Decide whether chat input names a package to analyse or is a follow-up
 * question about the last result.
 *
 * A package spec is one PyPI name, optionally with a version that starts with a
 * digit (`requests`, `requests==2.31.0`, `requests 2.31.0`). It may sit inside
 * a short request that only makes sense as "analyse this": `check flask`,
 * `pip install flask`, `is flask safe?`, `what about flask`. The version must
 * look like a version so that "why not" is not read as package `why`, version
 * `not`.
 *
 * Everything else is a question, answered from the analyses already on screen
 * (the most recent one unless another is named).
 */
const NAME = '[A-Za-z0-9][A-Za-z0-9._-]*'
const VERSION = '(?:\\s*(?:==|@|\\s)\\s*v?\\d[\\w.!+-]*)?'
const SPEC = `(${NAME}${VERSION})`

const PATTERNS = [
  // flask · check flask · analyse flask==3.0.0 · pip install flask
  new RegExp(`^(?:(?:please\\s+)?(?:check|analy[sz]e|scan|test|inspect)\\s+(?:package\\s+)?|pip3?\\s+install\\s+)?${SPEC}\\s*[?.!]?$`, 'i'),
  // is flask safe? · is flask malicious · is flask==3.0 legit?
  new RegExp(`^is\\s+${SPEC}\\s+(?:safe|malicious|malware|legit(?:imate)?|dangerous|ok|okay|clean)\\s*\\??$`, 'i'),
  // what about flask? · how about flask · and flask?
  new RegExp(`^(?:what|how)\\s+about\\s+${SPEC}\\s*\\??$|^and\\s+${SPEC}\\s*\\?$`, 'i'),
]

/**
 * Words that are conversation, not package names.
 * (Most of these exist on PyPI, but nobody typing "why" after a verdict means
 * the package called `why`.) Pronouns stop "is it safe?" and "what about this?"
 * from being read as packages called `it` and `this`.
 */
const CHAT_WORDS = new Set([
  'why', 'how', 'what', 'explain', 'more', 'details', 'elaborate', 'really',
  'ok', 'okay', 'thanks', 'thank', 'thx', 'yes', 'no', 'sure', 'hmm', 'and',
  'so', 'continue', 'go', 'hi', 'hello', 'hey', 'help', 'it', 'this', 'that',
  'these', 'those', 'them', 'they', 'one', 'package', 'same', 'again', 'code',
  'evidence', 'summary', 'summarize', 'summarise', 'example', 'examples',
])

export type Route = { kind: 'analyze'; spec: string } | { kind: 'question'; text: string }

export function route(input: string): Route {
  const text = input.trim()
  for (const re of PATTERNS) {
    const m = re.exec(text)
    if (!m) continue
    const spec = (m.slice(1).find(Boolean) ?? '').trim()
    const name = spec.split(/[\s=@]/)[0].toLowerCase()
    if (CHAT_WORDS.has(name)) break
    return { kind: 'analyze', spec }
  }
  return { kind: 'question', text }
}
