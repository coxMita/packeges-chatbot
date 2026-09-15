export interface Evidence {
  feature: string
  value: number
  contribution: number
  direction: 'malicious' | 'benign'
  description: string
}

export interface Metadata {
  summary: string
  author: string
  home_page: string
  n_releases: number
  sdist_bytes: number
  upload_time: string
}

export interface AnalysisResult {
  package: string
  version: string
  metadata: Metadata
  verdict: 'malicious' | 'benign'
  /** Percentage, 0-100. Distance from the decision boundary, not the raw probability. */
  confidence: number
  malicious_probability: number
  threshold: number
  model_agreement: string | null
  embed_probability: number | null
  evidence: Evidence[]
}

export interface Health {
  model_loaded: boolean
  llm_available: boolean
  llm_model: string
  n_features: number
}

export type Message =
  | { kind: 'user'; id: string; text: string }
  | { kind: 'error'; id: string; text: string }
  | { kind: 'result'; id: string; result: AnalysisResult; explanation: string; streaming: boolean }
