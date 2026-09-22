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

export interface Neighbour {
  package: string
  version: string
  label: 'malicious' | 'benign'
  pool: string
  similarity: number
}

export interface CodeMatch {
  similarity: number
  query_file: string
  query_code: string
  known_package: string
  known_version: string
  known_label: 'malicious' | 'benign'
  known_file: string
  known_code: string
}

/** Nearest known training packages by code embedding -- a second opinion. */
export interface Similarity {
  malicious_percent: number
  k: number
  n_malicious: number
  neighbours: Neighbour[]
  matches: CodeMatch[]
  n_files_compared: number
}

/** Both methods combined, calibrated on held-out data (backend/assess.py). */
export interface Assessment {
  tier: 'malicious' | 'suspicious' | 'clean'
  classifier_flags: boolean
  similarity_flags: boolean | null
  /** Percent, re-weighted to a random-PyPI prior (prior_low). */
  chance_low: number
  /** Percent, re-weighted to an already-suspected prior (prior_high). */
  chance_high: number
  prior_low: number
  prior_high: number
  basis: string
  novel_malware_caught: string
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
  similarity: Similarity | null
  assessment: Assessment | null
}

export interface Health {
  model_loaded: boolean
  llm_available: boolean
  llm_model: string
  n_features: number
  similarity_ready: boolean
  calibration_ready: boolean
}

export type Message =
  | { kind: 'user'; id: string; text: string }
  | { kind: 'error'; id: string; text: string }
  | { kind: 'result'; id: string; result: AnalysisResult; explanation: string; streaming: boolean }
