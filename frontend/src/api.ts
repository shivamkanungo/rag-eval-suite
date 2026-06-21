const BASE = '/api'

export interface QueryResult {
  answer: string
  rewritten_query: string
  prompt_version: string
  latency_ms: Record<string, number>
  metadata: Record<string, unknown>
  sources: Source[]
  evaluation: EvalScores | null
}

export interface Source {
  source: string
  chunk_id: string
  rerank_score: number | null
  content_preview: string
}

export interface EvalScores {
  answer_relevancy: number
  faithfulness: number
  context_precision: number
  context_recall: number
  aggregate: number
}

export interface CollectionStats {
  document_count: number
  collection: string
}

export async function sendQuery(
  question: string,
  options: { ground_truth?: string; use_hyde?: boolean; run_evaluation?: boolean } = {}
): Promise<QueryResult> {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      run_evaluation: options.run_evaluation ?? true,
      use_hyde: options.use_hyde ?? false,
      ground_truth: options.ground_truth,
    }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function ingestText(text: string, source?: string): Promise<{ ingested: number }> {
  const res = await fetch(`${BASE}/ingest/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, source: source ?? 'manual' }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function ingestFile(file: File): Promise<{ filename: string; ingested: number }> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/ingest/file`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getStats(): Promise<CollectionStats> {
  const res = await fetch(`${BASE}/stats`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getHealth(): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Backend offline')
  return res.json()
}
