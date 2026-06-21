import { useState, useEffect, useRef } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid
} from 'recharts'
import {
  Send, Upload, FileText, Database, Zap, ChevronDown,
  ChevronUp, AlertCircle, CheckCircle, Loader2, Search, Settings
} from 'lucide-react'
import { sendQuery, ingestText, ingestFile, getStats, getHealth, QueryResult, CollectionStats } from './api'
import './App.css'

// ─── Types ────────────────────────────────────────────────────────────────────

interface HistoryEntry {
  id: string
  question: string
  result: QueryResult
  timestamp: Date
}

// ─── Metric Gauge ─────────────────────────────────────────────────────────────

function MetricGauge({ label, value }: { label: string; value: number }) {
  const pct = value < 0 ? 0 : Math.round(value * 100)
  const color = value >= 0.8 ? '#34d399' : value >= 0.6 ? '#fbbf24' : value < 0 ? '#4b5262' : '#f87171'
  return (
    <div className="metric-gauge">
      <div className="metric-label">{label}</div>
      <div className="metric-bar-wrap">
        <div className="metric-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="metric-value" style={{ color }}>
        {value < 0 ? '—' : `${pct}%`}
      </div>
    </div>
  )
}

// ─── Score Card ───────────────────────────────────────────────────────────────

function EvalCard({ scores }: { scores: QueryResult['evaluation'] }) {
  if (!scores) return (
    <div className="eval-card empty">
      <AlertCircle size={14} /> Evaluation unavailable (no docs retrieved or RAGAS not installed)
    </div>
  )

  const radarData = [
    { metric: 'Relevancy', value: scores.answer_relevancy < 0 ? 0 : scores.answer_relevancy },
    { metric: 'Faithfulness', value: scores.faithfulness < 0 ? 0 : scores.faithfulness },
    { metric: 'Precision', value: scores.context_precision < 0 ? 0 : scores.context_precision },
    { metric: 'Recall', value: scores.context_recall < 0 ? 0 : scores.context_recall },
  ]

  return (
    <div className="eval-card">
      <div className="eval-header">
        <span className="eval-title">RAGAS Evaluation</span>
        <span className="aggregate-badge" style={{
          background: scores.aggregate >= 0.8 ? 'rgba(52,211,153,0.15)' : 'rgba(251,191,36,0.15)',
          color: scores.aggregate >= 0.8 ? '#34d399' : '#fbbf24',
        }}>
          {scores.aggregate >= 0 ? `${Math.round(scores.aggregate * 100)}% agg` : 'N/A'}
        </span>
      </div>

      <div className="eval-grid">
        <div className="eval-metrics">
          <MetricGauge label="Answer Relevancy" value={scores.answer_relevancy} />
          <MetricGauge label="Faithfulness" value={scores.faithfulness} />
          <MetricGauge label="Context Precision" value={scores.context_precision} />
          <MetricGauge label="Context Recall" value={scores.context_recall} />
        </div>

        <div className="radar-wrap">
          <ResponsiveContainer width="100%" height={160}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#252836" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#6b7280', fontSize: 11 }} />
              <Radar name="scores" dataKey="value" stroke="#6c8cff" fill="#6c8cff" fillOpacity={0.2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

// ─── Latency Bar ──────────────────────────────────────────────────────────────

function LatencyBreakdown({ latency }: { latency: Record<string, number> }) {
  const stages = [
    { key: 'query_rewrite_ms', label: 'Rewrite' },
    { key: 'retrieval_ms', label: 'Retrieval' },
    { key: 'rerank_ms', label: 'Rerank' },
    { key: 'generation_ms', label: 'Generate' },
  ]
  const data = stages.map(s => ({ label: s.label, ms: Math.round(latency[s.key] ?? 0) }))

  return (
    <div className="latency-section">
      <div className="section-label">Latency breakdown — {Math.round(latency.total_ms ?? 0)}ms total</div>
      <ResponsiveContainer width="100%" height={70}>
        <BarChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a1d28" vertical={false} />
          <XAxis dataKey="label" tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{ background: '#13161e', border: '1px solid #252836', borderRadius: 6, fontSize: 11 }}
            formatter={(v: number) => [`${v}ms`, 'Latency']}
          />
          <Bar dataKey="ms" fill="#6c8cff" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Query Result Card ────────────────────────────────────────────────────────

function ResultCard({ entry }: { entry: HistoryEntry }) {
  const [expanded, setExpanded] = useState(false)
  const [showRaw, setShowRaw] = useState(false)
  const r = entry.result

  return (
    <div className="result-card">
      <div className="result-header">
        <div className="result-question">{entry.question}</div>
        <div className="result-meta">
          <span className="mono text-dim" style={{ fontSize: 11 }}>
            {entry.timestamp.toLocaleTimeString()}
          </span>
          <span className="badge">{r.prompt_version}</span>
          {r.metadata.use_hyde && <span className="badge badge-purple">HyDE</span>}
        </div>
      </div>

      <div className="result-answer">{r.answer}</div>

      {r.rewritten_query !== entry.question && (
        <div className="rewrite-hint">
          <Search size={11} /> Rewritten: <span className="mono">{r.rewritten_query}</span>
        </div>
      )}

      <EvalCard scores={r.evaluation} />
      <LatencyBreakdown latency={r.latency_ms} />

      <button className="expand-btn" onClick={() => setExpanded(x => !x)}>
        {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        {expanded ? 'Hide' : 'Show'} sources ({r.sources.length})
      </button>

      {expanded && (
        <div className="sources-list">
          {r.sources.map((src, i) => (
            <div className="source-item" key={i}>
              <div className="source-meta">
                <span className="mono text-accent" style={{ fontSize: 11 }}>{src.source}</span>
                {src.rerank_score != null && (
                  <span className="rerank-score">
                    ↑ {(src.rerank_score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <div className="source-preview">{src.content_preview}…</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Ingest Panel ─────────────────────────────────────────────────────────────

function IngestPanel({ onSuccess }: { onSuccess: () => void }) {
  const [tab, setTab] = useState<'text' | 'file'>('text')
  const [text, setText] = useState('')
  const [source, setSource] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<{ ok: boolean; msg: string } | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleText = async () => {
    if (!text.trim()) return
    setLoading(true); setStatus(null)
    try {
      const r = await ingestText(text, source || 'manual')
      setStatus({ ok: true, msg: `Ingested ${r.ingested} chunks` })
      setText(''); setSource('')
      onSuccess()
    } catch (e) {
      setStatus({ ok: false, msg: String(e) })
    } finally { setLoading(false) }
  }

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setLoading(true); setStatus(null)
    try {
      const r = await ingestFile(file)
      setStatus({ ok: true, msg: `Ingested ${r.ingested} chunks from ${r.filename}` })
      onSuccess()
    } catch (e) {
      setStatus({ ok: false, msg: String(e) })
    } finally { setLoading(false); if (fileRef.current) fileRef.current.value = '' }
  }

  return (
    <div className="ingest-panel">
      <div className="panel-title"><Database size={14} /> Ingest Documents</div>
      <div className="tab-row">
        <button className={`tab-btn ${tab === 'text' ? 'active' : ''}`} onClick={() => setTab('text')}>
          <FileText size={12} /> Text
        </button>
        <button className={`tab-btn ${tab === 'file' ? 'active' : ''}`} onClick={() => setTab('file')}>
          <Upload size={12} /> File
        </button>
      </div>

      {tab === 'text' ? (
        <>
          <input
            className="input-field"
            placeholder="Source name (optional)"
            value={source}
            onChange={e => setSource(e.target.value)}
          />
          <textarea
            className="textarea-field"
            placeholder="Paste document text to index…"
            value={text}
            onChange={e => setText(e.target.value)}
            rows={5}
          />
          <button className="btn-primary" onClick={handleText} disabled={loading || !text.trim()}>
            {loading ? <Loader2 size={13} className="spin" /> : <Zap size={13} />}
            Index Text
          </button>
        </>
      ) : (
        <>
          <label className="file-drop">
            <input ref={fileRef} type="file" accept=".pdf,.txt,.md" onChange={handleFile} style={{ display: 'none' }} />
            <Upload size={20} className="text-dim" />
            <span>Click to upload PDF or TXT</span>
          </label>
        </>
      )}

      {status && (
        <div className={`status-msg ${status.ok ? 'ok' : 'err'}`}>
          {status.ok ? <CheckCircle size={12} /> : <AlertCircle size={12} />}
          {status.msg}
        </div>
      )}
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [question, setQuestion] = useState('')
  const [groundTruth, setGroundTruth] = useState('')
  const [useHyde, setUseHyde] = useState(false)
  const [showOptions, setShowOptions] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [stats, setStats] = useState<CollectionStats | null>(null)
  const [backendOk, setBackendOk] = useState<boolean | null>(null)

  const refreshStats = async () => {
    try {
      const s = await getStats()
      setStats(s)
    } catch { /* ignore */ }
  }

  useEffect(() => {
    getHealth()
      .then(() => { setBackendOk(true); refreshStats() })
      .catch(() => setBackendOk(false))
  }, [])

  const handleQuery = async () => {
    if (!question.trim() || loading) return
    setLoading(true); setError(null)
    try {
      const result = await sendQuery(question, { use_hyde: useHyde, ground_truth: groundTruth || undefined })
      setHistory(h => [{
        id: crypto.randomUUID(),
        question,
        result,
        timestamp: new Date(),
      }, ...h])
      setQuestion('')
    } catch (e) {
      setError(String(e))
    } finally { setLoading(false) }
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleQuery()
  }

  return (
    <div className="app-layout">
      {/* ─── Sidebar ─────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <Zap size={18} className="text-accent" />
          <span>RAG<span className="text-accent">Eval</span></span>
        </div>

        <div className="status-pill" style={{
          color: backendOk === null ? '#6b7280' : backendOk ? '#34d399' : '#f87171',
          borderColor: backendOk === null ? '#252836' : backendOk ? 'rgba(52,211,153,0.3)' : 'rgba(248,113,113,0.3)',
        }}>
          <span className="status-dot" style={{
            background: backendOk === null ? '#6b7280' : backendOk ? '#34d399' : '#f87171'
          }} />
          {backendOk === null ? 'Connecting…' : backendOk ? 'API Online' : 'API Offline'}
        </div>

        {stats && (
          <div className="stats-card">
            <div className="stats-row">
              <span className="text-dim">Documents</span>
              <span className="mono text-accent">{stats.document_count}</span>
            </div>
            <div className="stats-row">
              <span className="text-dim">Collection</span>
              <span className="mono" style={{ fontSize: 11 }}>{stats.collection}</span>
            </div>
          </div>
        )}

        <IngestPanel onSuccess={refreshStats} />

        <div className="sidebar-footer">
          <a href="https://docs.ragas.io" target="_blank" rel="noreferrer" className="footer-link">RAGAS Docs</a>
          <a href="https://smith.langchain.com" target="_blank" rel="noreferrer" className="footer-link">LangSmith</a>
        </div>
      </aside>

      {/* ─── Main ────────────────────────────────────────────────── */}
      <main className="main-content">
        <header className="main-header">
          <div>
            <h1 className="main-title">RAG Evaluation Suite</h1>
            <p className="main-subtitle">
              Vector search · Prompt engineering · RAGAS + LangSmith evaluation
            </p>
          </div>
        </header>

        {/* Query Input */}
        <div className="query-box">
          <textarea
            className="query-input"
            placeholder="Ask a question about your documents… (⌘+Enter to send)"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKey}
            rows={3}
            disabled={loading}
          />

          <div className="query-controls">
            <button className="options-toggle" onClick={() => setShowOptions(x => !x)}>
              <Settings size={13} /> Options {showOptions ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
            <button className="btn-primary btn-send" onClick={handleQuery} disabled={loading || !question.trim()}>
              {loading ? <Loader2 size={14} className="spin" /> : <Send size={14} />}
              {loading ? 'Thinking…' : 'Query'}
            </button>
          </div>

          {showOptions && (
            <div className="options-panel">
              <label className="option-row">
                <input type="checkbox" checked={useHyde} onChange={e => setUseHyde(e.target.checked)} />
                <span>Use HyDE (hypothetical document expansion)</span>
              </label>
              <input
                className="input-field"
                placeholder="Ground truth answer (optional, improves context recall metric)"
                value={groundTruth}
                onChange={e => setGroundTruth(e.target.value)}
              />
            </div>
          )}
        </div>

        {error && (
          <div className="error-banner">
            <AlertCircle size={14} /> {error}
          </div>
        )}

        {/* Results */}
        <div className="results-area">
          {history.length === 0 && !loading && (
            <div className="empty-state">
              <Search size={32} className="text-dim" />
              <p>Ask a question to see RAG pipeline output with live RAGAS metrics</p>
              <p className="text-dim" style={{ fontSize: 12 }}>
                Tip: Ingest documents first using the panel on the left, or run{' '}
                <code className="mono">python -m scripts.seed_data</code> to load demo data.
              </p>
            </div>
          )}

          {loading && (
            <div className="loading-card">
              <Loader2 size={20} className="spin text-accent" />
              <span>Running pipeline: retrieve → rerank → generate → evaluate…</span>
            </div>
          )}

          {history.map(entry => (
            <ResultCard key={entry.id} entry={entry} />
          ))}
        </div>
      </main>
    </div>
  )
}
