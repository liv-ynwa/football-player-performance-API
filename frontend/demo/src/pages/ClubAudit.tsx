import { useState, useEffect } from 'react'
import { api, fmt } from '../api'
import { SectionHeader, SearchInput, Badge, Loading, ErrorMessage, Empty } from '../components/ui'

export default function ClubAudit() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { search('') }, [])

  async function search(q: string) {
    try {
      setLoading(true)
      setError(null)
      const params: Record<string, string> = { limit: '30' }
      if (q) params.q = q
      const data = await api.style.clubAudit(params)
      setResults(data.items)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function fitColor(pct: number): string {
    if (pct >= 70) return 'bg-accent-lime/15 border-accent-lime/30 text-accent-lime'
    if (pct >= 50) return 'bg-accent-amber/15 border-accent-amber/30 text-accent-amber'
    return 'bg-accent-coral/15 border-accent-coral/30 text-accent-coral'
  }

  function fitLabel(pct: number): string {
    if (pct >= 70) return 'Strong Fit'
    if (pct >= 50) return 'Moderate Fit'
    return 'Weak Fit'
  }

  return (
    <div className="animate-slide-up">
      <SectionHeader
        title="CLUB AUDIT"
        subtitle="How well do players fit their current club's tactical style?"
      />

      <div className="mb-6">
        <SearchInput
          value={query}
          onChange={setQuery}
          onSubmit={() => search(query)}
          placeholder="Search by player name..."
        />
      </div>

      {loading && <Loading />}
      {error && <ErrorMessage message={error} onRetry={() => search(query)} />}

      {!loading && !error && (
        <div>
          <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-3">
            {results.length} audit entries · sorted by match quality (worst first)
          </div>

          {results.length === 0 && <Empty message="No audit data found" />}

          <div className="space-y-2">
            {results.map((r, i) => {
              const pct = Number(r.match_percentage || 0)
              return (
                <div key={i} className="card p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-3">
                        <span className="font-semibold text-text-primary">{r.full_name}</span>
                        <Badge label={r.position || '—'} />
                        <Badge label={r.match_role_group || '—'} variant="default" />
                      </div>
                      <div className="text-[10px] font-mono text-text-muted mt-2">
                        {r.player_season}
                      </div>

                      <div className="mt-3 flex items-center gap-2">
                        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-text-muted shrink-0">
                          <path d="M3 13L13 3M13 3H6M13 3v7" />
                        </svg>
                        <span className="text-sm text-text-secondary">
                          {r.current_club}
                        </span>
                        <span className="text-text-muted mx-1">→</span>
                        <span className="text-sm text-text-primary">
                          {r.team_name}
                        </span>
                        {r.team_country && (
                          <span className="text-[10px] font-mono text-text-muted">({r.team_country})</span>
                        )}
                      </div>

                      {r.team_style_cluster_label && (
                        <div className="text-[10px] text-text-muted mt-1.5 font-mono">
                          Style: {r.team_style_cluster_label}
                        </div>
                      )}
                    </div>

                    <div className="shrink-0 text-right">
                      <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded border ${fitColor(pct)}`}>
                        <span className="font-mono text-xl font-bold">{pct.toFixed(1)}%</span>
                      </div>
                      <div className="text-[10px] font-mono text-text-muted mt-1.5">{fitLabel(pct)}</div>
                      <div className="text-[10px] font-mono text-text-muted">
                        Dist: {fmt(r.style_distance)}
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
