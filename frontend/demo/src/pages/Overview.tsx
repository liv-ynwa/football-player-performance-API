import { useState, useEffect } from 'react'
import { api, dimLabel } from '../api'
import { StatCard, Badge, Loading, ErrorMessage } from '../components/ui'

export default function Overview() {
  const [health, setHealth] = useState<any>(null)
  const [styleHealth, setStyleHealth] = useState<any>(null)
  const [metadata, setMetadata] = useState<any>(null)
  const [styleMetadata, setStyleMetadata] = useState<any>(null)
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { loadAll() }, [])

  async function loadAll() {
    try {
      setLoading(true)
      setError(null)
      const [h, sh, m, sm, s] = await Promise.allSettled([
        api.health(), api.style.health(), api.metadata(), api.style.metadata(), api.style.stats(),
      ])
      if (h.status === 'fulfilled') setHealth(h.value)
      if (sh.status === 'fulfilled') setStyleHealth(sh.value)
      if (m.status === 'fulfilled') setMetadata(m.value)
      if (sm.status === 'fulfilled') setStyleMetadata(sm.value)
      if (s.status === 'fulfilled') setStats(s.value)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <Loading />
  if (error) return <ErrorMessage message={error} onRetry={loadAll} />

  const models = metadata?.models || []
  const metricsArr = Array.isArray(styleMetadata?.metrics) ? styleMetadata.metrics : []
  const activeDims = styleMetadata?.active_match_dimensions
    || (styleMetadata?.summary && JSON.parse(styleMetadata.summary.active_match_dimensions || '[]'))
    || []
  const excludedDims = styleMetadata?.excluded_match_dimensions
    || (styleMetadata?.summary && JSON.parse(styleMetadata.summary.excluded_match_dimensions || '[]'))
    || []

  return (
    <div className="space-y-8 animate-slide-up">
      {/* Hero */}
      <div className="relative overflow-hidden card p-8 pb-10">
        <div className="absolute inset-0 bg-gradient-to-br from-accent-amber/5 via-transparent to-accent-cyan/5" />
        <div className="relative">
          <div className="font-mono text-[10px] text-accent-amber tracking-[0.3em] uppercase mb-3">
            Open Source Football Analytics
          </div>
          <h1 className="font-display text-6xl md:text-7xl text-text-primary tracking-wider leading-none">
            FUTRIXMETRICS
          </h1>
          <p className="text-text-secondary mt-3 max-w-xl text-sm leading-relaxed">
            Base model API for football player performance scoring and style matching.
            Covering player ratings, style profiles, team-player fit analysis, and ML model quality metrics.
          </p>
          <div className="flex gap-3 mt-5">
            <Badge label="Base Model v1" variant="warning" />
            <Badge label={`${models.length} Trained Models`} variant="info" />
            <Badge label={styleHealth?.sample_mode ? 'Sample Data' : 'Full Data'} variant="default" />
          </div>
        </div>
      </div>

      {/* Health + Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          label="Base API"
          value={health?.status === 'ok' ? 'Online' : 'Offline'}
          status={health?.status === 'ok' ? 'ok' : 'error'}
        />
        <StatCard
          label="Style Engine"
          value={styleHealth?.status === 'ok' ? 'Online' : 'Offline'}
          status={styleHealth?.status === 'ok' ? 'ok' : 'error'}
        />
        <StatCard label="Player Profiles" value={stats?.player_style_profiles ?? '—'} />
        <StatCard label="Team Profiles" value={stats?.team_style_profiles ?? '—'} />
        <StatCard label="Predicted Players" value={stats?.predicted_players ?? '—'} />
        <StatCard label="Predicted Teams" value={stats?.predicted_teams ?? '—'} />
        <StatCard label="Realistic Matches" value={stats?.realistic_matches ?? '—'} />
        <StatCard label="Club Audit" value={stats?.club_audit_entries ?? '—'} />
      </div>

      {/* Base Model Performance */}
      {models.length > 0 && (
        <div className="card p-6">
          <h2 className="font-display text-2xl text-text-primary tracking-wide mb-6">
            BASE MODEL PERFORMANCE
          </h2>
          <div className="space-y-3">
            {models.map((m: any) => {
              const name = m.target.replace('base_', '').replace(/_/g, ' ')
              const quality = 1 - (m.metrics?.rmse_over_std || 0)
              const qPct = Math.max(0, quality * 100)
              return (
                <div key={m.target} className="flex items-center gap-4">
                  <span className="w-28 text-xs text-text-secondary font-mono capitalize truncate">{name}</span>
                  <div className="flex-1 h-2 bg-bg-deep rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full animate-bar-fill"
                      style={{
                        width: `${qPct}%`,
                        backgroundColor: quality > 0.6 ? '#a6e22e' : quality > 0.3 ? '#e8a838' : '#e8573a',
                      }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-text-muted w-20 text-right">
                    RMSE/s {(m.metrics?.rmse_over_std || 0).toFixed(2)}
                  </span>
                  <span className="font-mono text-[10px] text-text-secondary w-20 text-right">
                    n={m.n_train?.toLocaleString()}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Style Dimensions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card p-6">
          <h2 className="font-display text-2xl text-text-primary tracking-wide mb-4">
            ACTIVE DIMENSIONS
          </h2>
          <div className="space-y-2">
            {(typeof activeDims === 'string' ? JSON.parse(activeDims) : activeDims).map((d: string) => {
              const metric = metricsArr.find((m: any) => m.dimension === d)
              return (
                <div key={d} className="flex items-center gap-3">
                  <Badge label={dimLabel(d)} variant="success" />
                  {metric && (
                    <span className="font-mono text-[10px] text-text-muted">
                      R2={Number(metric.r2).toFixed(3)} RMSE={Number(metric.rmse).toFixed(1)}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        <div className="card p-6">
          <h2 className="font-display text-2xl text-text-primary tracking-wide mb-4">
            EXCLUDED DIMENSIONS
          </h2>
          <div className="space-y-2">
            {(typeof excludedDims === 'string' ? JSON.parse(excludedDims) : excludedDims).map((d: string) => {
              const metric = metricsArr.find((m: any) => m.dimension === d)
              return (
                <div key={d} className="flex items-center gap-3">
                  <Badge label={dimLabel(d)} variant="danger" />
                  {metric && (
                    <span className="font-mono text-[10px] text-text-muted">
                      R2={Number(metric.r2).toFixed(3)} — below threshold
                    </span>
                  )}
                </div>
              )
            })}
            {excludedDims.length === 0 && (
              <p className="text-text-muted font-mono text-xs">None excluded</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
