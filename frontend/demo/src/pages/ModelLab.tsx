import { useState, useEffect } from 'react'
import { api, fmt, dimLabel } from '../api'
import { SectionHeader, Badge, ScoreBar, Loading, ErrorMessage, Empty } from '../components/ui'

export default function ModelLab() {
  const [metadata, setMetadata] = useState<any>(null)
  const [styleMetadata, setStyleMetadata] = useState<any>(null)
  const [dimensions, setDimensions] = useState<any[]>([])
  const [roles, setRoles] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { loadAll() }, [])

  async function loadAll() {
    try {
      setLoading(true)
      setError(null)
      const [m, sm, d, r] = await Promise.allSettled([
        api.metadata(),
        api.style.metadata(),
        api.style.dimensions(),
        api.style.playerRoles(),
      ])
      if (m.status === 'fulfilled') setMetadata(m.value)
      if (sm.status === 'fulfilled') setStyleMetadata(sm.value)
      if (d.status === 'fulfilled') setDimensions(d.value.items)
      if (r.status === 'fulfilled') setRoles(r.value.items)
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
  const summary = styleMetadata?.summary || (styleMetadata?.scope ? styleMetadata : null)

  const dimGroups: Record<string, any[]> = {}
  dimensions.forEach((d) => {
    const key = d.dimension
    if (!dimGroups[key]) dimGroups[key] = []
    dimGroups[key].push(d)
  })

  return (
    <div className="animate-slide-up space-y-8">
      <SectionHeader title="MODEL LAB" subtitle="ML model performance, dimensions, and pipeline details" />

      {/* Base Model Metrics */}
      {models.length > 0 && (
        <div className="card p-6">
          <h2 className="font-display text-2xl text-text-primary tracking-wide mb-5">
            BASE MODEL METRICS
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  {['Target', 'RMSE', 'RMSE/Std', 'Std(y)', 'Shuffled RMSE', 'Train', 'Valid'].map((h) => (
                    <th key={h} className="text-left text-[10px] font-mono text-text-muted uppercase tracking-wider py-3 px-3">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.map((m: any) => {
                  const ratio = m.metrics?.rmse_over_std || 0
                  return (
                    <tr key={m.target} className="border-b border-border/50 hover:bg-bg-hover transition-colors">
                      <td className="py-3 px-3 font-mono text-text-primary text-xs">
                        {m.target.replace('base_', '')}
                      </td>
                      <td className="py-3 px-3 font-mono text-xs">{fmt(m.metrics?.rmse_valid)}</td>
                      <td className="py-3 px-3">
                        <span className={`font-mono text-xs font-semibold ${
                          ratio < 0.4 ? 'text-accent-lime' : ratio < 0.7 ? 'text-accent-amber' : 'text-accent-coral'
                        }`}>
                          {fmt(ratio)}
                        </span>
                      </td>
                      <td className="py-3 px-3 font-mono text-xs text-text-secondary">{fmt(m.metrics?.std_y_valid)}</td>
                      <td className="py-3 px-3 font-mono text-xs text-text-muted">{fmt(m.metrics?.rmse_valid_shuffled_y)}</td>
                      <td className="py-3 px-3 font-mono text-xs text-text-secondary">{m.n_train?.toLocaleString()}</td>
                      <td className="py-3 px-3 font-mono text-xs text-text-muted">{m.n_valid?.toLocaleString()}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Style Matching Dimension Quality */}
      {metricsArr.length > 0 && (
        <div className="card p-6">
          <h2 className="font-display text-2xl text-text-primary tracking-wide mb-5">
            STYLE DIMENSION QUALITY
          </h2>
          <div className="space-y-3">
            {metricsArr.map((m: any) => {
              const r2 = Number(m.r2 || 0)
              const active = r2 >= 0.1
              return (
                <div key={m.dimension} className="flex items-center gap-4">
                  <span className="w-36 text-xs font-mono text-text-secondary truncate">
                    {dimLabel(m.dimension)}
                  </span>
                  <div className="flex-1 h-2 bg-bg-deep rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full animate-bar-fill"
                      style={{
                        width: `${Math.max(2, r2 * 100)}%`,
                        backgroundColor: r2 > 0.3 ? '#a6e22e' : r2 > 0.1 ? '#e8a838' : '#e8573a',
                      }}
                    />
                  </div>
                  <span className="font-mono text-xs text-text-primary w-14 text-right">
                    R2 {r2.toFixed(3)}
                  </span>
                  <Badge label={active ? 'Active' : 'Excluded'} variant={active ? 'success' : 'danger'} />
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Experiment Summary */}
      {summary && (
        <div className="card p-6">
          <h2 className="font-display text-2xl text-text-primary tracking-wide mb-5">
            EXPERIMENT SUMMARY
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              ['Model', summary.model || '—'],
              ['Player Rows', summary.player_rows?.toLocaleString()],
              ['Team Rows', summary.team_rows?.toLocaleString()],
              ['CV Folds', summary.cv_folds],
              ['Overall R2', fmt(summary.overall_r2)],
              ['Overall RMSE', fmt(summary.overall_rmse)],
              ['Top N/Player', summary.top_n_per_player],
              ['Realistic Top N', summary.realistic_top_n_per_player],
            ].map(([label, value]) => (
              <div key={label as string} className="p-3 bg-bg-deep rounded-lg">
                <div className="text-[10px] font-mono text-text-muted uppercase tracking-wider">{label}</div>
                <div className="font-mono text-sm text-text-primary mt-1">{value ?? '—'}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dimension Feature Weights */}
      {Object.keys(dimGroups).length > 0 && (
        <div className="card p-6">
          <h2 className="font-display text-2xl text-text-primary tracking-wide mb-5">
            DIMENSION FEATURE WEIGHTS
          </h2>
          <div className="space-y-6">
            {Object.entries(dimGroups).map(([dim, features]) => (
              <div key={dim}>
                <div className="flex items-center gap-2 mb-3">
                  <Badge label={dimLabel(dim)} variant="info" />
                  <span className="text-[10px] font-mono text-text-muted">
                    {features.length} features
                  </span>
                </div>
                <div className="space-y-1.5">
                  {features.slice(0, 6).map((f, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="w-48 text-[10px] text-text-secondary font-mono truncate">
                        {f.feature_name}
                      </span>
                      <div className="flex-1 h-1.5 bg-bg-deep rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-accent-blue"
                          style={{ width: `${(Number(f.global_normalized_weight || 0)) * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-[10px] text-text-muted w-12 text-right">
                        {(Number(f.global_normalized_weight || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Player Style Pipeline */}
      {roles.length > 0 && (
        <div className="card p-6">
          <h2 className="font-display text-2xl text-text-primary tracking-wide mb-5">
            PLAYER STYLE PIPELINE
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {roles.map((r: any) => (
              <div key={r.role} className="p-4 bg-bg-deep rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <Badge label={r.role} variant="warning" />
                  <span className="text-sm text-text-primary font-semibold">{r.role_name}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {[
                    ['Players', r.player_rows?.toLocaleString()],
                    ['Clusters (k)', r.best_k],
                    ['PCA Dims', r.pca_components],
                    ['Features', r.selected_feature_count],
                    ['Silhouette', fmt(r.silhouette)],
                    ['Variance', fmt(r.explained_variance_ratio)],
                  ].map(([label, val]) => (
                    <div key={label as string}>
                      <span className="text-text-muted font-mono text-[10px]">{label}</span>
                      <div className="text-text-primary font-mono">{val ?? '—'}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
