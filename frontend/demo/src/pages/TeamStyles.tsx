import { useState, useEffect } from 'react'
import { api, dimLabel, fmt } from '../api'
import { SectionHeader, SearchInput, Badge, Loading, ErrorMessage, Empty } from '../components/ui'
import RadarChart from '../components/RadarChart'

const TEAM_DIMS = [
  'attacking_tempo',
  'possession_dominance',
  'high_pressing',
  'set_piece_reliance',
  'vertical_threat',
  'defensive_positioning',
  'wide_play',
  'physicality',
]

export default function TeamStyles() {
  const [query, setQuery] = useState('')
  const [clusters, setClusters] = useState<any[]>([])
  const [teams, setTeams] = useState<any[]>([])
  const [selected, setSelected] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { loadData() }, [])

  async function loadData() {
    try {
      setLoading(true)
      setError(null)
      const [clusterRes, teamRes] = await Promise.all([
        api.style.teamClusters(),
        api.style.searchTeamStyles({ limit: '30' }),
      ])
      setClusters(clusterRes.items)
      setTeams(teamRes.items)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function search() {
    const params: Record<string, string> = { limit: '30' }
    if (query) params.q = query
    api.style.searchTeamStyles(params).then((data) => {
      setTeams(data.items)
      setSelected(null)
    }).catch((err) => setError(err.message))
  }

  function getTeamDimValues(t: any): number[] {
    return TEAM_DIMS.map((d) => Number(t[d] || 0))
  }

  function getClusterAvgValues(c: any): number[] {
    return TEAM_DIMS.map((d) => Number(c[`avg_${d}`] || 0))
  }

  if (loading) return <Loading />
  if (error) return <ErrorMessage message={error} onRetry={loadData} />

  return (
    <div className="animate-slide-up">
      <SectionHeader title="TEAM STYLES" subtitle="Team tactical profiles and style clusters" />

      {/* Cluster overview */}
      {clusters.length > 0 && (
        <div className="mb-6">
          <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-3">
            Style Clusters ({clusters.length})
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {clusters.map((c) => (
              <div key={c.style_cluster_id} className="card p-4">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded bg-accent-cyan/15 text-accent-cyan font-mono text-sm flex items-center justify-center font-bold shrink-0">
                    {c.style_cluster_id}
                  </div>
                  <div className="min-w-0">
                    <div className="font-mono text-xs text-text-muted">{c.team_count} teams</div>
                    <div className="text-xs text-text-secondary mt-1 leading-relaxed">{c.label}</div>
                    {c.top_dimensions && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {String(c.top_dimensions).split(';').slice(0, 3).map((d: string) => (
                          <Badge key={d} label={d.trim()} variant="success" />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mb-5">
        <SearchInput value={query} onChange={setQuery} onSubmit={search} placeholder="Search team name..." />
      </div>

      {/* Teams + Detail */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
        <div className="lg:col-span-2 space-y-1.5">
          <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-2">
            {teams.length} teams
          </div>
          {teams.length === 0 && <Empty message="No teams found" />}
          {teams.map((t, i) => {
            const isSelected = selected?.source_team_rowid === t.source_team_rowid
            return (
              <div
                key={`${t.source_team_rowid}-${i}`}
                onClick={() => setSelected(t)}
                className={`card card-hover p-3.5 cursor-pointer ${
                  isSelected ? 'border-accent-cyan bg-bg-hover' : ''
                }`}
              >
                <div className="font-semibold text-sm text-text-primary">{t.team_name}</div>
                <div className="text-[10px] font-mono text-text-muted mt-1">
                  {t.country} · {t.season} · Quality: {fmt(t.style_data_quality)}
                </div>
                <div className="flex gap-1.5 mt-2">
                  <Badge label={t.style_cluster_label?.split(' ')[0] || `Cluster ${t.style_cluster_id}`} variant="info" />
                </div>
              </div>
            )
          })}
        </div>

        <div className="lg:col-span-3">
          {selected ? (
            <div className="card p-6 sticky top-8">
              <h3 className="font-display text-3xl text-text-primary tracking-wide">
                {selected.common_name || selected.team_name}
              </h3>
              <div className="flex flex-wrap gap-1.5 mt-2">
                <Badge label={selected.country} />
                <Badge label={selected.season} />
                <Badge label={`Cluster ${selected.style_cluster_id}`} variant="info" />
              </div>
              {selected.style_cluster_label && (
                <div className="mt-3 text-xs text-text-secondary">{selected.style_cluster_label}</div>
              )}

              <div className="mt-6">
                <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-3">
                  Style Profile
                </div>
                <RadarChart
                  labels={TEAM_DIMS.map((d) => dimLabel(d).replace('Dominance', 'Dom.').replace('Positioning', 'Pos.').replace('Reliance', 'Rel.'))}
                  values={getTeamDimValues(selected)}
                  maxValue={100}
                  size={300}
                  color="#42d4c8"
                  comparison={
                    clusters.find((c) => c.style_cluster_id === selected.style_cluster_id)
                      ? getClusterAvgValues(clusters.find((c) => c.style_cluster_id === selected.style_cluster_id))
                      : undefined
                  }
                  comparisonColor="#e8a838"
                />
                <div className="flex justify-center gap-6 mt-2 text-[10px] font-mono text-text-muted">
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 bg-accent-cyan inline-block" /> Team
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 bg-accent-amber inline-block border-t border-dashed border-accent-amber" /> Cluster Avg
                  </span>
                </div>
              </div>

              <div className="mt-6 space-y-2">
                <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-2">
                  Dimension Values
                </div>
                {TEAM_DIMS.map((d) => (
                  <div key={d} className="flex items-center gap-3">
                    <span className="w-28 text-xs text-text-secondary font-mono truncate">{dimLabel(d)}</span>
                    <div className="flex-1 h-1.5 bg-bg-deep rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-accent-cyan animate-bar-fill"
                        style={{ width: `${Number(selected[d] || 0)}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs text-text-primary w-10 text-right">
                      {fmt(selected[d])}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <Empty message="Select a team to view style profile" />
          )}
        </div>
      </div>
    </div>
  )
}
