import { useState, useEffect } from 'react'
import { api, fmt } from '../api'
import { SectionHeader, SearchInput, TabBar, Badge, Loading, ErrorMessage, Empty } from '../components/ui'
import RadarChart from '../components/RadarChart'

const ROLES = [
  { id: '', label: 'All' },
  { id: 'FW', label: 'Forward' },
  { id: 'MF', label: 'Midfield' },
  { id: 'DF', label: 'Defender' },
  { id: 'GK', label: 'Keeper' },
]

const LATENT_DIMS = Array.from({ length: 8 }, (_, i) => `Dim ${i + 1}`)

export default function PlayerStyles() {
  const [role, setRole] = useState('')
  const [query, setQuery] = useState('')
  const [clusters, setClusters] = useState<any[]>([])
  const [players, setPlayers] = useState<any[]>([])
  const [selected, setSelected] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { loadData() }, [role])

  async function loadData() {
    try {
      setLoading(true)
      setError(null)
      const params: Record<string, string> = { limit: '30' }
      if (role) params.role = role
      if (query) params.q = query
      const [clusterRes, playerRes] = await Promise.all([
        api.style.playerClusters(role || undefined),
        api.style.searchPlayerStyles(params),
      ])
      setClusters(clusterRes.items)
      setPlayers(playerRes.items)
      setSelected(null)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function search() {
    const params: Record<string, string> = { limit: '30' }
    if (role) params.role = role
    if (query) params.q = query
    api.style.searchPlayerStyles(params).then((data) => {
      setPlayers(data.items)
      setSelected(null)
    }).catch((err) => setError(err.message))
  }

  function getLatentValues(p: any): number[] {
    return Array.from({ length: 8 }, (_, i) => {
      const v = p[`latent_style_dim_${i + 1}`]
      return v != null ? Number(v) : 0
    })
  }

  if (loading) return <Loading />
  if (error) return <ErrorMessage message={error} onRetry={loadData} />

  return (
    <div className="animate-slide-up">
      <SectionHeader title="PLAYER STYLES" subtitle="Style clustering and latent style profiles" />

      <div className="flex flex-wrap gap-4 mb-6">
        <TabBar tabs={ROLES} active={role} onChange={setRole} />
        <div className="flex-1 min-w-[220px]">
          <SearchInput value={query} onChange={setQuery} onSubmit={search} placeholder="Search player..." />
        </div>
      </div>

      {/* Clusters */}
      {clusters.length > 0 && (
        <div className="mb-6">
          <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-3">
            {clusters[0]?.role_name || 'All'} Clusters
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {clusters.map((c) => (
              <div key={`${c.role}-${c.cluster_id}`} className="card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-7 h-7 rounded bg-accent-amber/15 text-accent-amber font-mono text-xs flex items-center justify-center font-bold">
                    {c.cluster_id}
                  </div>
                  <span className="font-mono text-[10px] text-text-muted">{c.player_count} players</span>
                </div>
                <div className="text-xs text-text-secondary leading-relaxed line-clamp-2">
                  {c.style_label || c.feature_signature}
                </div>
                {c.representative_players && (
                  <div className="mt-2 text-[10px] font-mono text-text-muted truncate">
                    {c.representative_players}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Players + Detail */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
        <div className="lg:col-span-2 space-y-1.5">
          <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-2">
            {players.length} players
          </div>
          {players.length === 0 && <Empty message="No players found" />}
          {players.map((p, i) => {
            const isSelected = selected?.player_id === p.player_id
            return (
              <div
                key={`${p.player_id}-${i}`}
                onClick={() => setSelected(p)}
                className={`card card-hover p-3.5 cursor-pointer ${
                  isSelected ? 'border-accent-amber bg-bg-hover' : ''
                }`}
              >
                <div className="font-semibold text-sm text-text-primary">{p.full_name}</div>
                <div className="text-[10px] font-mono text-text-muted mt-1">
                  {p.position} · {p['Current Club']} · {p.season}
                </div>
                <div className="flex gap-1.5 mt-2">
                  <Badge label={p.role} />
                  <Badge label={`Cluster ${p.cluster_id}`} variant="warning" />
                </div>
              </div>
            )
          })}
        </div>

        <div className="lg:col-span-3">
          {selected ? (
            <div className="card p-6 sticky top-8">
              <h3 className="font-display text-3xl text-text-primary tracking-wide">
                {selected.full_name}
              </h3>
              <div className="flex flex-wrap gap-1.5 mt-2">
                <Badge label={selected.position} />
                <Badge label={selected['Current Club']} />
                <Badge label={`Cluster ${selected.cluster_id}`} variant="warning" />
              </div>

              {selected.main_style_label && (
                <div className="mt-4 p-3 bg-bg-deep rounded-lg">
                  <div className="text-[10px] font-mono text-accent-amber tracking-wider uppercase mb-1">Main Style</div>
                  <div className="text-sm text-text-primary">{selected.main_style_label}</div>
                  {selected.secondary_tendencies && (
                    <>
                      <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mt-3 mb-1">Secondary</div>
                      <div className="text-xs text-text-secondary">{selected.secondary_tendencies}</div>
                    </>
                  )}
                </div>
              )}

              <div className="mt-6">
                <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-3">
                  Latent Style Dimensions
                </div>
                <RadarChart
                  labels={LATENT_DIMS}
                  values={getLatentValues(selected)}
                  maxValue={100}
                  size={280}
                />
              </div>

              <div className="mt-4 grid grid-cols-3 gap-3 text-center">
                <div>
                  <div className="text-text-muted font-mono text-[10px]">Confidence</div>
                  <div className="font-mono text-sm text-text-primary">{fmt(selected.style_confidence)}%</div>
                </div>
                <div>
                  <div className="text-text-muted font-mono text-[10px]">Mix Index</div>
                  <div className="font-mono text-sm text-text-primary">{fmt(selected.style_mix_index)}%</div>
                </div>
                <div>
                  <div className="text-text-muted font-mono text-[10px]">Minutes</div>
                  <div className="font-mono text-sm text-text-primary">{fmt(selected.minutes_played_overall)}</div>
                </div>
              </div>
            </div>
          ) : (
            <Empty message="Select a player to view style profile" />
          )}
        </div>
      </div>
    </div>
  )
}
