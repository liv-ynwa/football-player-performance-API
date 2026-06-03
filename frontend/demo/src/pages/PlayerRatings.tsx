import { useState, useEffect } from 'react'
import { api, fmt } from '../api'
import { SectionHeader, SearchInput, ScoreBar, Badge, Loading, ErrorMessage, Empty } from '../components/ui'

export default function PlayerRatings() {
  const [query, setQuery] = useState('')
  const [players, setPlayers] = useState<any[]>([])
  const [selected, setSelected] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { search('') }, [])

  async function search(q: string) {
    try {
      setLoading(true)
      setError(null)
      const params: Record<string, string> = { limit: '30' }
      if (q) params.q = q
      const data = await api.players(params)
      setPlayers(data.items)
      setSelected(null)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const scoreFields = [
    { key: 'attack_score_score_pct', label: 'Attack' },
    { key: 'assist_score_score_pct', label: 'Assist' },
    { key: 'conceded_score_score_pct', label: 'Conceded' },
    { key: 'foul_card_score_score_pct', label: 'Foul/Card' },
    { key: 'goalkeeper_score_score_pct', label: 'Goalkeeper' },
    { key: 'appearance_score_score_pct', label: 'Appearance' },
  ]

  return (
    <div className="animate-slide-up">
      <SectionHeader title="PLAYER RATINGS" subtitle="Base model performance scores (0-10 scale)" />

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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* Player list */}
          <div className="lg:col-span-2 space-y-1.5">
            <div className="text-[10px] font-mono text-text-muted tracking-wider mb-3 uppercase">
              {players.length} players
            </div>
            {players.length === 0 && <Empty message="No players found" />}
            {players.map((p, i) => {
              const isSelected = selected?.player_id === p.player_id && selected?.season === p.season
              return (
                <div
                  key={`${p.player_id}-${p.season}-${i}`}
                  onClick={() => setSelected(p)}
                  className={`card card-hover p-4 cursor-pointer ${
                    isSelected ? 'border-accent-amber bg-bg-hover' : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="min-w-0">
                      <div className="font-semibold text-text-primary truncate">{p.full_name}</div>
                      <div className="text-[10px] font-mono text-text-muted mt-1 truncate">
                        {p.position} · {p['current club']} · {p.league} · {p.season}
                      </div>
                    </div>
                    <div className="text-right shrink-0 ml-4">
                      <div className="font-mono text-2xl font-bold text-accent-amber">
                        {p.rating_display_score_pct != null ? Number(p.rating_display_score_pct).toFixed(1) : '—'}
                      </div>
                      <div className="text-[9px] font-mono text-text-muted tracking-wider">RATING</div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Detail panel */}
          <div>
            {selected ? (
              <div className="card p-6 sticky top-8">
                <h3 className="font-display text-3xl text-text-primary tracking-wide">
                  {selected.full_name}
                </h3>
                <div className="flex flex-wrap gap-1.5 mt-3">
                  <Badge label={selected.position_group || selected.position} />
                  <Badge label={selected.league} />
                  <Badge label={selected.season} />
                </div>
                <div className="mt-2 text-sm text-text-secondary">{selected['current club']}</div>

                <div className="mt-6 space-y-2.5">
                  <div className="text-[10px] font-mono text-text-muted uppercase tracking-widest mb-3">
                    Score Breakdown
                  </div>
                  {scoreFields.map(({ key, label }) => {
                    const val = selected[key]
                    return val != null ? (
                      <ScoreBar key={key} label={label} value={Number(val)} />
                    ) : null
                  })}
                </div>

                <div className="mt-6 pt-5 border-t border-border">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    {[
                      ['Minutes', selected.minutes_played_overall],
                      ['Apps', selected.appearances_overall],
                      ['Goals', selected.goals_overall],
                      ['Assists', selected.assists_overall],
                      ['Clean Sheets', selected.clean_sheets_overall],
                      ['Yellow Cards', selected.yellow_cards_overall],
                    ].map(([label, val]) => (
                      <div key={label as string}>
                        <div className="text-text-muted font-mono text-[10px] uppercase tracking-wider">{label}</div>
                        <div className="text-text-primary font-mono mt-0.5">{fmt(val)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <Empty message="Select a player to view score breakdown" />
            )}
          </div>
        </div>
      )}
    </div>
  )
}
