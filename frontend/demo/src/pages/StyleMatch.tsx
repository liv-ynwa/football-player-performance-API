import { useState, useEffect } from 'react'
import { api, fmt, dimLabel } from '../api'
import { SectionHeader, SearchInput, Badge, Loading, ErrorMessage, Empty } from '../components/ui'

export default function StyleMatch() {
  const [playerQuery, setPlayerQuery] = useState('Oscar')
  const [teamQuery, setTeamQuery] = useState('Basel')
  const [players, setPlayers] = useState<any[]>([])
  const [teams, setTeams] = useState<any[]>([])
  const [matches, setMatches] = useState<any[]>([])
  const [matchMode, setMatchMode] = useState<'player' | 'team' | null>(null)
  const [selectedPlayer, setSelectedPlayer] = useState<any>(null)
  const [selectedTeam, setSelectedTeam] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    searchPlayers('Oscar')
    searchTeams('Basel')
  }, [])

  async function searchPlayers(q: string) {
    try {
      const data = await api.style.searchPlayers({ q, limit: '12' })
      setPlayers(data.items)
    } catch {}
  }

  async function searchTeams(q: string) {
    try {
      const data = await api.style.searchTeams({ q, limit: '12' })
      setTeams(data.items)
    } catch {}
  }

  async function loadPlayerMatches(player: any) {
    setSelectedPlayer(player)
    setSelectedTeam(null)
    setMatchMode('player')
    setLoading(true)
    try {
      const data = await api.style.playerTeamMatches(player.player_id, {
        season: player.season,
        realistic: 'true',
        limit: '10',
      })
      setMatches(data.items)
    } catch {
      setMatches([])
    } finally {
      setLoading(false)
    }
  }

  async function loadTeamMatches(team: any) {
    setSelectedTeam(team)
    setSelectedPlayer(null)
    setMatchMode('team')
    setLoading(true)
    try {
      const data = await api.style.teamPlayerMatches(team.team_row_id, { limit: '10' })
      setMatches(data.items)
    } catch {
      setMatches([])
    } finally {
      setLoading(false)
    }
  }

  function matchColor(pct: number): string {
    if (pct >= 70) return 'text-accent-lime'
    if (pct >= 50) return 'text-accent-amber'
    return 'text-accent-coral'
  }

  return (
    <div className="animate-slide-up">
      <SectionHeader title="STYLE MATCH" subtitle="Player-to-team and team-to-player style fit analysis" />

      {/* Search panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded bg-accent-amber/15 text-accent-amber font-mono text-[10px] flex items-center justify-center font-bold">01</div>
            <span className="font-display text-lg text-text-primary tracking-wide">PLAYER TO CLUBS</span>
          </div>
          <SearchInput
            value={playerQuery}
            onChange={setPlayerQuery}
            onSubmit={() => searchPlayers(playerQuery)}
            placeholder="Search player..."
          />
        </div>
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded bg-accent-cyan/15 text-accent-cyan font-mono text-[10px] flex items-center justify-center font-bold">02</div>
            <span className="font-display text-lg text-text-primary tracking-wide">CLUB TO PLAYERS</span>
          </div>
          <SearchInput
            value={teamQuery}
            onChange={setTeamQuery}
            onSubmit={() => searchTeams(teamQuery)}
            placeholder="Search team..."
          />
        </div>
      </div>

      {/* Three-column workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.5fr_1fr] gap-4">
        {/* Player list */}
        <div className="space-y-1.5">
          <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-2">
            Players ({players.length})
          </div>
          {players.map((p, i) => (
            <div
              key={`${p.player_id}-${i}`}
              onClick={() => loadPlayerMatches(p)}
              className={`card card-hover p-3 cursor-pointer text-sm ${
                selectedPlayer?.player_id === p.player_id ? 'border-accent-amber bg-bg-hover' : ''
              }`}
            >
              <div className="font-semibold text-text-primary">{p.full_name}</div>
              <div className="text-[10px] font-mono text-text-muted mt-1">
                {p.position} · {p['Current Club']} · {p.season}
              </div>
            </div>
          ))}
          {players.length === 0 && <Empty message="Search for players" />}
        </div>

        {/* Match results */}
        <div>
          <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-2">
            Matches ({matches.length})
          </div>
          {loading && <Loading />}
          {!loading && matches.length === 0 && (
            <Empty message="Select a player or team to see style matches" />
          )}
          {!loading && matches.map((m, i) => {
            const title = matchMode === 'team' ? m.full_name : m.team_name
            const subtitle = matchMode === 'team'
              ? `${m.position || ''} · ${m.current_club || ''} · ${m.player_season || ''}`
              : `${m.team_country || ''} · ${m.team_season || ''} · ${m.team_style_cluster_label || ''}`
            const best = String(m.best_fit_dimensions || '').split(', ').filter(Boolean)
            const weak = String(m.weak_fit_dimensions || '').split(', ').filter(Boolean)
            const pct = Number(m.match_percentage || 0)

            return (
              <div key={i} className="card p-4 mb-2">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="font-semibold text-text-primary text-sm">{title}</div>
                    <div className="text-[10px] font-mono text-text-muted mt-1 truncate">{subtitle}</div>
                  </div>
                  <div className={`font-mono text-2xl font-bold shrink-0 ${matchColor(pct)}`}>
                    {pct.toFixed(1)}
                  </div>
                </div>
                {(best.length > 0 || weak.length > 0) && (
                  <div className="flex flex-wrap gap-1 mt-3">
                    {best.map((d) => (
                      <Badge key={d} label={dimLabel(d)} variant="success" />
                    ))}
                    {weak.map((d) => (
                      <Badge key={d} label={dimLabel(d)} variant="danger" />
                    ))}
                  </div>
                )}
                {m.is_current_club_match === 1 && (
                  <Badge label="Current Club" variant="info" />
                )}
              </div>
            )
          })}
        </div>

        {/* Team list */}
        <div className="space-y-1.5">
          <div className="text-[10px] font-mono text-text-muted tracking-wider uppercase mb-2">
            Teams ({teams.length})
          </div>
          {teams.map((t, i) => (
            <div
              key={`${t.team_row_id}-${i}`}
              onClick={() => loadTeamMatches(t)}
              className={`card card-hover p-3 cursor-pointer text-sm ${
                selectedTeam?.team_row_id === t.team_row_id ? 'border-accent-cyan bg-bg-hover' : ''
              }`}
            >
              <div className="font-semibold text-text-primary">{t.team_name}</div>
              <div className="text-[10px] font-mono text-text-muted mt-1">
                {t.country} · {t.season} · {t.style_cluster_label || ''}
              </div>
            </div>
          ))}
          {teams.length === 0 && <Empty message="Search for teams" />}
        </div>
      </div>
    </div>
  )
}
