const API_BASE = import.meta.env.VITE_API_BASE || '';

async function fetchApi<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(await res.text() || res.statusText);
  return res.json();
}

export interface CountResponse {
  count: number;
  items: Record<string, any>[];
}

export const api = {
  health: () => fetchApi<Record<string, any>>('/health'),
  metadata: () => fetchApi<Record<string, any>>('/metadata'),
  players: (params: Record<string, string>) => {
    const qs = new URLSearchParams(params).toString();
    return fetchApi<CountResponse>(`/players?${qs}`);
  },
  player: (id: number) => fetchApi<CountResponse>(`/players/${id}`),

  style: {
    health: () => fetchApi<Record<string, any>>('/style/health'),
    metadata: () => fetchApi<Record<string, any>>('/style/metadata'),
    stats: () => fetchApi<Record<string, any>>('/style/stats'),

    searchPlayers: (params: Record<string, string>) => {
      const qs = new URLSearchParams(params).toString();
      return fetchApi<CountResponse>(`/style/players/search?${qs}`);
    },
    playerTeamMatches: (playerId: number, params: Record<string, string>) => {
      const qs = new URLSearchParams(params).toString();
      return fetchApi<CountResponse>(`/style/players/${playerId}/team-matches?${qs}`);
    },
    searchTeams: (params: Record<string, string>) => {
      const qs = new URLSearchParams(params).toString();
      return fetchApi<CountResponse>(`/style/teams/search?${qs}`);
    },
    teamPlayerMatches: (teamRowId: number, params: Record<string, string>) => {
      const qs = new URLSearchParams(params).toString();
      return fetchApi<CountResponse>(`/style/teams/${teamRowId}/player-matches?${qs}`);
    },
    clubAudit: (params: Record<string, string>) => {
      const qs = new URLSearchParams(params).toString();
      return fetchApi<CountResponse>(`/style/current-club-audit?${qs}`);
    },

    searchPlayerStyles: (params: Record<string, string>) => {
      const qs = new URLSearchParams(params).toString();
      return fetchApi<CountResponse>(`/style/player-styles/search?${qs}`);
    },
    playerStyleProfile: (playerId: number) =>
      fetchApi<CountResponse>(`/style/players/${playerId}/style-profile`),
    playerClusters: (role?: string) => {
      const qs = role ? `?role=${role}` : '';
      return fetchApi<CountResponse>(`/style/player-clusters${qs}`);
    },
    playerRoles: () => fetchApi<CountResponse>('/style/player-roles'),
    searchTeamStyles: (params: Record<string, string>) => {
      const qs = new URLSearchParams(params).toString();
      return fetchApi<CountResponse>(`/style/team-styles/search?${qs}`);
    },
    teamStyleProfile: (id: number) =>
      fetchApi<CountResponse>(`/style/teams/${id}/style-profile`),
    teamClusters: () => fetchApi<CountResponse>('/style/team-clusters'),
    dimensions: () => fetchApi<CountResponse>('/style/dimensions'),
  },
};

export function fmt(v: any, fallback = '—'): string {
  if (v === null || v === undefined || v === '') return fallback;
  if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
  return String(v);
}

export function dimLabel(s: string): string {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
