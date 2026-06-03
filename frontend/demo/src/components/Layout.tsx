import { type ReactNode } from 'react'

const NAV = [
  { id: 'overview', label: 'Overview', icon: M_Grid },
  { id: 'ratings', label: 'Player Ratings', icon: M_Star },
  { id: 'player-styles', label: 'Player Styles', icon: M_User },
  { id: 'team-styles', label: 'Team Styles', icon: M_Shield },
  { id: 'match', label: 'Style Match', icon: M_Link },
  { id: 'audit', label: 'Club Audit', icon: M_Search },
  { id: 'lab', label: 'Model Lab', icon: M_Flask },
]

function M_Grid() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="1.5" y="1.5" width="5" height="5" rx="1" />
      <rect x="9.5" y="1.5" width="5" height="5" rx="1" />
      <rect x="1.5" y="9.5" width="5" height="5" rx="1" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
    </svg>
  )
}

function M_Star() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M8 1.5l2 4.1 4.5.6-3.3 3.2.8 4.5L8 11.6l-4 2.3.8-4.5L1.5 6.2l4.5-.6z" />
    </svg>
  )
}

function M_User() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="8" cy="5" r="3" />
      <path d="M2.5 14.5c0-3 2.5-5 5.5-5s5.5 2 5.5 5" />
    </svg>
  )
}

function M_Shield() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M8 1.5L2.5 4v4c0 3.5 2.3 5.5 5.5 6.5 3.2-1 5.5-3 5.5-6.5V4z" />
    </svg>
  )
}

function M_Link() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M6 10l4-4" />
      <path d="M9 3l2.5 2.5M4.5 10.5L7 13" />
      <rect x="1" y="8.5" width="5" height="5" rx="1" transform="rotate(-45 3.5 11)" />
      <rect x="7.5" y="2" width="5" height="5" rx="1" transform="rotate(-45 10 4.5)" />
    </svg>
  )
}

function M_Search() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="7" cy="7" r="4.5" />
      <line x1="10.5" y1="10.5" x2="14" y2="14" />
    </svg>
  )
}

function M_Flask() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M5.5 1.5h5M6 1.5v5l-4 7.5h12L10 6.5V1.5" />
      <path d="M4 11.5h8" strokeDasharray="2 2" />
    </svg>
  )
}

interface LayoutProps {
  children: ReactNode
  activePage: string
  onNavigate: (page: string) => void
}

export default function Layout({ children, activePage, onNavigate }: LayoutProps) {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-[260px] shrink-0 border-r border-border bg-bg-panel flex flex-col sticky top-0 h-screen">
        <div className="p-6 pb-2">
          <div className="font-display text-2xl tracking-wider text-accent-amber">FUTRIX</div>
          <div className="font-mono text-[10px] text-text-muted tracking-widest mt-0.5">
            METRICS · OPEN MODEL
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
          {NAV.map(({ id, label, icon: Icon }) => {
            const active = activePage === id
            return (
              <button
                key={id}
                onClick={() => onNavigate(id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-all ${
                  active
                    ? 'bg-accent-amber/10 text-accent-amber border-l-2 border-accent-amber -ml-px'
                    : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
                }`}
              >
                <Icon />
                <span className={active ? 'font-semibold' : ''}>{label}</span>
              </button>
            )
          })}
        </nav>

        <div className="p-4 border-t border-border">
          <div className="text-[10px] font-mono text-text-muted leading-relaxed">
            Base Model v1<br />
            Open Source · Apache 2.0
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 min-w-0 p-8 animate-fade-in">
        {children}
      </main>
    </div>
  )
}
