import { type ReactNode, type FormEvent } from 'react'

/* ── Stat Card ────────────────────────────────────────────────────── */

export function StatCard({
  label,
  value,
  subtitle,
  status,
}: {
  label: string
  value: string | number
  subtitle?: string
  status?: 'ok' | 'error'
}) {
  return (
    <div className="card p-5">
      <div className="text-text-muted font-mono text-[10px] uppercase tracking-widest">{label}</div>
      <div className="mt-2 font-display text-3xl text-text-primary tracking-wide">{value}</div>
      {subtitle && <div className="mt-1 text-xs text-text-secondary">{subtitle}</div>}
      {status && (
        <div
          className={`mt-2.5 inline-flex items-center gap-1.5 font-mono text-[10px] tracking-wider ${
            status === 'ok' ? 'text-accent-lime' : 'text-accent-coral'
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              status === 'ok' ? 'bg-accent-lime animate-pulse' : 'bg-accent-coral'
            }`}
          />
          {status === 'ok' ? 'ONLINE' : 'OFFLINE'}
        </div>
      )}
    </div>
  )
}

/* ── Score Bar ─────────────────────────────────────────────────────── */

export function ScoreBar({
  label,
  value,
  max = 10,
}: {
  label: string
  value: number
  max?: number
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  const barColor =
    value >= 7 ? '#a6e22e' : value >= 4 ? '#e8a838' : '#e8573a'

  return (
    <div className="flex items-center gap-3">
      <span className="w-24 text-xs text-text-secondary font-mono truncate">{label}</span>
      <div className="flex-1 h-[7px] bg-bg-deep rounded-full overflow-hidden">
        <div
          className="h-full rounded-full animate-bar-fill"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
      <span className="font-mono text-sm font-semibold text-text-primary w-10 text-right">
        {value.toFixed(1)}
      </span>
    </div>
  )
}

/* ── Badge ─────────────────────────────────────────────────────────── */

export function Badge({
  label,
  variant = 'default',
}: {
  label: string
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info'
}) {
  const colors: Record<string, string> = {
    default: 'bg-bg-hover text-text-secondary border-border',
    success: 'bg-accent-lime/10 text-accent-lime border-accent-lime/20',
    warning: 'bg-accent-amber/10 text-accent-amber border-accent-amber/20',
    danger: 'bg-accent-coral/10 text-accent-coral border-accent-coral/20',
    info: 'bg-accent-cyan/10 text-accent-cyan border-accent-cyan/20',
  }
  return (
    <span className={`inline-block px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider border rounded ${colors[variant]}`}>
      {label}
    </span>
  )
}

/* ── Search Input ──────────────────────────────────────────────────── */

export function SearchInput({
  value,
  onChange,
  onSubmit,
  placeholder,
}: {
  value: string
  onChange: (v: string) => void
  onSubmit?: () => void
  placeholder?: string
}) {
  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    onSubmit?.()
  }
  return (
    <form onSubmit={handleSubmit} className="relative">
      <svg
        className="absolute left-3.5 top-1/2 -translate-y-1/2 text-text-muted"
        width="14"
        height="14"
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <circle cx="7" cy="7" r="5" />
        <line x1="11" y1="11" x2="14" y2="14" />
      </svg>
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || 'Search...'}
        className="w-full bg-bg-deep border border-border rounded-lg pl-10 pr-4 py-2.5 text-text-primary placeholder-text-muted font-mono text-sm outline-none focus:border-accent-amber/60 transition-colors"
      />
    </form>
  )
}

/* ── Tab Bar ───────────────────────────────────────────────────────── */

export function TabBar({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: string; label: string }[]
  active: string
  onChange: (id: string) => void
}) {
  return (
    <div className="flex gap-1 bg-bg-panel rounded-lg p-1 border border-border">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`px-4 py-2 rounded-md text-xs font-mono uppercase tracking-wider transition-all ${
            active === tab.id
              ? 'bg-accent-amber/15 text-accent-amber'
              : 'text-text-muted hover:text-text-secondary'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

/* ── Section Header ────────────────────────────────────────────────── */

export function SectionHeader({
  title,
  subtitle,
  right,
}: {
  title: string
  subtitle?: string
  right?: ReactNode
}) {
  return (
    <div className="flex items-end justify-between gap-4 mb-6">
      <div>
        <h1 className="font-display text-4xl text-text-primary tracking-wide">{title}</h1>
        {subtitle && <p className="text-text-secondary text-sm mt-1">{subtitle}</p>}
      </div>
      {right}
    </div>
  )
}

/* ── Loading / Error ───────────────────────────────────────────────── */

export function Loading() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="flex items-center gap-3 text-text-muted font-mono text-sm">
        <div className="w-2 h-2 rounded-full bg-accent-amber animate-pulse" />
        Loading data...
      </div>
    </div>
  )
}

export function ErrorMessage({
  message,
  onRetry,
}: {
  message: string
  onRetry?: () => void
}) {
  return (
    <div className="text-center py-16">
      <p className="text-accent-coral font-mono text-sm mb-4">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-5 py-2 bg-bg-card border border-border rounded-lg text-sm text-text-secondary hover:text-text-primary hover:border-border-light transition-all"
        >
          Retry
        </button>
      )}
    </div>
  )
}

export function Empty({ message }: { message: string }) {
  return (
    <div className="border border-dashed border-border rounded-lg p-8 text-center text-text-muted font-mono text-xs">
      {message}
    </div>
  )
}
