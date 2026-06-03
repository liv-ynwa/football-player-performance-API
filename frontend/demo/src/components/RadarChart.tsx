interface RadarChartProps {
  labels: string[]
  values: number[]
  maxValue?: number
  size?: number
  color?: string
  fillOpacity?: number
  comparison?: number[]
  comparisonColor?: string
}

export default function RadarChart({
  labels,
  values,
  maxValue = 100,
  size = 260,
  color = '#e8a838',
  fillOpacity = 0.15,
  comparison,
  comparisonColor = '#42d4c8',
}: RadarChartProps) {
  const cx = size / 2
  const cy = size / 2
  const r = size / 2 - 44
  const n = labels.length

  if (n < 3) return null

  function getPoint(index: number, value: number): [number, number] {
    const angle = (Math.PI * 2 * index) / n - Math.PI / 2
    const ratio = Math.min(Math.max(value, 0), maxValue) / maxValue
    return [cx + r * ratio * Math.cos(angle), cy + r * ratio * Math.sin(angle)]
  }

  function polygon(vals: number[]): string {
    return (
      vals.map((v, i) => {
        const [x, y] = getPoint(i, v)
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
      }).join(' ') + 'Z'
    )
  }

  const levels = [0.25, 0.5, 0.75, 1.0]

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="mx-auto">
      {/* Grid polygons */}
      {levels.map((level) => (
        <path
          key={level}
          d={polygon(Array(n).fill(maxValue * level))}
          fill="none"
          stroke="#262a3d"
          strokeWidth={level === 1 ? 1.2 : 0.7}
        />
      ))}

      {/* Axis lines */}
      {Array.from({ length: n }, (_, i) => {
        const [x, y] = getPoint(i, maxValue)
        return (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={x}
            y2={y}
            stroke="#262a3d"
            strokeWidth={0.7}
          />
        )
      })}

      {/* Comparison polygon */}
      {comparison && (
        <path
          d={polygon(comparison)}
          fill={comparisonColor}
          fillOpacity={0.06}
          stroke={comparisonColor}
          strokeWidth={1.2}
          strokeDasharray="4,3"
        />
      )}

      {/* Data polygon */}
      <path
        d={polygon(values)}
        fill={color}
        fillOpacity={fillOpacity}
        stroke={color}
        strokeWidth={2}
        className="transition-all duration-500"
      />

      {/* Data points */}
      {values.map((v, i) => {
        const [x, y] = getPoint(i, v)
        return (
          <circle
            key={i}
            cx={x}
            cy={y}
            r={3}
            fill={color}
            className="transition-all duration-500"
          />
        )
      })}

      {/* Labels */}
      {labels.map((label, i) => {
        const angle = (Math.PI * 2 * i) / n - Math.PI / 2
        const lx = cx + (r + 28) * Math.cos(angle)
        const ly = cy + (r + 28) * Math.sin(angle)
        const anchor =
          Math.abs(Math.cos(angle)) < 0.15
            ? 'middle'
            : Math.cos(angle) > 0
              ? 'start'
              : 'end'
        return (
          <text
            key={i}
            x={lx}
            y={ly}
            textAnchor={anchor}
            dominantBaseline="central"
            fill="#9b9a94"
            fontSize={9}
            fontFamily="'JetBrains Mono', monospace"
          >
            {label}
          </text>
        )
      })}
    </svg>
  )
}
