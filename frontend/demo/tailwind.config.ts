import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          deep: '#070810',
          panel: '#0d0f18',
          card: '#151825',
          hover: '#1e2235',
        },
        border: {
          DEFAULT: '#262a3d',
          light: '#333750',
        },
        text: {
          primary: '#e6e2da',
          secondary: '#9b9a94',
          muted: '#5c5b57',
        },
        accent: {
          amber: '#e8a838',
          lime: '#a6e22e',
          cyan: '#42d4c8',
          coral: '#e8573a',
          blue: '#5089e8',
          violet: '#9b6dff',
        },
      },
      fontFamily: {
        display: ['"Bebas Neue"', 'sans-serif'],
        body: ['"DM Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.35s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'bar-fill': 'barFill 0.7s ease-out',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        barFill: {
          from: { width: '0%' },
        },
      },
    },
  },
  plugins: [],
} satisfies Config
