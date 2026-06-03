import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/metadata': 'http://127.0.0.1:8000',
      '/players': 'http://127.0.0.1:8000',
      '/style': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
  },
})
