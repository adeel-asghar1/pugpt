import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Forward API calls to the FastAPI backend during development,
    // so the frontend can use relative URLs like fetch('/api/chat').
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
