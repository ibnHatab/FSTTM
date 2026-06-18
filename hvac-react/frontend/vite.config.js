import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    // proxy API + websocket to the FastAPI backend during dev
    proxy: {
      '/state': 'http://127.0.0.1:8000',
      '/config': 'http://127.0.0.1:8000',
      '/command': 'http://127.0.0.1:8000',
      '/set': 'http://127.0.0.1:8000',
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
