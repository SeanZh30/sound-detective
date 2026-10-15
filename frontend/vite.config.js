import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

// Sample WAVs live once in data/samples; FastAPI serves them in dev and preview.
const proxy = Object.fromEntries(['/api', '/audio', '/ESC50-LICENSE.txt'].map(path => [path, 'http://127.0.0.1:8000']));
export default defineConfig({
  plugins: [react()],
  server: { proxy, fs: { allow: [fileURLToPath(new URL('..', import.meta.url))] } },
  preview: { proxy },
  test: { environment: 'jsdom', setupFiles: './src/test-setup.js' },
});
