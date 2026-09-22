import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `npm run dev` serves the gallery. There is no library build: the docs compile src/ themselves.
export default defineConfig({ plugins: [react()] });
