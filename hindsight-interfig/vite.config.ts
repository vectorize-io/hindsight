import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `npm run dev` serves the gallery. There is no library build: the docs compile src/ themselves.
// The docs site's static folder is served too, so figures can use the same asset paths the docs do (/img/logo.png).
export default defineConfig({ plugins: [react()], publicDir: '../hindsight-docs/static' });
