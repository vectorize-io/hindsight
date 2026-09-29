# Control center UI (Preact + Tailwind)

Source for the control-center SPA. Built with Vite to `../static/`, which is
**committed** and served at runtime by the Python `http.server` (no Node at
install/runtime — it works offline).

```bash
npm install      # standalone; not part of the root npm workspace
npm run build    # → ../static/ (index.html + assets/* + public assets)
npm run dev      # local dev server (proxy /api to a running control center)
```

After changing anything under `src/` or `public/`, **rebuild and commit
`../static/`** — the Python package ships that built output, not this source.
Brand assets (logo, favicon, fonts) live in `public/` and are copied verbatim.
