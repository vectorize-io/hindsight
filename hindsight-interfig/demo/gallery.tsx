import { useEffect, useState } from 'react';
import { Flow, type Figure } from '../src';

// Every file in figures shows up here. Add a file to add a figure.
const figures = Object.entries(import.meta.glob<{ default: Figure }>('../figures/*.ts', { eager: true }))
  .map(([path, m]) => ({ slug: path.replace(/^.*\/|\.ts$/g, ''), path, ...m.default }))
  .sort((a, b) => a.title.localeCompare(b.title));
const sources = import.meta.glob<string>('../figures/*.ts', { eager: true, query: '?raw', import: 'default' });

const DARK = { accent: '#3396e8', fg: '#e3e3e3', muted: '#9aa0a6', bg: '#1b1b1d', surface: '#242526', border: '#3a3b3c' };
const slugFromHash = () => decodeURIComponent(location.hash.slice(1)) || figures[0]?.slug;

export function Gallery() {
  const [slug, setSlug] = useState(slugFromHash);
  const [dark, setDark] = useState(() => localStorage.getItem('fig-dark') === '1');
  const [code, setCode] = useState(() => localStorage.getItem('fig-code') === '1');
  useEffect(() => {
    const on = () => setSlug(slugFromHash());
    addEventListener('hashchange', on);
    return () => removeEventListener('hashchange', on);
  }, []);
  useEffect(() => localStorage.setItem('fig-dark', dark ? '1' : '0'), [dark]);
  useEffect(() => localStorage.setItem('fig-code', code ? '1' : '0'), [code]);

  const fig = figures.find((f) => f.slug === slug) ?? figures[0];
  const c = dark
    ? { bg: '#1b1b1d', side: '#242526', fg: '#e3e3e3', muted: '#9aa0a6', line: '#3a3b3c' }
    : { bg: '#fff', side: '#f5f7fa', fg: '#1c1e21', muted: '#606770', line: '#e3e6ea' };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: c.bg, color: c.fg }}>
      <nav
        style={{ width: 240, flexShrink: 0, background: c.side, borderRight: `1px solid ${c.line}`, padding: 16, boxSizing: 'border-box' }}
      >
        <div style={{ fontWeight: 700, marginBottom: 16 }}>interfig</div>
        {figures.map((f) => (
          <a
            key={f.slug}
            href={'#' + f.slug}
            style={{
              display: 'block',
              padding: '6px 10px',
              borderRadius: 6,
              fontSize: 14,
              textDecoration: 'none',
              color: c.fg,
              background: f.slug === fig?.slug ? c.line : 'transparent',
            }}
          >
            {f.title}
          </a>
        ))}
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 24, fontSize: 13, color: c.muted }}>
          <input type="checkbox" checked={dark} onChange={(e) => setDark(e.target.checked)} /> Dark mode
        </label>
        <label style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8, fontSize: 13, color: c.muted }}>
          <input type="checkbox" checked={code} onChange={(e) => setCode(e.target.checked)} /> Show code
        </label>
      </nav>
      {/* same width as the Hindsight docs content column, so what fits here fits there */}
      <main style={{ flex: 1, minWidth: 0, maxWidth: 760, padding: '32px 40px' }}>
        {fig ? (
          <>
            <h1 style={{ margin: 0, fontSize: 24 }}>{fig.title}</h1>
            {fig.source && (
              <div style={{ fontSize: 13, color: c.muted, marginTop: 4 }}>
                from {fig.source} · figures/{fig.slug}.ts
              </div>
            )}
            {/* key: start fresh when switching figures */}
            <Flow key={fig.slug} {...fig.props} theme={dark ? DARK : undefined} />
            {code && (
              <pre
                style={{
                  margin: 0,
                  padding: 16,
                  borderRadius: 8,
                  background: c.side,
                  border: `1px solid ${c.line}`,
                  fontSize: 12.5,
                  lineHeight: 1.5,
                  overflow: 'auto',
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                }}
              >
                {`// how it renders:\n<Flow {...figure.props} />\n\n// figures/${fig.slug}.ts\n` + sources[fig.path]}
              </pre>
            )}
          </>
        ) : (
          <p>No figures yet. Add one to figures/.</p>
        )}
      </main>
    </div>
  );
}
