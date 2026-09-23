import { createRoot } from 'react-dom/client';
import { Gallery } from './gallery';
import { ExportPage } from './export';
import type { Figure } from '../src';

const params = new URLSearchParams(location.search);
const root = createRoot(document.getElementById('root')!);

if (params.has('export')) {
  // scripts/export.mjs drives this: ?export&dark#<figure slug>
  const slug = decodeURIComponent(location.hash.slice(1));
  const figures = import.meta.glob<{ default: Figure }>('../figures/*.ts', { eager: true });
  const entry = Object.entries(figures).find(([path]) => path.endsWith(`/${slug}.ts`));
  if (!entry) throw new Error(`no figure named ${slug}`);
  document.body.style.margin = '0';
  root.render(<ExportPage figure={entry[1].default} dark={params.has('dark')} />);
} else {
  root.render(<Gallery />);
}
