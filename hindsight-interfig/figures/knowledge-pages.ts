import { createElement as h, type ReactNode } from 'react';
import type { Figure, FigRow } from '../src';

// "Knowledge Pages" for hindsight-docs/docs/developer/knowledge-pages.mdx, also exported as a square
// social clip (`npm run export -- knowledge-pages --square`). One straight story:
// A team's tools retain what happened → Hindsight learns from it → the "Deploying the API" page
// updates itself → another client asks, and gets the answer from the updated page.
// Two frames make it plain what is yours and what Hindsight runs.

const MUTED = 'var(--fig-muted, #6b7280)';
const GREEN = '#16a34a';
const RED = '#dc2626';
const ORANGE = '#ea8a0c';
const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';

const CSS = `
@keyframes kp-type{from{opacity:0;transform:translateX(-6px);clip-path:inset(0 100% 0 0)}to{opacity:1;transform:none;clip-path:inset(0 0 0 0)}}
@keyframes kp-add{0%{background:color-mix(in srgb,${GREEN} 45%,transparent)}100%{background:color-mix(in srgb,${GREEN} 12%,transparent)}}
@keyframes kp-del{0%{opacity:1}100%{opacity:.5}}
@keyframes kp-pulse{0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,${ORANGE} 60%,transparent)}50%{box-shadow:0 0 0 6px transparent}}
@keyframes kp-pop{0%{transform:scale(.6);opacity:0}70%{transform:scale(1.12)}100%{transform:scale(1);opacity:1}}
`;

// --- The page: a markdown document, one line per row, with a status badge on top --------------------

type Line = { t: string; kind?: 'h1' | 'h2' | 'li'; state?: 'add' | 'del' };
type Status = 'fresh' | 'stale' | 'updated';

const badge = (s: Status) =>
  h(
    'span',
    {
      style: {
        padding: '1px 8px',
        borderRadius: 999,
        fontSize: 10.5,
        fontWeight: 700,
        letterSpacing: '.05em',
        textTransform: 'uppercase',
        color: '#fff',
        background: s === 'stale' ? ORANGE : GREEN,
        animation: s === 'stale' ? 'kp-pop .4s both, kp-pulse 1.2s .4s infinite' : 'kp-pop .4s both',
      },
    },
    s === 'stale' ? 'out of date' : s === 'updated' ? 'just updated' : 'up to date',
  );

const doc = (lines: Line[], status: Status, meta?: ReactNode) =>
  h(
    'div',
    { style: { fontSize: 13, lineHeight: '19px' } },
    h('style', null, CSS),
    h('div', { style: { display: 'flex', justifyContent: 'flex-end', marginBottom: 2 } }, badge(status)),
    ...lines.map((l, i) => {
      const kind = l.kind ?? 'li';
      return h(
        'div',
        {
          key: i,
          style: {
            display: 'flex',
            gap: 6,
            padding: '0 4px',
            marginTop: kind === 'h2' && i > 0 ? 7 : 0,
            borderRadius: 3,
            fontSize: kind === 'h1' ? 16 : kind === 'h2' ? 14 : 13,
            fontWeight: kind === 'li' ? 400 : 700,
            color: l.state === 'del' ? RED : undefined,
            textDecoration: l.state === 'del' ? 'line-through' : undefined,
            animation:
              l.state === 'add'
                ? `kp-type .5s ${i * 0.04}s both, kp-add 2s .4s both`
                : l.state === 'del'
                  ? 'kp-del .8s .2s both'
                  : undefined,
          },
        },
        h(
          'span',
          { style: { width: 18, flex: 'none', fontFamily: MONO, color: l.state === 'add' ? GREEN : l.state === 'del' ? RED : MUTED } },
          l.state === 'add' ? '+' : l.state === 'del' ? '−' : kind === 'h1' ? '#' : kind === 'h2' ? '##' : '•',
        ),
        h('span', null, l.t),
      );
    }),
    meta && h('div', { style: { marginTop: 8, fontSize: 11.5, color: MUTED } }, meta),
  );

const V1: Line[] = [
  { t: 'Deploying the API', kind: 'h1' },
  { t: 'How a release ships', kind: 'h2' },
  { t: 'Merging to main builds and ships the API' },
  { t: 'Production needs a sign-off' },
  { t: 'Database changes', kind: 'h2' },
  { t: 'They run before new traffic arrives' },
  { t: 'Keep them small: a slow one broke July’s release' },
  { t: 'Rolling back', kind: 'h2' },
  { t: 'Redeploy the previous version' },
];

const V2: Line[] = [
  V1[0],
  V1[1],
  V1[2],
  V1[3],
  { t: 'It goes to 10% of users for 15 min first', state: 'add' },
  V1[4],
  V1[5],
  { ...V1[6], state: 'del' },
  { t: 'Releases wait until database changes finish', state: 'add' },
  V1[7],
  V1[8],
  { t: 'Incidents', kind: 'h2', state: 'add' },
  { t: 'Sep 24: a release hung 20 min, went live too early', state: 'add' },
];
const V2_CLEAN: Line[] = V2.filter((l) => l.state !== 'del').map(({ state: _, ...l }) => l);

// --- Cards ---------------------------------------------------------------------------------------

const OLD_MEM: FigRow[] = [
  { tag: 'doc', tone: 'purple', text: 'Deploy runbook, v3' },
  { tag: 'slack', tone: 'purple', text: '“Prod needs a sign-off from on-call”' },
  { tag: 'incident', tone: 'purple', text: 'July: a slow database change broke the release' },
];
const NEW_MEM: FigRow[] = [
  { tag: 'incident', tone: 'purple', text: 'A release hung 20 min: it went live before the database was ready' },
  { tag: 'slack', tone: 'purple', text: '“We ship to 10% of users first now”' },
  { tag: 'code change', tone: 'purple', text: 'Wait for database changes before going live' },
];

const OBS: FigRow[] = [
  { text: 'Merging to main ships the API', meta: '6 sources' },
  { text: 'Production needs a sign-off', meta: '4 sources' },
  { text: 'Keep database changes small', meta: '2 sources' },
];
const OBS_NEW: FigRow[] = [
  OBS[0],
  OBS[1],
  { text: 'Releases wait for database changes', meta: '3 sources', mark: 'corrected' },
  { text: 'Production goes to 10% of users first', meta: '2 sources', mark: 'new' },
];

const QUESTION: FigRow = { tag: 'question', tone: 'gray', text: '“Why did the release hang?”' };

// --- The figure ---------------------------------------------------------------------------------

const props: Figure['props'] = {
  speed: 2200,
  // Two frames: what is in your hands (left) and what Hindsight runs for you (right). Inside Hindsight the
  // story runs down the right column (memories → learn → beliefs → rewrite) and back up the left (page → search).
  layout: {
    gap: 64,
    children: [
      {
        label: 'You',
        direction: 'column',
        gap: 380,
        children: [
          { id: 'tools', label: 'Your Team’s Tools', sub: 'send what happens', lines: 3, width: 250 },
          { id: 'app', label: 'Another Client', sub: 'an agent, a teammate', lines: 3, width: 250 },
        ],
      },
      {
        label: 'Hindsight',
        logo: '/img/logo.png',
        gap: 60,
        children: [
          {
            direction: 'column',
            gap: 56,
            children: [
              { id: 'retain', label: 'Retain', sub: 'takes in what you send', width: 200 },
              { id: 'page', label: 'Deploying the API', sub: 'a knowledge page', shape: 'store', lines: 15, width: 390 },
              { id: 'api', label: 'Search', sub: 'finds the right page', width: 200 },
            ],
          },
          {
            direction: 'column',
            gap: 44,
            children: [
              { id: 'mem', label: 'New memories', sub: 'memory bank', shape: 'store', lines: 3, width: 310 },
              { id: 'consolidate', label: 'Consolidation', sub: 'learns', lines: 1, width: 310 },
              { id: 'obs', label: 'Observations', sub: 'what it believes', shape: 'store', lines: 4, width: 310 },
              { id: 'refresh', label: 'Refresh', sub: 'rewrites pages', lines: 3, width: 310 },
            ],
          },
        ],
      },
    ],
  },
  edges: [
    { id: 'send', from: 'tools', to: 'retain', label: 'retain' },
    { id: 'store', from: 'retain', to: 'mem' },
    { id: 'learn', from: 'mem', to: 'consolidate' },
    { id: 'believe', from: 'consolidate', to: 'obs' },
    { id: 'changed', from: 'obs', to: 'refresh' },
    { id: 'write', from: 'refresh', to: 'page', label: 'update' },
    { id: 'ask', from: 'app', to: 'api', label: 'ask' },
    { id: 'find', from: 'api', to: 'page' },
  ],
  steps: [
    {
      label: 'knowledge pages',
      flow: [
        {
          show: {
            tools: [{ tag: 'every day', tone: 'gray', text: 'chat, incidents, code changes' }],
            mem: OLD_MEM,
            consolidate: [{ tag: 'done', tone: 'green', text: 'everything learned' }],
            obs: OBS,
            refresh: [{ tag: 'idle', tone: 'gray', text: 'every page up to date' }],
            page: doc(V1, 'fresh', 'written by Hindsight from what the bank believes'),
            app: [{ tag: 'waiting', tone: 'gray', text: 'will ask about deploys later' }],
          },
          say: 'Hindsight keeps a wiki for your team. This page on deploys is written from what the bank knows.',
          ms: 3600,
        },
        // 1. Retain.
        {
          edges: { edge: 'send', data: 'what happened' },
          show: { tools: NEW_MEM },
          say: 'Then something happens. Your tools send it to Hindsight: an incident, a Slack thread, a code change.',
          ms: 3800,
        },
        {
          edges: 'store',
          show: { mem: NEW_MEM },
          say: 'Hindsight stores it as new memories.',
          ms: 2600,
        },
        // 2. Learn.
        {
          edges: { edge: 'learn', data: 'what’s new' },
          show: { consolidate: [{ tag: 'learned', tone: 'orange', text: '1 new · 1 corrected' }] },
          say: 'It learns from them…',
          ms: 2400,
        },
        {
          edges: 'believe',
          show: { obs: OBS_NEW },
          say: '…adds what is new, and corrects what is no longer true.',
          ms: 3200,
        },
        // 3. The page updates itself.
        {
          edges: { edge: 'changed', data: 'what changed' },
          show: {
            refresh: [{ tag: 'updating', tone: 'orange', text: 'Deploying the API' }],
            page: doc(V1, 'stale', 'written by Hindsight from what the bank believes'),
          },
          say: 'The page on deploys is now out of date, so Hindsight updates it.',
          ms: 3000,
        },
        {
          edges: { edge: 'write', data: '3 changes' },
          show: {
            refresh: [
              { tag: 'added', tone: 'green', text: 'the 10% rollout' },
              { tag: 'fixed', tone: 'green', text: 'the database advice' },
              { tag: 'added', tone: 'green', text: 'an incidents section' },
            ],
            page: doc(V2, 'updated', 'only what changed; the rest is untouched'),
          },
          say: 'Only what changed: the old advice is crossed out, the new facts land where they belong.',
          ms: 5600,
        },
        // 4. Someone else asks.
        {
          edges: { edge: 'ask', data: '“Why did the release hang?”' },
          show: { app: [QUESTION], page: doc(V2_CLEAN, 'fresh', 'up to date') },
          say: 'Later, another client asks: an AI agent, or a teammate.',
          ms: 3000,
        },
        {
          edges: { edge: 'find', data: 'search' },
          say: 'Hindsight finds the page, already written and up to date.',
          ms: 2600,
        },
        {
          edges: { edge: 'find', back: true, data: 'the page' },
          ms: 2000,
        },
        {
          edges: { edge: 'ask', back: true, data: 'the answer' },
          show: {
            app: [
              QUESTION,
              { tag: 'answer', tone: 'green', text: 'It went live before the database was ready. Releases now wait for it.', mark: '✓' },
            ],
          },
          say: 'The answer already includes what just happened. Nobody had to update the wiki.',
          ms: 4200,
        },
      ],
    },
  ],
};

// Plays at 1.5x: the player only has 1x and 2x, so every beat is shortened here instead.
const PACE = 1.5;
props.speed = props.speed! / PACE;
for (const step of props.steps ?? [])
  step.flow = step.flow.map((b) => (typeof b === 'object' && !Array.isArray(b) && 'ms' in b && b.ms ? { ...b, ms: b.ms / PACE } : b));

export default {
  title: 'Knowledge Pages',
  source: 'hindsight-docs/docs/developer/knowledge-pages.mdx',
  props,
} satisfies Figure;
