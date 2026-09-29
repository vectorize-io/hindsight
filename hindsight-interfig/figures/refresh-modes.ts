import type { Figure, FigRow } from '../src';

// Refresh Mode for hindsight-docs/docs/developer/api/mental-models.mdx: the same mental model and the same
// new memory, refreshed in `full` mode and in `delta` mode, plus the case where delta falls back to full.

const config = (mode: string, query = '“Who is on the team and what do they work on?”'): FigRow[] => [
  { tag: 'mental model', tone: 'gray', text: 'Team overview' },
  { tag: 'source query', tone: 'gray', text: query },
  { tag: 'trigger.mode', tone: 'gray', text: mode, mono: true },
];

// The bank: two memories the last refresh already saw, one that arrived after it.
const ALICE: FigRow = { tag: 'jan', tone: 'gray', text: 'Alice: research team at Google' };
const BOB: FigRow = { tag: 'feb', tone: 'gray', text: 'Bob leads the ML project' };
const CAROL: FigRow = { tag: 'sep', tone: 'blue', text: 'Carol joined the ML project' };
const WATERMARK: FigRow = { text: 'last_memory_seen_at = Feb', mono: true };

// The document before the refresh: three sections.
const BEFORE: FigRow[] = [
  { tag: 'people', tone: 'orange', text: 'Alice: research team, Google' },
  { tag: 'research', tone: 'orange', text: 'Retrieval quality study' },
  { tag: 'ml project', tone: 'orange', text: 'Bob leads it' },
];

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 44,
    children: [
      { id: 'cfg', label: 'Mental model', sub: 'its config', width: 200 },
      {
        label: 'Hindsight Worker · Refresh',
        direction: 'column',
        gap: 48,
        children: [
          { id: 'reflect', label: 'Reflect', sub: 'on the source query', width: 196 },
          { id: 'ops', label: 'Delta edit', sub: 'LLM emits operations', width: 196 },
        ],
      },
      {
        label: 'Memory Bank',
        direction: 'column',
        gap: 36,
        children: [
          { id: 'mem', label: 'Memories', sub: 'in the model’s scope', shape: 'store', width: 220 },
          { id: 'doc', label: 'Team overview', sub: 'the stored document', shape: 'store', width: 220 },
        ],
      },
    ],
  },
  edges: [
    { id: 'go', from: 'cfg', to: 'reflect', label: 'refresh' },
    { id: 'read', from: 'reflect', to: 'mem', label: 'read' },
    { id: 'draft', from: 'reflect', to: 'ops', label: 'new findings' },
    { id: 'baseline', from: 'doc', to: 'ops', label: 'current sections', quiet: true },
    { id: 'apply', from: 'ops', to: 'doc', label: 'apply' },
    { id: 'rewrite', from: 'reflect', to: 'doc', label: 'rewrite', quiet: true },
  ],
  steps: [
    {
      label: 'full',
      flow: [
        {
          edges: { edge: 'go', data: 'mode: full' },
          show: { cfg: config('"full"  (default)'), doc: BEFORE },
          say: 'Full is the default. Every refresh writes the whole document again.',
        },
        {
          edges: { edge: 'read', data: 'whole scope' },
          show: {
            mem: [
              { ...ALICE, mark: 'read' },
              { ...BOB, mark: 'read' },
              { ...CAROL, mark: 'read' },
            ],
            reflect: [{ text: 'window: all of its scope' }, { text: '3 memories read' }],
          },
          say: 'It reads everything in the model’s scope, old and new.',
        },
        {
          edges: { edge: 'rewrite', data: 'new document' },
          show: {
            doc: [
              { tag: 'people', tone: 'orange', text: 'Alice works in Google’s research group', mark: 'rewritten' },
              { tag: 'research', tone: 'orange', text: 'A study of retrieval quality', mark: 'rewritten' },
              { tag: 'ml project', tone: 'orange', text: 'Led by Bob, with Carol', mark: 'rewritten' },
            ],
          },
          say: 'The LLM writes a fresh document. Carol is in, but untouched sections come back reworded too, and small drifts add up over many refreshes.',
          ms: 3600,
        },
      ],
    },
    {
      label: 'delta',
      flow: [
        {
          edges: { edge: 'go', data: 'mode: delta' },
          show: { cfg: config('"delta"'), doc: BEFORE },
          say: 'Delta edits the document instead of rewriting it.',
        },
        {
          edges: { edge: 'read', data: 'since Feb' },
          show: {
            mem: [{ ...ALICE, mark: 'seen' }, { ...BOB, mark: 'seen' }, { ...CAROL, mark: 'new' }, WATERMARK],
            reflect: [{ text: 'window: since last_memory_seen_at' }, { text: '1 new memory read' }],
          },
          say: 'It only reads memories newer than the last one the previous refresh saw.',
          ms: 3000,
        },
        {
          edges: [
            { edge: 'draft', data: 'Carol joined ML' },
            { edge: 'baseline', data: '3 sections' },
          ],
          show: {
            ops: [{ text: 'append_block', mono: true, meta: 'ML project' }, { text: '“Carol joined the ML project.”' }],
          },
          say: 'A second LLM call compares the new findings with the current sections and answers with edit operations, not a document.',
          ms: 3200,
        },
        {
          edges: { edge: 'apply', data: '1 operation' },
          show: {
            doc: [
              { ...BEFORE[0], mark: '=' },
              { ...BEFORE[1], mark: '=' },
              { ...BEFORE[2], mark: '=' },
              { text: '+ Carol joined the ML project', mark: 'new' },
            ],
            mem: [ALICE, BOB, CAROL, { text: 'last_memory_seen_at = Sep', mono: true }],
          },
          say: 'The operation is applied. Every section it does not touch is copied through byte for byte, and the watermark moves to Sep.',
          ms: 3600,
        },
      ],
    },
    {
      label: 'delta falls back',
      flow: [
        {
          edges: { edge: 'go', data: 'mode: delta' },
          show: { cfg: config('"delta"', '“Who works on what, and since when?”  ← edited'), doc: BEFORE },
          say: 'Delta needs something stable to edit. Here the source query was changed.',
        },
        {
          show: {
            reflect: [
              { text: 'fallback: full', mark: '!' },
              { text: 'source_query_changed', mono: true },
            ],
          },
          say: 'The topic moved, so the old structure may no longer fit. The refresh falls back to a full rewrite. The same happens when the model has no content yet.',
          ms: 3200,
        },
        {
          edges: [{ edge: 'read', data: 'whole scope' }],
          show: {
            mem: [
              { ...ALICE, mark: 'read' },
              { ...BOB, mark: 'read' },
              { ...CAROL, mark: 'read' },
            ],
          },
          say: 'It reads the whole scope, like full mode…',
        },
        {
          edges: { edge: 'rewrite', data: 'new document' },
          show: {
            doc: [
              { tag: 'people', tone: 'orange', text: 'Alice since Jan, Bob since Feb, Carol since Sep', mark: 'rewritten' },
              { tag: 'work', tone: 'orange', text: 'Research: Alice · ML: Bob, Carol', mark: 'rewritten' },
            ],
          },
          say: '…and writes a new document for the new question. The next refresh can edit this one in delta mode again.',
          ms: 3400,
        },
      ],
    },
  ],
};

export default {
  title: 'Refresh modes: full vs delta',
  source: 'hindsight-docs/docs/developer/api/mental-models.mdx',
  props,
} satisfies Figure;
