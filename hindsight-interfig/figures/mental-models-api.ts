import type { Figure, FigRow } from '../src';

// Mental Models (API) for hindsight-docs/docs/developer/api/mental-models.mdx: create one, watch it refresh
// itself in delta mode after consolidation, then see reflect use it first. One running example: "Team overview".

const CREATE: FigRow[] = [
  { tag: 'create', tone: 'gray', text: 'Team overview' },
  { tag: 'source query', tone: 'gray', text: '“Who is on the team and what do they work on?”' },
  { tag: 'trigger', tone: 'gray', text: 'refresh after consolidation', meta: 'delta' },
];
const QUESTION: FigRow[] = [{ tag: 'question', tone: 'gray', text: '“Who works on the ML project?”' }];
const ALICE_OBS: FigRow = { text: 'Alice: research team at Google', meta: '3 sources' };
const BOB_OBS: FigRow = { text: 'Bob leads the ML project', meta: '2 sources' };
const HEAD = (mark: string): FigRow => ({
  tag: 'team overview',
  tone: 'orange',
  text: 'refreshed just now',
  mark,
});
const ALICE = '• Alice: research team, Google';
const BOB = '• Bob: leads the ML project';
const CAROL = '• Carol: joined the ML project';
const step = (n: number, name: string, mark = '✓'): FigRow => ({
  text: `${n}. ${name}`,
  mono: true,
  mark,
});

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 40,
    children: [
      { id: 'app', label: 'Your App', lines: 11, width: 200 },
      {
        label: 'Hindsight API',
        direction: 'column',
        gap: 60,
        children: [
          { id: 'mmapi', label: 'Mental models API', sub: 'create · get · refresh', lines: 2 },
          { id: 'reflect', label: 'Reflect', sub: 'agent loop', lines: 3, width: 196 },
        ],
      },
      {
        label: 'Hindsight Worker',
        direction: 'column',
        gap: 40,
        children: [
          { id: 'consolidate', label: 'Consolidation', sub: 'facts → observations', lines: 2 },
          {
            id: 'refresh',
            label: 'Refresh',
            sub: 'reflect on the source query',
            lines: 6,
            width: 196,
          },
        ],
      },
      {
        id: 'bank',
        label: 'Memory Bank',
        direction: 'column',
        gap: 28,
        children: [
          {
            id: 'facts',
            label: 'Facts',
            sub: 'world · experience',
            shape: 'store',
            lines: 6,
            width: 196,
          },
          { id: 'obs', label: 'Observations', shape: 'store', lines: 6, width: 196 },
          { id: 'mm', label: 'Mental Models', shape: 'store', lines: 11, width: 210 },
        ],
      },
    ],
  },
  edges: [
    { id: 'create', from: 'app', to: 'mmapi', label: 'create' },
    { id: 'ask', from: 'app', to: 'reflect', label: 'reflect' },
    { id: 'queue', from: 'mmapi', to: 'refresh', label: 'queue refresh' },
    { id: 'new-facts', from: 'facts', to: 'consolidate', label: 'new facts' },
    { id: 'write-obs', from: 'consolidate', to: 'obs' },
    { id: 'trigger', from: 'consolidate', to: 'refresh', label: 'trigger' },
    { id: 'read-obs', from: 'refresh', to: 'obs' },
    { id: 'read-facts', from: 'refresh', to: 'facts' },
    { id: 'write', from: 'refresh', to: 'mm', label: 'write' },
    { id: 'search-mm', from: 'reflect', to: 'mm', label: 'search_mental_models', quiet: true },
  ],
  steps: [
    {
      label: 'create()',
      flow: [
        {
          edges: { edge: 'create', data: 'POST /mental-models' },
          show: { app: CREATE },
          say: 'You create a mental model: a name, the question it answers, and when it should refresh.',
        },
        {
          show: {
            mmapi: [{ text: 'Team overview saved' }, { text: 'content: not yet', meta: 'async' }],
          },
          say: 'The API saves it right away. Its content does not exist yet.',
        },
        {
          edges: [
            { edge: 'queue', data: 'refresh task' },
            { edge: 'create', back: true, data: 'operation_id' },
          ],
          show: {
            app: [...CREATE, { tag: 'response', tone: 'green', text: 'operation queued', mark: '…' }],
          },
          say: 'The content is written in the background. The response is an operation id you can poll.',
        },
        {
          edges: { edge: 'read-obs', data: 'search_observations' },
          show: {
            refresh: [{ text: 'reflect on the source query' }, step(1, 'search_observations')],
            obs: [
              { ...ALICE_OBS, mark: '✓' },
              { ...BOB_OBS, mark: '✓' },
            ],
          },
          say: 'A refresh is a reflect run on the source query. It reads observations first…',
          ms: 2800,
        },
        {
          edges: { edge: 'read-facts', data: 'recall' },
          show: {
            refresh: [{ text: 'reflect on the source query' }, step(1, 'search_observations'), step(2, 'recall')],
            facts: [
              {
                tag: 'world',
                tone: 'blue',
                text: 'Alice joined Google',
                meta: 'Mar 2026',
                mark: '✓',
              },
              {
                tag: 'world',
                tone: 'blue',
                text: 'Bob leads the ML project',
                meta: 'Feb 2026',
                mark: '✓',
              },
            ],
          },
          say: '…then raw facts, to check the details.',
        },
        {
          edges: { edge: 'write', data: 'content' },
          show: {
            refresh: [
              { text: 'reflect on the source query' },
              step(1, 'search_observations'),
              step(2, 'recall'),
              step(3, 'write document'),
            ],
            mm: [HEAD('new'), { text: ALICE }, { text: BOB }],
          },
          say: 'It writes the document and stores it with the memories it is based on.',
          ms: 3000,
        },
      ],
    },
    {
      label: 'auto refresh',
      flow: [
        {
          edges: { edge: 'new-facts', data: 'new fact' },
          show: {
            app: [{ tag: 'retain', tone: 'gray', text: '“Carol joined the ML project this week.”' }],
            facts: [
              {
                tag: 'world',
                tone: 'blue',
                text: 'Carol joined the ML project',
                meta: 'Sep 2026',
                mark: 'new',
              },
            ],
            consolidate: [{ text: '1 new fact' }, { text: 'merge into beliefs' }],
          },
          say: 'Later, your agent retains a new fact. Consolidation picks it up.',
          ms: 2800,
        },
        {
          edges: 'write-obs',
          show: {
            obs: [ALICE_OBS, BOB_OBS, { text: 'Carol works on the ML project', meta: '1 source', mark: 'new' }],
          },
          say: 'It writes an observation about Carol.',
        },
        {
          edges: { edge: 'trigger', data: 'refresh_after_consolidation' },
          show: {
            refresh: [{ text: 'scope changed since last refresh', mark: 'stale' }, { text: 'delta: read only what is new' }],
          },
          say: 'Before queueing a refresh, consolidation checks that the model’s scope holds a memory newer than the last one it read. If not, no refresh runs and no LLM is spent.',
          ms: 3200,
        },
        {
          edges: { edge: 'read-obs', data: 'since last refresh' },
          show: {
            refresh: [
              { text: 'scope changed since last refresh', mark: 'stale' },
              { text: 'delta: read only what is new' },
              { text: 'append_block', mono: true, meta: 'Carol', mark: '✓' },
            ],
          },
          say: 'In delta mode it reads only the new memories, and answers with small edits instead of a rewrite.',
          ms: 2800,
        },
        {
          edges: { edge: 'write', data: 'append_block' },
          show: {
            mm: [HEAD('↻'), { text: ALICE, mark: '=' }, { text: BOB, mark: '=' }, { text: CAROL, mark: 'new' }],
          },
          say: 'Lines no edit touches are copied byte for byte. Only Carol’s line is added.',
          ms: 3000,
        },
      ],
    },
    {
      label: 'reflect()',
      flow: [
        {
          edges: { edge: 'ask', data: '“Who works on the ML project?”' },
          show: { app: QUESTION },
          say: 'Now your agent asks a question.',
        },
        {
          edges: { edge: 'search-mm', data: 'search_mental_models' },
          show: {
            reflect: [step(1, 'search_mental_models'), { text: 'Team overview', meta: 'fresh' }],
            mm: [
              { tag: 'team overview', tone: 'orange', text: 'matched by meaning', mark: 'fresh' },
              { text: ALICE },
              { text: BOB },
              { text: CAROL },
            ],
          },
          say: 'Reflect searches mental models first, by meaning. Team overview matches, and it is up to date.',
          ms: 3000,
        },
        {
          show: {
            reflect: [step(1, 'search_mental_models'), { text: 'Team overview', meta: 'fresh' }, step(2, 'done')],
          },
          say: 'It covers the question, so the agent can answer from it. If it were stale or off topic, the agent would go on to observations and raw facts.',
          ms: 3200,
        },
        {
          edges: { edge: 'ask', back: true, data: 'answer' },
          show: {
            app: [
              ...QUESTION,
              {
                tag: 'answer',
                tone: 'green',
                text: '“Bob leads it, and Carol just joined.”',
                meta: 'from Team overview',
              },
            ],
          },
          say: 'The answer comes back, citing the mental model it used.',
          ms: 3000,
        },
      ],
    },
  ],
};

export default {
  title: 'Mental Models (API)',
  source: 'hindsight-docs/docs/developer/api/mental-models.mdx',
  props,
} satisfies Figure;
