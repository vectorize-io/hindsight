import type { Figure, FigRow } from '../src';

// "Mental Models" for hindsight-docs/docs/developer/mental-models.mdx: a standing answer to one
// question, written ahead of time by reflect, kept current in the background, read like a row.
// Running example: a "Team overview" model on a team's bank. Boxes process, cylinders store.

const QUESTION: FigRow = {
  tag: 'question',
  tone: 'gray',
  text: '“Who is on the team and what does each person work on?”',
};
const BOB: FigRow = { tag: 'research', tone: 'blue', text: 'Bob: retrieval evals' };
const CAROL: FigRow = { tag: 'ml team', tone: 'purple', text: 'Carol: ranker fine-tuning' };
const ALICE: FigRow = { tag: 'research', tone: 'blue', text: 'Alice: joined in March' };
const OBS: FigRow[] = [
  { text: 'Bob runs the retrieval evals', meta: '4 sources' },
  { text: 'Carol moved to ML to fine-tune the ranker', meta: '3 sources' },
];

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 40,
    children: [
      { id: 'app', label: 'Your App', lines: 6, width: 200 },
      {
        label: 'Hindsight',
        direction: 'column',
        gap: 32,
        children: [
          { id: 'api', label: 'Mental Models API', sub: 'create · read' },
          {
            label: 'Hindsight Worker',
            direction: 'column',
            gap: 32,
            children: [
              {
                id: 'trigger',
                label: 'Trigger',
                sub: 'after consolidation, or cron',
                lines: 2,
                width: 200,
              },
              {
                id: 'reflect',
                label: 'Reflect',
                sub: 'answers the question',
                lines: 3,
                width: 200,
              },
            ],
          },
        ],
      },
      {
        label: 'Memory Bank',
        direction: 'column',
        gap: 32,
        children: [
          {
            id: 'sources',
            label: 'Reads',
            gap: 24,
            children: [
              { id: 'obs', label: 'Observations', shape: 'store', lines: 6, width: 200 },
              { id: 'facts', label: 'Facts', shape: 'store', lines: 4, width: 190 },
            ],
          },
          {
            label: 'Writes',
            gap: 24,
            children: [
              {
                id: 'mm',
                label: 'Mental model',
                sub: 'Team overview',
                shape: 'store',
                lines: 8,
                width: 200,
              },
              {
                id: 'hist',
                label: 'History',
                sub: 'previous versions',
                shape: 'store',
                lines: 2,
                width: 190,
              },
            ],
          },
        ],
      },
    ],
  },
  edges: [
    { id: 'call', from: 'app', to: 'api' },
    { id: 'build', from: 'api', to: 'reflect', label: 'first build', quiet: true },
    { id: 'read-src', from: 'reflect', to: 'sources' },
    { id: 'write', from: 'reflect', to: 'mm', label: 'write' },
    { id: 'version', from: 'mm', to: 'hist' },
    { id: 'changed', from: 'sources', to: 'trigger', label: 'changed in scope' },
    { id: 'refresh', from: 'trigger', to: 'reflect', label: 'refresh' },
    { id: 'read', from: 'api', to: 'mm', label: 'read', quiet: true },
  ],
  steps: [
    {
      label: 'create',
      flow: [
        {
          edges: { edge: 'call', data: 'source question' },
          show: {
            app: [
              QUESTION,
              {
                tag: 'trigger',
                tone: 'gray',
                text: 'refresh after consolidation',
                meta: 'delta mode',
              },
            ],
          },
          say: 'You define the question once, plus how it should stay current.',
        },
        {
          edges: { edge: 'build', data: 'build it' },
          show: { reflect: [{ text: 'reflect over this bank', meta: 'in the model’s tag scope' }] },
          say: 'Hindsight writes the answer with reflect, in the background. Tags limit which memories it may read.',
        },
        {
          edges: { edge: 'read-src', data: 'observations + facts' },
          show: {
            obs: OBS.map((o) => ({ ...o, mark: '✓' })),
            facts: [
              {
                tag: 'world',
                tone: 'blue',
                text: 'Carol moved to the ML team',
                meta: 'Feb 2026',
                mark: '✓',
              },
            ],
          },
          say: 'It reads consolidated observations first, and raw facts to check details.',
          ms: 2800,
        },
        {
          edges: { edge: 'write', data: 'document v1' },
          show: { mm: [BOB, CAROL, { text: 'based on 2 observations, 1 fact', meta: 'v1' }] },
          say: 'The answer is stored as a document, together with the evidence it was built from.',
          ms: 3000,
        },
      ],
    },
    {
      label: 'stay current',
      flow: [
        {
          show: {
            obs: [...OBS, { text: 'Alice joined the research team', meta: '2 sources', mark: 'new' }],
            mm: [BOB, CAROL, { text: 'based on 2 observations, 1 fact', meta: 'v1' }],
          },
          say: 'The bank keeps learning. Consolidation writes a new observation inside this model’s scope.',
          ms: 2800,
        },
        {
          edges: { edge: 'changed', data: 'newer than v1' },
          show: {
            trigger: [{ tag: 'stale', tone: 'orange', text: 'in-scope memory newer than the last one v1 read' }],
          },
          say: 'The trigger checks first: is there a memory in this model’s scope newer than the newest one its last refresh read? Activity elsewhere in the bank does not count.',
          ms: 3200,
        },
        {
          edges: { edge: 'refresh', data: 'refresh' },
          show: { reflect: [{ text: 'read only what is new since v1' }] },
          say: 'In delta mode the refresh reads only the memories that arrived since the last one.',
        },
        {
          edges: { edge: 'read-src', data: 'what’s new' },
          show: {
            obs: [...OBS, { text: 'Alice joined the research team', meta: '2 sources', mark: '✓' }],
          },
        },
        {
          edges: [
            { edge: 'write', data: 'append_block → Research' },
            { edge: 'version', data: 'v1' },
          ],
          show: {
            reflect: [{ tag: 'edit', tone: 'green', text: 'append_block', meta: 'Research', mono: true }],
            mm: [
              { ...BOB, mark: 'same' },
              { ...ALICE, mark: 'added' },
              { ...CAROL, mark: 'same' },
              { text: 'evidence: 3 observations, 1 fact', meta: 'v2' },
            ],
            hist: [{ tag: 'v1', tone: 'gray', text: 'Bob, Carol' }],
          },
          say: 'Instead of rewriting, it applies edits: one bullet added to Research. Everything else is copied through untouched, and the old version goes to history.',
          ms: 3800,
        },
      ],
    },
    {
      label: 'read',
      flow: [
        {
          edges: { edge: 'call', data: 'get “Team overview”' },
          show: { app: [{ tag: 'read', tone: 'gray', text: 'Team overview' }] },
          say: 'When your app needs the answer, it asks for the model by id.',
        },
        {
          edges: { edge: 'read', data: 'database read' },
          say: 'That is a database read: no retrieval, no LLM call, no waiting.',
        },
        {
          edges: [{ edge: 'read', back: true, data: 'v2 · fresh' }],
          ms: 1600,
        },
        {
          edges: { edge: 'call', back: true, data: 'the document' },
          show: {
            app: [
              { tag: 'read', tone: 'gray', text: 'Team overview' },
              {
                tag: 'answer',
                tone: 'green',
                text: 'Research: Bob, Alice · ML: Carol',
                meta: 'fresh',
                mark: '✓',
              },
            ],
          },
          say: 'Everyone asking gets the same document. Reflect reads mental models first too, and trusts one only while it is fresh.',
          ms: 3400,
        },
      ],
    },
  ],
};

export default {
  title: 'Mental Models',
  source: 'hindsight-docs/docs/developer/mental-models.mdx',
  props,
} satisfies Figure;
