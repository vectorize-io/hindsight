import type { Figure, FigRow } from '../src';

// "Reflect Agent Loop" for hindsight-docs/docs/developer/reflect.mdx. One question through the agentic loop:
// bank config shapes the instructions, the agent calls tools from the most refined knowledge down to the raw
// source (mental models → observations → recall → expand), verifies a stale observation, then answers with
// citations that are checked against what it actually retrieved. Boxes process, cylinders store.

const QUESTION: FigRow[] = [{ tag: 'question', tone: 'gray', text: '“Should Alice join the ML project?”' }];

// The agent's running log, one row per tool call.
const LOG: FigRow[] = [
  { text: '1. search_mental_models', mono: true, mark: '✓' },
  { text: '2. search_observations', mono: true, mark: 'stale' },
  { text: '3. recall', mono: true, meta: 'verify', mark: '✓' },
  { text: '4. expand', mono: true, mark: '✓' },
  { text: '5. done', mono: true, mark: '✓' },
];
const log = (n: number): FigRow[] => [{ tag: 'iteration', tone: 'gray', text: `${n} of 10 max` }, ...LOG.slice(0, n)];

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 40,
    children: [
      {
        direction: 'column',
        gap: 28,
        children: [
          { id: 'agent', label: 'Your AI Agent', lines: 8, width: 200 },
          {
            id: 'config',
            label: 'Bank config',
            direction: 'column',
            gap: 8,
            children: [
              { id: 'mission', label: 'Mission', shape: 'store', lines: 2 },
              { id: 'disposition', label: 'Disposition', shape: 'store', lines: 1 },
              { id: 'directives', label: 'Directives', shape: 'store', lines: 1 },
            ],
          },
        ],
      },
      {
        id: 'reflect',
        label: 'Reflect',
        direction: 'column',
        gap: 40,
        children: [
          { id: 'loop', label: 'Agent loop', sub: 'LLM picks the next tool', lines: 8, width: 210 },
          { id: 'answer', label: 'Answer', sub: 'final synthesis', lines: 6, width: 210 },
        ],
      },
      {
        id: 'tools',
        label: 'Tools',
        direction: 'column',
        gap: 72,
        children: [
          { id: 't-mm', label: 'search_mental_models' },
          { id: 't-obs', label: 'search_observations' },
          { id: 't-recall', label: 'recall', sub: '4-way search' },
          { id: 't-expand', label: 'expand', sub: 'memory → chunk → document' },
        ],
      },
      {
        id: 'bank',
        label: 'Memory Bank',
        direction: 'column',
        gap: 12,
        children: [
          { id: 'mm', label: 'Mental Models', shape: 'store', lines: 3, width: 196 },
          { id: 'obs', label: 'Observations', shape: 'store', lines: 5, width: 196 },
          {
            id: 'facts',
            label: 'Facts',
            sub: 'world · experience',
            shape: 'store',
            lines: 7,
            width: 196,
          },
          { id: 'chunks', label: 'Chunks & documents', shape: 'store', lines: 2, width: 196 },
        ],
      },
    ],
  },
  edges: [
    { id: 'ask', from: 'agent', to: 'loop', label: 'reflect()' },
    { id: 'shape', from: 'config', to: 'loop', label: 'instructions' },
    { id: 'c-mm', from: 'loop', to: 't-mm' },
    { id: 'c-obs', from: 'loop', to: 't-obs' },
    { id: 'c-recall', from: 'loop', to: 't-recall' },
    { id: 'c-expand', from: 'loop', to: 't-expand' },
    { id: 'r-mm', from: 't-mm', to: 'mm' },
    { id: 'r-obs', from: 't-obs', to: 'obs' },
    { id: 'r-recall', from: 't-recall', to: 'facts' },
    { id: 'r-expand', from: 't-expand', to: 'chunks' },
    { id: 'done', from: 'loop', to: 'answer', label: 'done' },
    { id: 'reply', from: 'answer', to: 'agent' },
  ],
  steps: [
    {
      label: 'reflect()',
      flow: [
        {
          edges: { edge: 'ask', data: '“Should Alice join the ML project?”' },
          show: { agent: QUESTION },
          say: 'reflect() gets a question that needs reasoning, not just lookup.',
        },
        {
          edges: { edge: 'shape', data: 'mission · disposition · directives' },
          show: {
            mission: [{ text: 'Help the team staff projects well' }],
            disposition: [{ text: 'skepticism 4 · empathy 3' }],
            directives: [{ text: 'Never share salaries' }],
            loop: [{ tag: 'plan', tone: 'gray', text: 'start from the most refined knowledge' }],
          },
          say: 'The bank’s mission, disposition and directives go into the agent’s instructions before it looks anything up.',
          ms: 3000,
        },
        {
          edges: { edge: 'c-mm', data: '“Alice ML project”' },
          show: { loop: log(1) },
          say: 'Each turn is one tool call. Retrieval starts at the top, with mental models: curated summaries kept up to date.',
        },
        {
          edges: 'r-mm',
          show: {
            mm: [
              {
                tag: 'model',
                tone: 'orange',
                text: 'Team overview: Alice, research',
                mark: 'fresh',
              },
            ],
          },
          say: 'A fresh mental model can be enough to answer. This one covers the team, not the ML project, so the agent keeps looking.',
          ms: 2400,
        },
        {
          edges: { edge: 'c-obs', data: '“Alice”' },
          show: { loop: log(2) },
          say: 'Then observations: beliefs consolidated from many facts.',
          ms: 1800,
        },
        {
          edges: 'r-obs',
          show: {
            obs: [
              { text: 'Alice works at Google, research team', meta: '3 sources' },
              { tag: 'stale', tone: 'orange', text: 'newer facts not consolidated' },
            ],
          },
          say: 'The result is flagged stale: the bank has facts not yet consolidated into observations. So the agent must check the raw facts.',
          ms: 3000,
        },
        {
          edges: { edge: 'c-recall', data: 'recall("Alice ML")' },
          show: { loop: log(3) },
          say: 'recall() runs the full four-way search over raw facts: the ground truth.',
          ms: 1800,
        },
        {
          edges: 'r-recall',
          show: {
            facts: [
              {
                tag: 'world',
                tone: 'blue',
                text: 'Alice asked to join the ML project',
                meta: 'Sep 2026',
                mark: 'new',
              },
              {
                tag: 'experience',
                tone: 'purple',
                text: 'I suggested Alice for ML',
                meta: 'Apr 2026',
              },
            ],
          },
          say: 'The newer fact is what the stale observation was missing: Alice asked to move to ML herself.',
          ms: 3000,
        },
        {
          edges: { edge: 'c-expand', data: 'expand → document' },
          show: { loop: log(4) },
          ms: 1800,
        },
        {
          edges: 'r-expand',
          show: {
            chunks: [
              {
                tag: 'doc',
                tone: 'gray',
                text: 'Sep 1:1 notes: “I’d love to work on the ML side.”',
                mark: 'read',
              },
            ],
          },
          say: 'recall already returns the chunk each fact came from. expand goes further: the whole source document, for the full context.',
          ms: 2600,
        },
        {
          edges: { edge: 'done', data: 'done' },
          show: {
            loop: log(5),
            answer: [
              { text: 'model · team overview', mark: '✓' },
              { text: 'observation · alice-google', mark: '✓' },
              { text: 'fact · ml-request', mark: '✓' },
              { text: 'fact · 7f3a', meta: 'never retrieved', mark: '✗' },
            ],
          },
          say: 'The agent calls done. Every cited ID is checked against what it actually retrieved; anything else is dropped. It must gather evidence first, and stops after 10 iterations at most.',
          ms: 3800,
        },
        {
          edges: { edge: 'reply', data: 'answer + 3 sources' },
          show: {
            agent: [
              ...QUESTION,
              {
                tag: 'answer',
                tone: 'green',
                text: '“Yes. She is on Google’s research team and asked to join ML in September.”',
                meta: '3 sources',
              },
            ],
          },
          say: 'The answer follows the directives and the bank’s disposition, and comes back with the memories it is based on.',
          ms: 3200,
        },
      ],
    },
  ],
};

export default {
  title: 'Reflect Agent Loop',
  source: 'hindsight-docs/docs/developer/reflect.mdx',
  props,
} satisfies Figure;
