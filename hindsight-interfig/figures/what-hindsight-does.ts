import { createElement as h } from 'react';
import { MiniGraph, type Figure, type FigRow } from '../src';

// "What Hindsight Does" for hindsight-docs/docs/developer/index.mdx: the whole memory pipeline,
// told with one running example (Alice joins Google) so every store shows real data as it fills.
// Boxes process, cylinders store. Recall reaches facts and observations through the same four indexes.

const graph = (lit: string[] = []) =>
  h(MiniGraph, {
    nodes: ['Alice', 'Google', 'ML project'],
    links: [
      ['Alice', 'Google'],
      ['Alice', 'ML project'],
    ],
    lit,
  });

// Rows reused across steps.
const WORLD: FigRow = { tag: 'world', tone: 'blue', text: 'Alice joined Google', meta: 'Mar 2026' };
const EXPERIENCE: FigRow = { tag: 'experience', tone: 'purple', text: 'I suggested Alice for the ML project' };
// Already in the bank before this conversation: it contradicts the new fact, and consolidation settles it.
const OLD_WORLD: FigRow = { tag: 'world', tone: 'blue', text: 'Alice works at Microsoft', meta: 'Jan 2025' };
const OBSERVATION: FigRow = { text: 'Alice works at Google, on the research team', meta: '2 sources' };
// The observation the bank held before this conversation, built from OLD_WORLD.
const OLD_OBSERVATION: FigRow = { text: 'Alice works at Microsoft', meta: '1 source' };
const RESOLVED: FigRow = { tag: 'resolved', tone: 'orange', text: 'Microsoft (Jan 2025) → Google (Mar 2026)' };
const RETAIN_INPUT: FigRow[] = [
  { tag: 'user', tone: 'gray', text: '“Alice joined Google in March, she loves the research team.”' },
  { tag: 'agent', tone: 'gray', text: '“Noted, I’ll suggest her for the ML project.”' },
];
const RECALL_INPUT: FigRow[] = [{ tag: 'query', tone: 'gray', text: '“Where does Alice work?”' }];
const REFLECT_INPUT: FigRow[] = [{ tag: 'question', tone: 'gray', text: '“Is Alice a good fit for the ML project?”' }];
const tool = (n: number, name: string, done = true): FigRow => ({ text: `${n}. ${name}`, mono: true, mark: done ? '✓' : '…' });
const TOOLS = ['search_mental_models', 'search_observations', 'recall', 'expand'];
const tools = (upTo: number, last = true) => TOOLS.slice(0, upTo).map((t, i) => tool(i + 1, t, last || i < upTo - 1));

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 40,
    children: [
      { id: 'agent', label: 'Your AI Agent', lines: 9, width: 196 },
      {
        id: 'api',
        label: 'Hindsight API',
        direction: 'column',
        gap: 24,
        children: [
          { id: 'retain', label: 'Retain', sub: 'LLM extraction' },
          {
            id: 'recall',
            label: 'Recall',
            direction: 'column',
            gap: 10,
            children: [
              { id: 'semantic', label: 'Semantic', sub: 'by meaning' },
              { id: 'keyword', label: 'Keyword', sub: 'exact words' },
              { id: 'graph', label: 'Graph', sub: 'via entities' },
              { id: 'temporal', label: 'Temporal', sub: 'by time' },
            ],
          },
          { id: 'reflect', label: 'Reflect', sub: 'agent loop', lines: 6, width: 200 },
        ],
      },
      {
        id: 'bank',
        label: 'Memory Bank',
        direction: 'column',
        gap: 36,
        children: [
          {
            id: 'sources',
            label: 'Sources',
            gap: 40,
            children: [
              { id: 'docs', label: 'Documents', shape: 'store' },
              { id: 'chunks', label: 'Chunks', shape: 'store' },
            ],
          },
          {
            id: 'memories',
            label: 'Memories',
            gap: 48,
            children: [
              {
                id: 'indexes',
                label: 'Indexes',
                direction: 'column',
                gap: 10,
                children: [
                  { id: 'vectors', label: 'Vectors', shape: 'store', lines: 1 },
                  { id: 'fulltext', label: 'Full text', shape: 'store', lines: 1 },
                  { id: 'egraph', label: 'Entity graph', shape: 'store', lines: 4 },
                  { id: 'time', label: 'Dates', shape: 'store', lines: 1 },
                ],
              },
              {
                direction: 'column',
                gap: 40,
                children: [
                  { id: 'facts', label: 'Facts', sub: 'world · experience', shape: 'store', lines: 10, width: 200 },
                  { id: 'obs', label: 'Observations', sub: 'consolidated beliefs', shape: 'store', lines: 7, width: 200 },
                ],
              },
            ],
          },
          {
            id: 'synth',
            label: 'Synthesized',
            gap: 40,
            children: [
              { id: 'mm', label: 'Mental Models', shape: 'store' },
              { id: 'kp', label: 'Knowledge Pages', shape: 'store' },
            ],
          },
        ],
      },
      {
        label: 'Hindsight Worker',
        direction: 'column',
        gap: 100,
        children: [
          { id: 'consolidate', label: 'Consolidation', sub: 'facts → observations' },
          { id: 'refresh', label: 'Refresh', sub: 'observations → pages' },
        ],
      },
    ],
  },
  edges: [
    { id: 'call-retain', from: 'agent', to: 'retain', label: 'retain()' },
    { id: 'call-recall', from: 'agent', to: 'recall', label: 'recall()' },
    { id: 'call-reflect', from: 'agent', to: 'reflect', label: 'reflect()' },
    { id: 'retain-docs', from: 'retain', to: 'docs' },
    { from: 'docs', to: 'chunks' },
    { id: 'extract', from: 'chunks', to: 'facts', label: 'extract' },
    { id: 's-idx', from: 'semantic', to: 'vectors' },
    { id: 'k-idx', from: 'keyword', to: 'fulltext' },
    { id: 'g-idx', from: 'graph', to: 'egraph' },
    { id: 't-idx', from: 'temporal', to: 'time' },
    { id: 'to-facts', from: 'indexes', to: 'facts' },
    { id: 'to-obs', from: 'indexes', to: 'obs' },
    { id: 'reflect-recall', from: 'reflect', to: 'recall' },
    { id: 'reflect-synth', from: 'reflect', to: 'synth' },
    { id: 'expand', from: 'reflect', to: 'sources', label: 'expand', quiet: true },
    { id: 'new-facts', from: 'facts', to: 'consolidate', label: 'new facts' },
    { id: 'write-obs', from: 'consolidate', to: 'obs' },
    { id: 'trigger', from: 'consolidate', to: 'refresh', label: 'when done' },
    { id: 'rewrite', from: 'refresh', to: 'synth', label: 'rewrite' },
  ],
  steps: [
    {
      label: 'retain()',
      flow: [
        {
          edges: { edge: 'call-retain', data: 'the conversation' },
          show: { agent: RETAIN_INPUT },
          say: 'Your agent sends what happened: a conversation, a document, a transcript.',
        },
        {
          edges: 'retain-docs',
          show: { docs: [{ tag: 'chat', tone: 'gray', text: 'Sep 22', meta: '2 messages' }] },
          say: 'The original text is stored as a document.',
        },
        {
          edges: 'docs->chunks',
          show: { chunks: [{ tag: '#1', tone: 'gray', text: '“Alice joined Google in March, she loves…”' }] },
          say: 'It is split into chunks, so the exact passage can be handed back later.',
        },
        {
          edges: 'extract',
          light: ['retain'],
          show: { facts: [OLD_WORLD, { ...WORLD, mark: 'new' }, { ...EXPERIENCE, mark: 'new' }] },
          say: 'An LLM pulls out facts: world facts about others, and experience facts about what the agent itself did. The bank already knew Alice worked at Microsoft.',
          ms: 3200,
        },
        {
          show: {
            vectors: [{ text: '2 embeddings' }],
            fulltext: [{ text: 'alice · google · research', mono: true }],
            egraph: graph(),
            time: [{ text: 'Mar 2026 · Sep 2026' }],
          },
          say: 'Each fact is indexed four ways: by meaning, by its words, by the entities it links, and by when it happened.',
          ms: 3200,
        },
        {
          edges: { edge: 'call-retain', back: true, data: '✓ stored' },
          show: { agent: [...RETAIN_INPUT, { tag: 'result', tone: 'green', text: 'stored', mark: '✓' }] },
          say: 'retain() is done. The rest happens in the background.',
          ms: 1800,
        },
        {
          edges: 'new-facts',
          show: {
            consolidate: [{ text: '2 new facts' }, { tag: 'conflict', tone: 'orange', text: 'Microsoft vs Google' }],
            obs: [{ ...OLD_OBSERVATION, mark: 'conflict' }],
          },
          say: 'Consolidation picks up the new facts and checks them against the observations the bank already holds. One disagrees: Microsoft or Google?',
          ms: 3000,
        },
        {
          edges: 'write-obs',
          show: {
            obs: [{ ...OBSERVATION, mark: 'updated' }, RESOLVED],
            consolidate: [{ text: '2 new facts' }, { tag: 'resolved', tone: 'green', text: 'state change: update, keep history' }],
          },
          say: 'It updates that observation instead of adding a second one: Alice moved from Microsoft to Google in March. Both facts stay as its sources, so the history is kept.',
          ms: 3800,
        },
        {
          edges: 'trigger',
          show: { refresh: [{ text: 'new memories in scope' }, { text: '2 pages now stale' }] },
          say: 'When consolidation finishes, it queues a refresh for every mental model and page set to refresh after it that now has new memories…',
        },
        {
          edges: 'rewrite',
          show: {
            mm: [{ tag: 'model', tone: 'orange', text: 'Team overview', meta: '+ Alice', mark: '↻' }],
            kp: [{ tag: 'page', tone: 'orange', text: 'People / Alice.md', mark: '↻' }],
          },
          say: '…and each one re-runs its question through reflect and is rewritten.',
          ms: 3000,
        },
      ],
    },
    {
      label: 'recall()',
      flow: [
        {
          edges: { edge: 'call-recall', data: '“Where does Alice work?”' },
          show: { agent: RECALL_INPUT },
          say: 'recall() finds the memories that matter for a query.',
        },
        {
          edges: [
            { edge: 's-idx', data: '≈ works at' },
            { edge: 'k-idx', data: '“Alice”' },
            { edge: 'g-idx', data: 'Alice → Google' },
          ],
          show: {
            vectors: [{ text: '“joined Google”', meta: '0.82', mark: '✓' }],
            fulltext: [{ text: '“alice” · “work”', meta: '4 hits', mark: '✓' }],
            egraph: graph(['Alice', 'Google']),
            time: [{ text: 'no date in query', meta: 'skipped' }],
          },
          say: 'Searches run at once, each through its own index: meaning, exact words and the entity graph. The time search only joins when the query names a date.',
          ms: 3400,
        },
        {
          edges: [
            { edge: 'to-facts', data: 'facts' },
            { edge: 'to-obs', data: 'observations' },
          ],
          show: {
            facts: [{ ...WORLD, mark: '✓' }, { ...EXPERIENCE, mark: '✓' }, OLD_WORLD],
            obs: [{ ...OBSERVATION, mark: '✓' }, RESOLVED],
          },
          say: 'The same indexes cover facts and observations, so both come back. They are merged and reranked; the old Microsoft fact falls below the cut.',
          ms: 3200,
        },
        {
          edges: { edge: 'call-recall', back: true, data: '3 memories, ranked' },
          show: {
            agent: [
              ...RECALL_INPUT,
              { tag: '1', tone: 'green', text: 'Alice works at Google, research team' },
              { tag: '2', tone: 'green', text: 'Alice joined Google', meta: 'Mar 2026' },
              { tag: '3', tone: 'green', text: 'I suggested Alice for ML' },
            ],
          },
          say: 'The agent gets ranked memories it can put straight into its prompt.',
          ms: 3200,
        },
      ],
    },
    {
      label: 'reflect()',
      flow: [
        {
          edges: { edge: 'call-reflect', data: '“Is Alice a good fit…?”' },
          show: { agent: REFLECT_INPUT },
          say: 'reflect() answers a question by reasoning over everything in the bank.',
        },
        {
          edges: { edge: 'reflect-synth', data: 'search_mental_models' },
          show: {
            reflect: tools(1),
            mm: [{ tag: 'model', tone: 'orange', text: 'Team overview', meta: 'Alice: research, ML', mark: '✓' }],
            kp: [{ tag: 'page', tone: 'orange', text: 'People / Alice.md', mark: '✓' }],
          },
          say: 'An agent loop decides what to look up. It starts with the most refined knowledge: mental models and knowledge pages.',
          ms: 3000,
        },
        {
          edges: { edge: 'reflect-recall', data: 'search_observations' },
          show: { reflect: tools(2, false) },
          say: 'Then observations, searched through the same indexes as recall. If new facts are still waiting to be consolidated, they are marked stale.',
          ms: 1800,
        },
        {
          edges: ['s-idx', 'k-idx', 'g-idx'],
          ms: 1400,
        },
        {
          edges: 'to-obs',
          show: { reflect: tools(2), obs: [{ ...OBSERVATION, meta: 'up to date', mark: 'cited' }, RESOLVED] },
          ms: 2400,
        },
        {
          edges: { edge: 'reflect-recall', data: 'recall("Alice")' },
          show: { reflect: tools(3, false) },
          say: 'Then raw facts through recall, for the details the summaries leave out.',
          ms: 1800,
        },
        {
          edges: ['s-idx', 'k-idx', 'g-idx'],
          show: { egraph: graph(['Alice', 'ML project']) },
          ms: 1800,
        },
        {
          edges: 'to-facts',
          show: {
            reflect: tools(3),
            facts: [
              { ...WORLD, mark: 'cited' },
              { ...EXPERIENCE, mark: 'cited' },
            ],
          },
          ms: 2400,
        },
        {
          edges: { edge: 'expand', data: 'expand → chunk' },
          show: { reflect: tools(4), chunks: [{ tag: '#1', tone: 'gray', text: '“…she loves the research team.”', mark: 'read' }] },
          say: 'When it needs the exact wording, it opens the chunk or document a fact came from.',
          ms: 2800,
        },
        {
          show: { reflect: [...tools(4), { text: '5. done', mono: true, meta: '3 sources', mark: '✓' }] },
          say: 'It stops when it has enough evidence, and writes an answer shaped by the bank’s mission and disposition. It can only cite what it found.',
          ms: 2600,
        },
        {
          edges: { edge: 'call-reflect', back: true, data: 'answer + 3 sources' },
          show: {
            agent: [
              ...REFLECT_INPUT,
              {
                tag: 'answer',
                tone: 'green',
                text: '“Yes. She joined Google’s research team in March, loves it, and I already suggested her.”',
                meta: '3 sources',
              },
            ],
          },
          say: 'The answer comes back with the memories it is based on.',
          ms: 3200,
        },
      ],
    },
  ],
};

export default {
  title: 'What Hindsight Does',
  source: 'hindsight-docs/docs/developer/index.mdx',
  props,
} satisfies Figure;
