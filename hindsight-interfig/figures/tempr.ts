import { createElement as h } from 'react';
import { MiniGraph, type Figure, type FigRow } from '../src';

// "Multi-Strategy Retrieval (TEMPR)" for hindsight-docs/docs/developer/index.mdx and retrieval.md.
// One query through the whole recall pipeline: four arms, each through its own index into the same
// memories (world, experience, observation), then RRF fusion → cross-encoder rerank → boosts → token budget.
// Boxes process, cylinders store. Scores follow the formulas in retrieval.md (RRF k = 60).

const graph = (lit: string[] = []) =>
  h(MiniGraph, {
    nodes: ['Alice', 'Google', 'ML project', 'Microsoft'],
    links: [
      ['Alice', 'Google'],
      ['Alice', 'ML project'],
      ['Alice', 'Microsoft'],
    ],
    lit,
  });

const QUERY: FigRow[] = [{ tag: 'query', tone: 'gray', text: '“What did Alice do in March 2026?”' }];

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 32,
    children: [
      { id: 'agent', label: 'Your AI Agent', lines: 8, width: 180 },
      {
        id: 'recall',
        label: 'Recall · 4 arms in parallel',
        direction: 'column',
        gap: 10,
        children: [
          { id: 'semantic', label: 'Semantic', sub: 'by meaning' },
          { id: 'keyword', label: 'Keyword', sub: 'exact words (BM25)' },
          { id: 'graph', label: 'Graph', sub: 'via entities' },
          { id: 'temporal', label: 'Temporal', sub: 'by time' },
        ],
      },
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
        id: 'units',
        label: 'Memories',
        sub: 'world · experience · observation',
        shape: 'store',
        lines: 12,
        width: 196,
      },
      {
        id: 'ranking',
        label: 'Ranking',
        direction: 'column',
        gap: 24,
        children: [
          { id: 'rrf', label: 'RRF fusion', sub: 'Σ 1 / (60 + rank)', lines: 4, width: 196 },
          {
            id: 'rerank',
            label: 'Cross-encoder',
            sub: 'reads query + memory',
            lines: 4,
            width: 196,
          },
          { id: 'boosts', label: 'Boosts', sub: 'recency · time · proof', lines: 4, width: 196 },
          { id: 'budget', label: 'Token budget', sub: 'max_tokens', lines: 4, width: 196 },
        ],
      },
    ],
  },
  edges: [
    { id: 'call', from: 'agent', to: 'recall', label: 'recall()' },
    { id: 's-idx', from: 'semantic', to: 'vectors' },
    { id: 'k-idx', from: 'keyword', to: 'fulltext' },
    { id: 'g-idx', from: 'graph', to: 'egraph' },
    { id: 't-idx', from: 'temporal', to: 'time' },
    { id: 'lookup', from: 'indexes', to: 'units', label: 'point to' },
    { id: 'lists', from: 'units', to: 'rrf', label: '4 ranked lists' },
    { from: 'rrf', to: 'rerank' },
    { from: 'rerank', to: 'boosts' },
    { from: 'boosts', to: 'budget' },
    { id: 'results', from: 'budget', to: 'agent', around: 'below', quiet: true },
  ],
  steps: [
    {
      label: 'recall()',
      flow: [
        {
          edges: { edge: 'call', data: '“What did Alice do in March 2026?”' },
          show: { agent: QUERY },
          say: 'recall() gets a query. Nothing is decided yet about which kind of search fits it best, so every arm that applies runs.',
        },
        {
          edges: [
            { edge: 's-idx', data: '≈ “joined”, “moved”' },
            { edge: 'k-idx', data: '“Alice”' },
            { edge: 'g-idx', data: 'Alice → …' },
            { edge: 't-idx', data: 'Mar 2026' },
          ],
          show: {
            vectors: [{ text: 'nearest by meaning', mark: '✓' }],
            fulltext: [{ text: '“alice”: 4 hits', mono: true, mark: '✓' }],
            egraph: graph(['Alice', 'Google', 'ML project', 'Microsoft']),
            time: [{ text: 'Mar 1 – Mar 31, 2026', mark: '✓' }],
          },
          say: 'Each arm searches its own index: meaning (vectors), exact words (BM25), the entity graph, and time. “March 2026” becomes a date range; a query with no date skips the time arm.',
          ms: 3600,
        },
        {
          edges: { edge: 'lookup', data: 'same memories' },
          show: {
            units: [
              {
                tag: 'world',
                tone: 'blue',
                text: 'Alice joined Google',
                meta: 'Mar 2026',
                mark: '4/4',
              },
              {
                tag: 'observation',
                tone: 'green',
                text: 'Alice works at Google, research team',
                mark: '3/4',
              },
              {
                tag: 'experience',
                tone: 'purple',
                text: 'I suggested Alice for ML',
                meta: 'Mar 2026',
                mark: '3/4',
              },
              {
                tag: 'world',
                tone: 'blue',
                text: 'Alice works at Microsoft',
                meta: 'Jan 2025',
                mark: '2/4',
              },
            ],
          },
          say: 'All four point into the same memories. Each arm returns its own ranked list, and facts and observations compete in every one. The mark shows how many arms found each.',
          ms: 3600,
        },
        {
          edges: 'lists',
          show: {
            rrf: [
              { text: 'joined Google', meta: '0.066' },
              { text: 'works at Google', meta: '0.048' },
              { text: 'suggested for ML', meta: '0.047' },
              { text: 'Microsoft', meta: '0.031' },
            ],
          },
          say: 'RRF fusion merges the lists by rank, not raw score: a memory found near the top by several arms beats one found by a single arm.',
          ms: 3000,
        },
        {
          edges: 'rrf->rerank',
          show: {
            rerank: [
              { text: 'joined Google', meta: '0.86' },
              { text: 'works at Google', meta: '0.84' },
              { text: 'suggested for ML', meta: '0.66' },
              { text: 'Microsoft', meta: '0.21' },
            ],
          },
          say: 'The top candidates (up to 300) go to a cross-encoder, which reads the query and each memory together and scores how well they match.',
          ms: 3000,
        },
        {
          edges: 'rerank->boosts',
          show: {
            boosts: [
              { text: 'joined Google', meta: '×1.09', mark: '0.94' },
              { text: 'works at Google', meta: '×1.05', mark: '0.88' },
              { text: 'suggested for ML', meta: '×1.07', mark: '0.71' },
              { text: 'Microsoft', meta: '×0.83', mark: '0.17' },
            ],
          },
          say: 'Small multiplicative boosts nudge the score: recent memories, memories inside the asked time range, and observations backed by more evidence.',
          ms: 3200,
        },
        {
          edges: 'boosts->budget',
          show: {
            budget: [
              { text: 'joined Google', mark: '✓' },
              { text: 'works at Google', mark: '✓' },
              { text: 'suggested for ML', mark: '✓' },
              { text: 'Microsoft', meta: 'over budget', mark: '✗' },
            ],
          },
          say: 'Results are packed best-first until max_tokens is used up. Only the memory text counts toward the budget.',
          ms: 3000,
        },
        {
          edges: { edge: 'results', data: '3 memories, ranked' },
          show: {
            agent: [
              ...QUERY,
              { tag: '1', tone: 'green', text: 'Alice joined Google', meta: 'Mar 2026' },
              { tag: '2', tone: 'green', text: 'Alice works at Google, research team' },
              { tag: '3', tone: 'green', text: 'I suggested Alice for ML' },
            ],
          },
          say: 'The agent gets a short, ranked list it can put straight into its prompt.',
          ms: 3200,
        },
      ],
    },
  ],
};

export default {
  title: 'Multi-Strategy Retrieval (TEMPR)',
  source: 'hindsight-docs/docs/developer/retrieval.md',
  props,
} satisfies Figure;
