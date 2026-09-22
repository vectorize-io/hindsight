import { createElement as h } from 'react';
import { MiniGraph, type Figure, type FigRow } from '../src';

// "Multilingual Support" for hindsight-docs/docs/developer/multilingual.md: Chinese in, Chinese stored,
// Chinese out. There is no separate language detector: the LLM is told to keep the input's language.
// The two language-sensitive pieces are the indexes: the embedding model and the full-text tokenizer.

const graph = (lit: string[] = []) =>
  h(MiniGraph, {
    nodes: ['王芳', 'Google', 'Microsoft', 'Amazon'],
    links: [
      ['王芳', 'Google'],
      ['王芳', 'Microsoft'],
      ['王芳', 'Amazon'],
    ],
    lit,
  });

const RETAIN_INPUT: FigRow[] = [
  {
    tag: 'content',
    tone: 'gray',
    text: '“王芳在Google北京办公室工作，她是一名高级产品经理。之前她在Microsoft和Amazon工作过。”',
  },
  { tag: 'context', tone: 'gray', text: '员工资料' },
];
const RECALL_INPUT: FigRow[] = [{ tag: 'query', tone: 'gray', text: '“王芳在哪里工作？”' }];
const REFLECT_INPUT: FigRow[] = [{ tag: 'question', tone: 'gray', text: '“王芳有哪些工作经历？”' }];
const FACT_1: FigRow = {
  tag: 'world',
  tone: 'blue',
  text: '王芳在Google北京办公室工作，担任高级产品经理',
};
const FACT_2: FigRow = { tag: 'world', tone: 'blue', text: '王芳曾在Microsoft和Amazon工作过' };

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 40,
    children: [
      { id: 'agent', label: 'Your AI Agent', lines: 9, width: 200 },
      {
        label: 'Hindsight API',
        direction: 'column',
        gap: 24,
        children: [
          { id: 'retain', label: 'Retain', sub: 'LLM extraction', lines: 3, width: 210 },
          { id: 'recall', label: 'Recall', sub: 'search', lines: 2, width: 210 },
          { id: 'reflect', label: 'Reflect', sub: 'agent loop', lines: 3, width: 210 },
        ],
      },
      {
        label: 'Memory Bank',
        gap: 40,
        children: [
          {
            id: 'indexes',
            label: 'Indexes',
            direction: 'column',
            gap: 12,
            children: [
              { id: 'vectors', label: 'Vectors', sub: 'embedding model', shape: 'store', lines: 3 },
              { id: 'fulltext', label: 'Full text', sub: 'tokenizer', shape: 'store', lines: 3 },
            ],
          },
          {
            direction: 'column',
            gap: 24,
            children: [
              { id: 'facts', label: 'Facts', shape: 'store', lines: 6, width: 250 },
              {
                id: 'entities',
                label: 'Entities',
                sub: 'names as written',
                shape: 'store',
                lines: 4,
                width: 250,
              },
            ],
          },
        ],
      },
    ],
  },
  edges: [
    { id: 'call-retain', from: 'agent', to: 'retain', label: 'retain()' },
    { id: 'call-recall', from: 'agent', to: 'recall', label: 'recall()' },
    { id: 'call-reflect', from: 'agent', to: 'reflect', label: 'reflect()' },
    { id: 'r-facts', from: 'retain', to: 'facts', quiet: true },
    { id: 'r-ent', from: 'retain', to: 'entities', quiet: true },
    { id: 'r-idx', from: 'retain', to: 'indexes', label: 'index' },
    { id: 'q-vec', from: 'recall', to: 'vectors' },
    { id: 'q-ft', from: 'recall', to: 'fulltext' },
    { id: 'idx-facts', from: 'indexes', to: 'facts', label: 'point to' },
    { id: 'reflect-recall', from: 'reflect', to: 'recall', label: 'uses' },
  ],
  steps: [
    {
      label: 'retain()',
      flow: [
        {
          edges: { edge: 'call-retain', data: 'Chinese content' },
          show: { agent: RETAIN_INPUT },
          say: 'Your agent retains Chinese text that mentions English company names.',
        },
        {
          show: {
            retain: [
              { text: 'Input language: Chinese' },
              { text: 'Write facts in Chinese', mark: '✓' },
              { text: 'Never translate names', mark: '✓' },
            ],
          },
          light: ['retain'],
          say: 'There is no separate language detector. The LLM is told to write facts in the language of the input, and to keep names as they are.',
          ms: 3400,
        },
        {
          edges: ['r-facts', 'r-ent'],
          show: {
            facts: [
              { ...FACT_1, mark: 'new' },
              { ...FACT_2, mark: 'new' },
            ],
            entities: graph(['王芳', 'Google', 'Microsoft', 'Amazon']),
          },
          say: 'Facts are stored in Chinese. 王芳 stays 王芳, not “Wang Fang”, and Google stays Google.',
          ms: 3400,
        },
        {
          edges: 'r-idx',
          show: {
            vectors: [{ text: 'bge-m3', mono: true, meta: '100+ languages', mark: '✓' }],
            fulltext: [{ text: 'pgroonga', mono: true, meta: 'splits Chinese text', mark: '✓' }],
          },
          say: 'Indexing is where language matters. The default embedding model and keyword index are English-only; pick a multilingual model and a tokenizer that can split Chinese.',
          ms: 3800,
        },
        {
          edges: { edge: 'call-retain', back: true, data: '✓ 2 facts' },
          show: {
            agent: [...RETAIN_INPUT, { tag: 'result', tone: 'green', text: '2 facts stored', mark: '✓' }],
          },
          say: 'Stored, in the language it came in.',
          ms: 1800,
        },
      ],
    },
    {
      label: 'recall()',
      flow: [
        {
          edges: { edge: 'call-recall', data: '“王芳在哪里工作？”' },
          show: { agent: RECALL_INPUT },
          say: 'A query in Chinese.',
        },
        {
          edges: [
            { edge: 'q-vec', data: '≈ 在哪里工作' },
            { edge: 'q-ft', data: '“王芳”' },
          ],
          show: {
            vectors: [{ text: 'matches by meaning', meta: '工作 ≈ 担任', mark: '✓' }],
            fulltext: [{ text: 'matches the name', meta: '王芳 in 2 facts', mark: '✓' }],
          },
          say: 'The multilingual embeddings match by meaning; the tokenizer lets the exact name 王芳 match as a word.',
          ms: 3200,
        },
        {
          edges: 'idx-facts',
          show: {
            facts: [
              { ...FACT_1, mark: '✓' },
              { ...FACT_2, mark: '✓' },
            ],
          },
          say: 'Both point to the Chinese facts. The reranker that then orders them is English-only by default too, so pick a multilingual one.',
          ms: 2000,
        },
        {
          edges: { edge: 'call-recall', back: true, data: '2 facts' },
          show: {
            agent: [
              ...RECALL_INPUT,
              { tag: '1', tone: 'green', text: '王芳在Google北京办公室工作，担任高级产品经理' },
              { tag: '2', tone: 'green', text: '王芳曾在Microsoft和Amazon工作过' },
            ],
          },
          say: 'The facts come back exactly as stored. Nothing was translated on the way.',
          ms: 2800,
        },
      ],
    },
    {
      label: 'reflect()',
      flow: [
        {
          edges: { edge: 'call-reflect', data: '“王芳有哪些工作经历？”' },
          show: { agent: REFLECT_INPUT },
          say: 'A question in Chinese.',
        },
        {
          edges: { edge: 'reflect-recall', data: 'recall' },
          show: { reflect: [{ text: 'Question language: Chinese' }, { text: 'searching…' }] },
          say: 'Reflect searches the bank like before…',
        },
        {
          edges: ['q-vec', 'q-ft'],
          ms: 1600,
          say: '…through the same indexes…',
        },
        {
          edges: 'idx-facts',
          show: {
            facts: [
              { ...FACT_1, mark: 'cited' },
              { ...FACT_2, mark: 'cited' },
            ],
          },
          ms: 2000,
          say: '…and finds both facts.',
        },
        {
          show: {
            reflect: [{ text: 'Question language: Chinese' }, { text: '2 facts cited' }, { text: 'Answer in Chinese', mark: '✓' }],
          },
          light: ['reflect'],
          say: 'The answer is written in the language of the question.',
          ms: 2400,
        },
        {
          edges: { edge: 'call-reflect', back: true, data: 'answer in Chinese' },
          show: {
            agent: [
              ...REFLECT_INPUT,
              {
                tag: 'answer',
                tone: 'green',
                text: '“王芳目前在Google北京办公室担任高级产品经理，之前在Microsoft和Amazon工作过。”',
              },
            ],
          },
          say: 'To force one language everywhere instead, set HINDSIGHT_API_LLM_OUTPUT_LANGUAGE.',
          ms: 3200,
        },
      ],
    },
  ],
};

export default {
  title: 'Multilingual Support',
  source: 'hindsight-docs/docs/developer/multilingual.md',
  props,
} satisfies Figure;
