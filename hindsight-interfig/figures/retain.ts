import { createElement as h } from 'react';
import { MiniGraph, type Figure, type FigRow } from '../src';

// "What Retain Does" for hindsight-docs/docs/developer/retain.md: the retain pipeline up close.
// Boxes process, cylinders store. One running example: Bob tells the agent about Alice.

const graph = (lit: string[]) =>
  h(MiniGraph, {
    nodes: ['Alice Chen', 'Zurich office'],
    links: [['Alice Chen', 'Zurich office']],
    lit,
  });

const INPUT_1: FigRow[] = [
  {
    tag: 'content',
    tone: 'gray',
    text: '“Alice Chen worked 80-hour weeks and burned out, so she moved to the Zurich office in June 2024.”',
  },
  { tag: 'context', tone: 'gray', text: 'Bob is talking about his team' },
  { tag: 'timestamp', tone: 'gray', text: 'Jan 10, 2025' },
];
const INPUT_2: FigRow[] = [
  { tag: 'content', tone: 'gray', text: '“Alice C. now runs the Zurich office.”' },
  { tag: 'context', tone: 'gray', text: 'Bob is talking about his team' },
  { tag: 'timestamp', tone: 'gray', text: 'Mar 3, 2025' },
];
const FACTS: FigRow[] = [
  { tag: 'world', tone: 'blue', text: 'Alice Chen worked 80-hour weeks' },
  { tag: 'world', tone: 'blue', text: 'Alice Chen burned out' },
  {
    tag: 'world',
    tone: 'blue',
    text: 'Alice Chen moved to the Zurich office',
    meta: 'happened Jun 2024 · learned Jan 2025',
  },
];
const LINKS: FigRow[] = [
  { text: 'entity', mono: true, meta: 'all 3 facts share Alice Chen' },
  { text: 'temporal', mono: true, meta: 'facts close in time' },
  { text: 'semantic', mono: true, meta: 'facts with a similar meaning' },
  { text: 'caused_by', mono: true, meta: 'burned out ← 80-hour weeks' },
];

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 40,
    children: [
      { id: 'input', label: 'retain()', sub: 'content + context', lines: 10, width: 210 },
      {
        label: 'Retain pipeline',
        direction: 'column',
        gap: 20,
        children: [
          { id: 'chunk', label: 'Chunking', sub: 'split long content', lines: 2, width: 210 },
          {
            id: 'extract',
            label: 'LLM extraction',
            sub: 'what · when · where · who · why',
            lines: 7,
            width: 210,
          },
          {
            id: 'resolve',
            label: 'Entity resolution',
            sub: 'one name, one entity',
            lines: 4,
            width: 210,
          },
          { id: 'link', label: 'Embed & link', sub: 'vectors + connections', lines: 2, width: 210 },
        ],
      },
      {
        label: 'Memory Bank',
        direction: 'column',
        gap: 28,
        children: [
          {
            label: 'Sources',
            gap: 40,
            children: [
              { id: 'docs', label: 'Documents', shape: 'store', lines: 2 },
              { id: 'chunks', label: 'Chunks', shape: 'store', lines: 3 },
            ],
          },
          {
            id: 'facts',
            label: 'Facts',
            sub: 'world · experience',
            shape: 'store',
            lines: 8,
            width: 400,
          },
          {
            label: 'Graph',
            gap: 40,
            children: [
              { id: 'entities', label: 'Entities', shape: 'store', lines: 4, width: 180 },
              { id: 'links', label: 'Links', shape: 'store', lines: 8, width: 230 },
            ],
          },
        ],
      },
    ],
  },
  edges: [
    { id: 'call', from: 'input', to: 'chunk', label: 'retain()' },
    { id: 'c-docs', from: 'chunk', to: 'docs' },
    { id: 'c-chunks', from: 'chunk', to: 'chunks' },
    { id: 'c-x', from: 'chunk', to: 'extract' },
    { id: 'x-facts', from: 'extract', to: 'facts', label: 'facts' },
    { id: 'x-r', from: 'extract', to: 'resolve' },
    { id: 'r-ent', from: 'resolve', to: 'entities', label: 'entities' },
    { id: 'r-l', from: 'resolve', to: 'link' },
    { id: 'l-links', from: 'link', to: 'links', label: 'links', quiet: true },
  ],
  steps: [
    {
      label: 'new facts',
      flow: [
        {
          edges: { edge: 'call', data: 'content + context' },
          show: { input: INPUT_1 },
          say: 'You send content, a context that says who is speaking, and when it was said.',
        },
        {
          edges: ['c-docs', 'c-chunks'],
          show: {
            chunk: [{ text: '1 chunk', meta: 'long content is split' }],
            docs: [{ text: 'notes-2025-01-10', meta: 'Jan 10, 2025' }],
            chunks: [{ tag: '#1', tone: 'gray', text: '“Alice Chen worked 80-hour weeks and burned out…”' }],
          },
          say: 'The original is kept as a document and split into chunks, so the exact passage can be handed back later.',
        },
        {
          edges: { edge: 'c-x', data: 'chunk #1' },
          show: {
            extract: [
              { text: 'What: Alice moved to the Zurich office' },
              { text: 'When: June 2024' },
              { text: 'Where: Zurich' },
              { text: 'Who: Alice Chen' },
              { text: 'Why: she burned out' },
            ],
          },
          say: 'An LLM reads each chunk and pulls out facts with what, when, where, who and why. It keeps the reason, not just the event. Each fact is embedded right away.',
          ms: 3400,
        },
        {
          edges: { edge: 'x-r', data: '“Alice Chen”, “Zurich office”' },
          show: {
            resolve: [
              { text: '“Alice Chen”', meta: 'not in the bank yet', mark: 'new' },
              { text: '“Zurich office”', meta: 'not in the bank yet', mark: 'new' },
            ],
          },
          say: 'Entity resolution decides who each name is. It compares each name with the entities the bank already has. Neither is known yet, so both become new entities.',
          ms: 3200,
        },
        {
          edges: 'r-ent',
          show: { entities: graph(['Alice Chen', 'Zurich office']) },
          say: 'Entities are stored once, and every fact that mentions them points to them.',
        },
        {
          edges: 'x-facts',
          show: { facts: FACTS.map((f) => ({ ...f, mark: 'new' })) },
          say: 'The facts are world facts: Bob is talking about Alice, not about the agent. Each keeps two times: when it happened and when Hindsight learned it.',
          ms: 3600,
        },
        {
          edges: ['r-l', 'l-links'],
          show: {
            link: [{ text: '3 facts embedded' }, { text: '4 kinds of links' }],
            links: LINKS,
          },
          say: 'Finally the facts are linked: through shared entities, closeness in time, similar meaning (from the embeddings), and cause and effect.',
          ms: 3800,
        },
        {
          edges: { edge: 'call', back: true, data: '✓ 3 facts' },
          show: {
            input: [...INPUT_1, { tag: 'result', tone: 'green', text: '3 facts stored', mark: '✓' }],
          },
          say: 'retain() is done. Observations are built from these facts later, in the background.',
          ms: 2400,
        },
      ],
    },
    {
      label: 'same person, new name',
      flow: [
        {
          edges: { edge: 'call', data: 'content + context' },
          show: { input: INPUT_2 },
          say: 'Two months later, Bob mentions “Alice C.”',
        },
        {
          edges: ['c-docs', 'c-chunks'],
          show: {
            docs: [
              { text: 'notes-2025-01-10', meta: 'Jan 10, 2025' },
              { text: 'notes-2025-03-03', meta: 'Mar 3, 2025', mark: 'new' },
            ],
            chunks: [{ tag: '#1', tone: 'gray', text: '“Alice C. now runs the Zurich office.”' }],
          },
          say: 'Same path: stored as a new document, split into chunks…',
        },
        {
          edges: 'c-x',
          show: {
            extract: [{ text: 'What: Alice C. runs the Zurich office' }, { text: 'When: March 2025' }, { text: 'Who: Alice C.' }],
          },
          say: '…and read by the LLM. The fact names “Alice C.”, a name the bank has never seen.',
          ms: 2600,
        },
        {
          edges: { edge: 'x-r', data: '“Alice C.”, “Zurich office”' },
          show: {
            resolve: [
              { text: '“Alice C.”', meta: 'close to Alice Chen' },
              { text: 'next to Zurich office', meta: 'seen with Alice Chen before', mark: '✓' },
              { text: '→ Alice Chen', mark: 'merged' },
              { text: '“Zurich office”', meta: 'same name', mark: 'match' },
            ],
          },
          say: 'The name is close to Alice Chen, and the same fact names the Zurich office she is already linked to. Together that is enough: same person.',
          ms: 3400,
        },
        {
          edges: 'r-ent',
          show: { entities: graph(['Alice Chen', 'Zurich office']) },
          say: 'No new entity is created: this mention of “Alice C.” counts as Alice Chen.',
        },
        {
          edges: 'x-facts',
          show: {
            facts: [
              ...FACTS,
              {
                tag: 'world',
                tone: 'blue',
                text: 'Alice C. runs the Zurich office',
                meta: 'about Alice Chen · Mar 2025',
                mark: 'new',
              },
            ],
          },
          say: 'The fact keeps its own wording, but it points to Alice Chen, so it joins everything else about her. Ask about Alice later and you get all of it.',
          ms: 3200,
        },
        {
          edges: ['r-l', 'l-links'],
          show: {
            links: [
              {
                text: 'entity',
                mono: true,
                meta: 'joins 3 earlier facts about Alice Chen',
                mark: 'new',
              },
              { text: 'semantic', mono: true, meta: '≈ Alice Chen moved to the Zurich office' },
            ],
          },
          say: 'Its entity link ties it to the three facts from January.',
          ms: 2600,
        },
        {
          edges: { edge: 'call', back: true, data: '✓ 1 fact' },
          show: {
            input: [...INPUT_2, { tag: 'result', tone: 'green', text: '1 fact stored', mark: '✓' }],
          },
          say: 'Done: one more fact about the same Alice.',
          ms: 2000,
        },
      ],
    },
  ],
};

export default {
  title: 'What Retain Does',
  source: 'hindsight-docs/docs/developer/retain.md',
  props,
} satisfies Figure;
