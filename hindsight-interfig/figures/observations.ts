import type { Figure, FigRow } from '../src';

// "Observations: Knowledge Consolidation" for hindsight-docs/docs/developer/observations.mdx.
// One running example (the React → Vue journey from the docs): each step feeds one new fact
// through consolidation. Boxes process, cylinders store.

const fact = (text: string, when: string): FigRow[] => [{ tag: 'new fact', tone: 'green', text, meta: when }];
const V1: FigRow = { text: 'User prefers React for frontend development', meta: '1 source' };
const V2: FigRow = {
  text: 'User is enthusiastic about React, especially its component model',
  meta: '2 sources',
};
const V3: FigRow = {
  text: 'User was a React enthusiast who liked its component model, but has now switched to Vue',
  meta: '3 sources',
};
const FLY: FigRow = { text: 'User deploys their apps on Fly.io', meta: '1 source' };

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 40,
    children: [
      { id: 'facts', label: 'Facts', sub: 'from retain', shape: 'store', lines: 4, width: 200 },
      {
        label: 'Hindsight Worker · Consolidation',
        direction: 'column',
        gap: 36,
        children: [
          {
            id: 'find',
            label: 'Find related',
            sub: 'observations in the same tag scope',
            lines: 2,
            width: 210,
          },
          {
            id: 'llm',
            label: 'LLM decides',
            sub: 'create · update · delete',
            lines: 2,
            width: 210,
          },
          {
            id: 'dedup',
            label: 'Near-duplicate check',
            sub: 'merge or keep',
            lines: 2,
            width: 210,
          },
        ],
      },
      {
        label: 'Memory Bank',
        direction: 'column',
        gap: 36,
        children: [
          {
            id: 'obs',
            label: 'Observations',
            sub: 'one belief per facet',
            shape: 'store',
            lines: 8,
            width: 230,
          },
          {
            id: 'history',
            label: 'Observation history',
            sub: 'previous versions',
            shape: 'store',
            lines: 6,
            width: 230,
          },
        ],
      },
    ],
  },
  edges: [
    { id: 'in', from: 'facts', to: 'find', label: 'new facts' },
    { id: 'search', from: 'find', to: 'obs', label: 'recall' },
    { from: 'find', to: 'llm' },
    { from: 'llm', to: 'dedup' },
    { id: 'write', from: 'dedup', to: 'obs', label: 'write' },
    { id: 'snapshot', from: 'dedup', to: 'history', label: 'old text' },
  ],
  steps: [
    {
      label: 'refine',
      flow: [
        {
          show: {
            facts: fact('User praises React’s component model', 'week 2'),
            obs: [{ ...V1, mark: 'stale' }],
          },
          say: 'A new fact lands. Until it is consolidated, observation searches are flagged stale, so reflect checks them against the raw facts.',
          ms: 3200,
        },
        {
          edges: { edge: 'in', data: '“praises React’s component model”' },
          show: { find: [{ text: 'search observations for this fact' }] },
          say: 'Consolidation runs in the background after retain. For each new fact it recalls related observations, only within the same tag scope.',
        },
        { edges: { edge: 'search', data: 'recall' } },
        {
          edges: { edge: 'search', back: true, data: '1 related' },
          show: { find: [{ text: '1 related', meta: 'React preference', mark: '✓' }] },
        },
        {
          edges: 'find->llm',
          show: { llm: [{ tag: 'update', tone: 'blue', text: 'same facet, more evidence' }] },
          say: 'One LLM call sees the new facts next to those observations and decides, facet by facet: create, update or delete.',
          ms: 2800,
        },
        {
          edges: 'llm->dedup',
          show: { dedup: [{ text: 'no near-identical observation', mark: 'keep' }] },
          say: 'Before anything is written, each new or rewritten observation is compared with its closest neighbours. Only a near-identical one gets a merge-or-keep check.',
        },
        {
          edges: [
            { edge: 'write', data: 'rewrite + 1 source' },
            { edge: 'snapshot', data: 'v1' },
          ],
          show: {
            obs: [{ ...V2, mark: 'updated' }],
            history: [{ tag: 'v1', tone: 'gray', text: V1.text }],
          },
          say: 'The observation is rewritten with the fact attached as evidence, so its proof count goes up. The previous wording is kept in history.',
          ms: 3200,
        },
        {
          show: { obs: [{ ...V2, mark: 'fresh' }] },
          say: 'The same write marks the fact consolidated, so the observation is fresh again.',
          ms: 2000,
        },
      ],
    },
    {
      label: 'contradict',
      flow: [
        {
          show: {
            facts: fact('User switched to Vue and won’t use React anymore', 'week 3'),
            obs: [{ ...V2, mark: 'stale' }],
            history: [{ tag: 'v1', tone: 'gray', text: V1.text }],
          },
          say: 'Now a fact that contradicts what the bank believes.',
          ms: 2400,
        },
        { edges: { edge: 'in', data: '“switched to Vue”' } },
        { edges: { edge: 'search', data: 'recall' } },
        {
          edges: { edge: 'search', back: true, data: '1 related' },
          show: { find: [{ text: '1 related', meta: 'React enthusiasm', mark: '✓' }] },
        },
        {
          edges: 'find->llm',
          show: { llm: [{ tag: 'update', tone: 'orange', text: 'state change: keep the story' }] },
          say: 'A change of state is not a reason to delete. The LLM updates the belief so it records what changed, with dates when it has them.',
          ms: 3000,
        },
        {
          edges: 'llm->dedup',
          show: { dedup: [{ text: 'no near-identical observation', mark: 'keep' }] },
        },
        {
          edges: [
            { edge: 'write', data: 'rewrite + 1 source' },
            { edge: 'snapshot', data: 'v2' },
          ],
          show: {
            obs: [{ ...V3, mark: 'updated' }],
            history: [
              { tag: 'v1', tone: 'gray', text: V1.text },
              { tag: 'v2', tone: 'gray', text: V2.text },
            ],
          },
          say: 'The observation now tells the whole journey, not just “prefers Vue”. It rests on all three facts, and both older versions stay in history.',
          ms: 3600,
        },
        { show: { obs: [{ ...V3, mark: 'fresh' }] }, ms: 2000 },
      ],
    },
    {
      label: 'something new',
      flow: [
        {
          show: {
            facts: fact('User deploys their apps on Fly.io', 'week 3'),
            obs: [{ ...V3, mark: 'stale' }],
            history: [
              { tag: 'v1', tone: 'gray', text: V1.text },
              { tag: 'v2', tone: 'gray', text: V2.text },
            ],
          },
          say: 'A fact about something the bank has no belief on yet.',
          ms: 2400,
        },
        { edges: { edge: 'in', data: '“deploys on Fly.io”' } },
        { edges: { edge: 'search', data: 'recall' } },
        {
          edges: { edge: 'search', back: true, data: 'nothing related' },
          show: { find: [{ text: 'no related observation', mark: '—' }] },
        },
        {
          edges: 'find->llm',
          show: { llm: [{ tag: 'create', tone: 'green', text: 'new facet, nothing to update' }] },
          say: 'Nothing covers this facet, so the LLM creates a new observation instead of bending an unrelated one.',
          ms: 2800,
        },
        {
          edges: 'llm->dedup',
          show: {
            dedup: [{ text: 'closest: the React/Vue belief', meta: 'not near-identical', mark: 'keep' }],
          },
          say: 'The near-duplicate check keeps it: different facets stay separate observations.',
          ms: 2800,
        },
        {
          edges: { edge: 'write', data: 'new observation' },
          show: {
            obs: [
              { ...V3, mark: 'fresh' },
              { ...FLY, mark: 'new' },
            ],
          },
          say: 'The new observation starts with one source. It will gain evidence as more facts repeat it.',
          ms: 2800,
        },
      ],
    },
  ],
};

export default {
  title: 'Observation Consolidation',
  source: 'hindsight-docs/docs/developer/observations.mdx',
  props,
} satisfies Figure;
