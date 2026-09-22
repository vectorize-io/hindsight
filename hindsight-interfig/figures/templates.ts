import type { Figure, FigRow } from '../src';

// Bank Templates for hindsight-docs/blog/2026-04-10-templates-hub-hindsight-050.md: a tuned bank's setup
// travels as a JSON manifest (config overrides, mental models, directives), never its memories.

const MANIFEST: FigRow[] = [
  { tag: 'version', tone: 'gray', text: '"1"', mono: true },
  { tag: 'bank', tone: 'blue', text: 'retain_mission, enable_observations' },
  {
    tag: 'mental_models',
    tone: 'orange',
    text: 'project-context',
    meta: 'refresh after consolidation',
  },
  { tag: 'directives', tone: 'purple', text: 'Cite file paths' },
];
const CONFIG: FigRow[] = [
  { text: 'retain_mission', mono: true, meta: 'set' },
  { text: 'enable_observations', mono: true, meta: 'true' },
];
const DIRECTIVE: FigRow[] = [{ text: 'Cite file paths' }];
// The Hub's real Coding Agent template (hindsight-docs/src/data/templates/coding-agent.json).
const CODING: FigRow[] = [
  { tag: 'version', tone: 'gray', text: '"1"', mono: true },
  { tag: 'bank', tone: 'blue', text: 'retain_mission, enable_observations, observations_mission, disposition_*' },
  {
    tag: 'mental_models',
    tone: 'orange',
    text: 'project-context, developer-preferences, review-patterns',
  },
];

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 32,
    children: [
      {
        id: 'src',
        label: 'Tuned bank',
        direction: 'column',
        gap: 12,
        children: [
          { id: 's-config', label: 'Config', sub: 'per-bank overrides', shape: 'store', lines: 3 },
          { id: 's-mm', label: 'Mental Models', shape: 'store', lines: 3 },
          { id: 's-dir', label: 'Directives', shape: 'store', lines: 1 },
          { id: 's-mem', label: 'Memories', shape: 'store', lines: 1 },
        ],
      },
      { id: 'export', label: 'Export', lines: 4 },
      {
        direction: 'column',
        gap: 40,
        children: [
          {
            id: 'hub',
            label: 'Templates Hub',
            sub: 'ready-made manifests',
            shape: 'store',
            lines: 3,
            width: 200,
          },
          { id: 'manifest', label: 'Manifest', sub: 'JSON', shape: 'store', lines: 12, width: 200 },
        ],
      },
      {
        label: 'Hindsight',
        direction: 'column',
        gap: 40,
        children: [
          { id: 'import', label: 'Import', lines: 5, width: 170 },
          { id: 'refresh', label: 'Refresh', sub: 'mental model content', lines: 2 },
        ],
      },
      {
        id: 'dst',
        label: 'New bank',
        direction: 'column',
        gap: 12,
        children: [
          { id: 'd-config', label: 'Config', sub: 'per-bank overrides', shape: 'store', lines: 4 },
          { id: 'd-mm', label: 'Mental Models', shape: 'store', lines: 3 },
          { id: 'd-dir', label: 'Directives', shape: 'store', lines: 1 },
          { id: 'd-mem', label: 'Memories', shape: 'store', lines: 1 },
        ],
      },
    ],
  },
  edges: [
    { id: 'out', from: 'src', to: 'export', label: 'export' },
    { id: 'to-manifest', from: 'export', to: 'manifest' },
    { id: 'from-hub', from: 'hub', to: 'manifest', label: 'copy' },
    { id: 'in', from: 'manifest', to: 'import', label: 'import' },
    { id: 'apply', from: 'import', to: 'dst' },
    { id: 'queue', from: 'import', to: 'refresh' },
    { id: 'gen', from: 'refresh', to: 'd-mm' },
  ],
  steps: [
    {
      label: 'export',
      flow: [
        {
          show: {
            's-config': CONFIG,
            's-mm': [{ text: 'project-context', mono: true, meta: 'filled' }],
            's-dir': DIRECTIVE,
            's-mem': [{ text: '1,240 memories' }],
          },
          say: 'A bank you have tuned: its own config, mental models, directives, and the memories it has learned.',
          ms: 2800,
        },
        {
          edges: { edge: 'out', data: 'export' },
          show: {
            export: [{ text: 'only what this bank overrides' }, { text: 'no server defaults, no API keys' }],
          },
          say: 'Export reads the setup. It keeps only the config set on this bank — not the server defaults, never credentials — so the manifest stays portable.',
          ms: 3000,
        },
        {
          edges: 'to-manifest',
          show: { manifest: MANIFEST, 's-mem': [{ text: '1,240 memories', mark: 'stay' }] },
          say: 'The manifest is plain JSON: config, mental model definitions (not their content), directives. Memories are never in it.',
          ms: 3200,
        },
      ],
    },
    {
      label: 'import',
      flow: [
        {
          edges: { edge: 'in', data: 'import' },
          show: { manifest: MANIFEST },
          say: 'Import it into another bank. If that bank does not exist yet, it is created.',
        },
        {
          show: {
            import: [
              { text: 'manifest valid', mark: '✓' },
              { text: 'config: 2 overrides' },
              { text: 'mental models: 1 new' },
              { text: 'directives: 1 new' },
            ],
          },
          say: 'The manifest is checked first. A dry run stops here and shows what would change.',
          ms: 2800,
        },
        {
          edges: 'apply',
          show: {
            'd-config': CONFIG,
            'd-mm': [{ text: 'project-context', mono: true, meta: 'no content yet' }],
            'd-dir': DIRECTIVE,
            'd-mem': [{ text: 'empty' }],
          },
          say: 'Config becomes per-bank overrides. Mental models match by id and directives by name: new ones are created, existing ones updated.',
          ms: 3400,
        },
        {
          edges: { edge: 'queue', data: 'operation_ids' },
          show: { refresh: [{ text: '1 refresh queued' }, { text: 'nothing to read yet' }] },
          say: 'Mental model content is written in the background. The response returns operation ids to track.',
        },
        {
          edges: 'gen',
          show: {
            'd-mm': [{ text: 'project-context', mono: true, meta: 'fills as memories arrive' }],
          },
          say: 'The new bank has no memories yet, so there is nothing to read. The model refreshes after each consolidation, so it fills in as the bank learns. Same behavior, new bank.',
          ms: 3200,
        },
      ],
    },
    {
      label: 'from the Hub',
      flow: [
        {
          show: {
            hub: [{ text: 'Coding Agent', mark: '✓' }, { text: 'Conversation' }, { text: 'Personal Assistant', meta: '+ more' }],
          },
          light: ['hub'],
          say: 'Or skip the export: the Templates Hub has ready-made manifests you can read before using.',
        },
        {
          edges: { edge: 'from-hub', data: 'Coding Agent' },
          show: { manifest: CODING },
          say: 'Copy one. It is the same JSON an export produces.',
        },
        { edges: { edge: 'in', data: 'import' }, say: 'Then import it the same way.' },
        {
          edges: 'apply',
          show: {
            'd-config': [
              { text: 'retain_mission', mono: true, meta: 'set' },
              { text: 'enable_observations', mono: true, meta: 'true' },
              { text: 'observations_mission', mono: true, meta: 'set' },
              { text: 'disposition_*', mono: true, meta: 'literalism 5 · skepticism 3' },
            ],
            'd-mm': [{ text: '3 models', meta: 'fill as memories arrive' }],
            'd-dir': [{ text: 'none in this template' }],
            'd-mem': [{ text: 'empty' }],
          },
          say: 'A new bank starts from a known-good setup instead of from scratch.',
          ms: 3000,
        },
      ],
    },
  ],
};

export default {
  title: 'Bank Templates',
  source: 'hindsight-docs/blog/2026-04-10-templates-hub-hindsight-050.md',
  props,
} satisfies Figure;
