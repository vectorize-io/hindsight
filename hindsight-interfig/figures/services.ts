import type { Figure, FigRow } from '../src';

// "Services" for hindsight-docs/docs/developer/services.md: which process does what, and how work moves
// between them. The API answers requests itself; anything slow is a row in async_operations that any
// worker (the API's internal one, or dedicated hindsight-worker processes) claims with FOR UPDATE SKIP LOCKED.

const RECALL_IN: FigRow[] = [{ tag: 'recall', tone: 'gray', text: '“Where does Alice work?”' }];
const RETAIN_IN: FigRow[] = [{ tag: 'retain', tone: 'gray', text: '“Alice joined Google in March.”', meta: 'async=true' }];
const op = (type: string, id: string, status: string, mark?: string): FigRow => ({
  tag: type,
  tone: 'orange',
  text: id,
  meta: status,
  mark,
});

const props: Figure['props'] = {
  speed: 1900,
  layout: {
    gap: 48,
    children: [
      {
        label: 'Clients',
        direction: 'column',
        gap: 28,
        children: [
          { id: 'app', label: 'Your app', sub: 'SDK or MCP client', width: 190 },
          { id: 'cp', label: 'Control Plane', sub: 'web UI · :9999', width: 190 },
        ],
      },
      {
        id: 'api',
        label: 'hindsight-api',
        direction: 'column',
        gap: 28,
        children: [
          { id: 'http', label: 'HTTP API', sub: 'REST · /mcp · :8888', width: 190 },
          { id: 'iworker', label: 'Internal worker', sub: 'on by default', width: 190 },
        ],
      },
      {
        id: 'pg',
        label: 'PostgreSQL',
        direction: 'column',
        gap: 28,
        children: [
          { id: 'mem', label: 'Memory banks', sub: 'facts · observations · models', shape: 'store', width: 200 },
          { id: 'ops', label: 'async_operations', sub: 'the task queue', shape: 'store', width: 200 },
        ],
      },
      {
        direction: 'column',
        gap: 36,
        children: [
          {
            id: 'workers',
            label: 'hindsight-worker × N',
            direction: 'column',
            gap: 16,
            children: [
              { id: 'w1', label: 'worker-1' },
              { id: 'w2', label: 'worker-2' },
            ],
          },
          {
            id: 'models',
            label: 'Model providers',
            direction: 'column',
            gap: 16,
            children: [
              { id: 'llm', label: 'LLM', sub: 'OpenAI, Anthropic, …' },
              { id: 'emb', label: 'Embeddings · reranker', sub: 'local or remote' },
            ],
          },
        ],
      },
    ],
  },
  edges: [
    { id: 'app-http', from: 'app', to: 'http', label: 'request' },
    { id: 'cp-http', from: 'cp', to: 'http', label: 'SDK' },
    { id: 'http-mem', from: 'http', to: 'mem' },
    { id: 'http-ops', from: 'http', to: 'ops', label: 'enqueue' },
    { id: 'http-emb', from: 'http', to: 'models', quiet: true },
    { id: 'claim-iw', from: 'ops', to: 'iworker', label: 'claim', quiet: true },
    { id: 'claim-w1', from: 'ops', to: 'w1', label: 'claim' },
    { id: 'claim-w2', from: 'ops', to: 'w2' },
    { id: 'w-models', from: 'workers', to: 'models' },
    { id: 'w-mem', from: 'workers', to: 'mem', quiet: true },
    { id: 'iw-mem', from: 'iworker', to: 'mem', quiet: true },
  ],
  steps: [
    {
      label: 'retain · async',
      flow: [
        {
          edges: { edge: 'app-http', data: 'retain(async=true)' },
          show: { app: RETAIN_IN },
          say: 'A retain is slow work: an LLM has to read it. With async=true the API does not do it in the request.',
        },
        {
          edges: { edge: 'http-ops', data: 'op_7f3 · pending' },
          show: { ops: [op('retain', 'op_7f3', 'pending')] },
          say: 'It writes a task row into async_operations — the queue is just a table in the same database.',
        },
        {
          edges: { edge: 'app-http', back: true, data: '{ operation_id }' },
          show: { app: [...RETAIN_IN, { tag: 'accepted', tone: 'green', text: 'op_7f3' }] },
          say: 'And answers right away with the operation id.',
        },
        {
          edges: [{ edge: 'claim-w1', data: 'FOR UPDATE SKIP LOCKED' }],
          show: {
            ops: [op('retain', 'op_7f3', 'processing · worker-1')],
            w1: [{ text: 'retain op_7f3', mono: true, mark: '…' }],
            w2: [{ text: 'row locked, skips it', meta: 'idle' }],
          },
          say: 'Every worker polls the table (every 500 ms). The claim locks the row, so exactly one worker gets it — the others skip it.',
          ms: 2800,
        },
        {
          edges: { edge: 'w-models', data: 'extract · embed' },
          show: { llm: [{ text: 'extract facts' }], emb: [{ text: 'embed 1 fact' }] },
          say: 'The worker calls the LLM to extract facts and embeds them.',
        },
        {
          edges: { edge: 'w-mem', data: '1 fact' },
          show: { mem: [{ tag: 'world', text: 'Alice joined Google', meta: 'Mar 2026', mark: 'new' }] },
          say: 'Then writes them to the bank.',
        },
        {
          edges: { edge: 'claim-w1', back: true, data: 'completed' },
          show: {
            ops: [op('retain', 'op_7f3', 'completed', '✓'), op('consolidation', 'op_8a1', 'pending')],
            w1: [{ text: 'retain op_7f3', mono: true, mark: '✓' }],
          },
          say: 'The task is marked completed. New facts queue their own follow-up: consolidation (when auto-consolidation is on), plus graph and index upkeep.',
          ms: 2800,
        },
        {
          edges: { edge: 'cp-http', data: 'operation status' },
          show: { cp: [{ text: 'op_7f3', meta: 'completed', mark: '✓' }] },
          say: 'The Control Plane talks to the same API, so you can watch the operation there.',
        },
      ],
    },
    {
      label: 'background',
      flow: [
        {
          edges: { edge: 'claim-w2', data: 'consolidation' },
          show: {
            ops: [op('consolidation', 'op_8a1', 'processing · worker-2')],
            w2: [{ text: 'consolidation op_8a1', mono: true, mark: '…' }],
          },
          say: 'Background work goes through the same queue. This time worker-2 wins the claim.',
        },
        {
          edges: { edge: 'w-models', data: 'consolidate' },
          show: { llm: [{ text: 'merge facts into observations' }] },
          say: 'Consolidation asks the LLM to merge new facts into observations.',
        },
        {
          edges: { edge: 'w-mem', data: 'observation' },
          show: { mem: [{ tag: 'observation', tone: 'green', text: 'Alice works at Google', meta: '1 source', mark: 'new' }] },
        },
        {
          edges: { edge: 'claim-w2', back: true, data: 'refresh queued' },
          show: {
            ops: [op('consolidation', 'op_8a1', 'completed', '✓'), op('refresh_mental_model', 'op_9c4', 'pending')],
            w2: [{ text: 'consolidation op_8a1', mono: true, mark: '✓' }],
          },
          say: 'Mental models that refresh after consolidation — and are now stale — get a refresh task of their own.',
          ms: 2600,
        },
        {
          edges: { edge: 'claim-iw', data: 'claim' },
          show: {
            ops: [op('refresh_mental_model', 'op_9c4', 'processing · api')],
            iworker: [{ text: 'refresh_mental_model op_9c4', mono: true, mark: '…' }],
          },
          say: 'The API’s internal worker claims from the same table. Set HINDSIGHT_API_WORKER_ENABLED=false and only dedicated workers do.',
          ms: 3000,
        },
        {
          edges: { edge: 'iw-mem', data: 'rewrite model' },
          show: {
            mem: [
              { tag: 'observation', tone: 'green', text: 'Alice works at Google', meta: '1 source' },
              { tag: 'model', tone: 'orange', text: 'Team overview', mark: '↻' },
            ],
            iworker: [{ text: 'refresh_mental_model op_9c4', mono: true, mark: '✓' }],
          },
          say: 'Any worker can run any task: they share the package, the image and the database, so you add more to scale.',
          ms: 3000,
        },
      ],
    },
    {
      label: 'recall · sync',
      flow: [
        {
          edges: { edge: 'app-http', data: 'recall()' },
          show: { app: RECALL_IN },
          say: 'Later, a recall. Like reflect and any read, it is answered inside the API process. Nothing is queued.',
        },
        {
          edges: { edge: 'http-emb', data: 'embed the query' },
          show: { http: [{ text: 'recall', meta: 'bank alice-team', mono: true }], emb: [{ text: 'query → vector' }] },
          say: 'The query is embedded — by a model loaded in the process itself, or a remote embeddings service.',
        },
        {
          edges: { edge: 'http-mem', data: '4-way search' },
          show: { mem: [{ tag: 'world', text: 'Alice joined Google', meta: 'Mar 2026', mark: '✓' }] },
          say: 'The four searches run as queries against PostgreSQL, where every bank lives.',
        },
        {
          edges: { edge: 'http-emb', data: 'rerank' },
          show: { emb: [{ text: 'query → vector' }, { text: 'cross-encoder rerank' }] },
          say: 'Candidates are reranked by the cross-encoder, local or remote like the embeddings.',
        },
        {
          edges: { edge: 'app-http', back: true, data: 'ranked memories' },
          show: { app: [...RECALL_IN, { tag: 'result', tone: 'green', text: 'Alice joined Google', mark: '✓' }] },
          say: 'The API keeps no state of its own, so you can run as many copies as you like behind a load balancer.',
        },
      ],
    },
  ],
};

export default {
  title: 'Services',
  source: 'hindsight-docs/docs/developer/services.md',
  props,
} satisfies Figure;
