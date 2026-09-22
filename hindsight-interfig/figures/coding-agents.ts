import type { Figure, FigRow } from '../src';

// "Coding Agents" for hindsight-docs/docs-integrations/coding-agents.md (generated from
// hindsight-integrations/coding-agents/README.md): how the plugin fills a per-repo bank and feeds it
// back to the agent. One running example: Claude Code in a `payments-api` repo, fixing invoice rounding.
// Boxes process, cylinders store. Grounded in src/core/{session-start,hook,retain-hook,chat,git,missions,
// hindsight}.ts and src/deepen.ts.

const BANK = 'coding-agent::payments-api';
const TOOLS: FigRow[] = [
  { text: 'hindsight_search_knowledge_pages', mono: true },
  { text: 'hindsight_list / read_knowledge_page', mono: true },
  { text: 'hindsight_reflect', mono: true },
  { text: 'hindsight_capture_initiative', mono: true },
  { text: 'hindsight_ingest_document', mono: true },
  { text: 'hindsight_sync_status · _diagnose', mono: true },
];
const PAGES: FigRow[] = [
  { text: 'Component map' },
  { text: 'Core concepts' },
  { text: 'Conventions and patterns' },
  { text: 'Key decisions and rationale' },
  { text: 'Initiatives and enhancements' },
];
const DECISION: FigRow = {
  tag: 'world',
  tone: 'blue',
  text: 'Invoice totals use banker’s rounding (half-even)',
  meta: 'knowledge:decision',
};
const PROMPT: FigRow = { tag: 'prompt', tone: 'gray', text: '“Invoice totals are off by a cent. Fix the rounding.”' };

const props: Figure['props'] = {
  speed: 2200,
  layout: {
    gap: 40,
    children: [
      {
        label: 'Your machine',
        direction: 'column',
        gap: 16,
        children: [
          { id: 'agent', label: 'Claude Code', sub: '~/src/payments-api', width: 220 },
          { id: 'git', label: 'Git history', shape: 'store', width: 220 },
          { id: 'sessions', label: 'Past sessions', sub: 'the agent’s own transcripts', shape: 'store', width: 220 },
        ],
      },
      {
        label: 'Hindsight plugin',
        direction: 'column',
        gap: 14,
        children: [
          { id: 'start', label: 'SessionStart hook' },
          { id: 'deepen', label: 'deepen', sub: 'background backfill' },
          { id: 'prompt', label: 'Prompt hook', sub: 'autoInject' },
          { id: 'tools', label: 'MCP tools', width: 220 },
          { id: 'stop', label: 'Stop hook', sub: 'after every reply' },
        ],
      },
      {
        id: 'bank',
        label: `Bank · ${BANK}`,
        direction: 'column',
        gap: 28,
        children: [
          { id: 'docs', label: 'Documents', shape: 'store', width: 220 },
          {
            id: 'memories',
            label: 'Memories',
            direction: 'column',
            gap: 20,
            children: [
              { id: 'facts', label: 'Facts', shape: 'store', width: 220 },
              { id: 'obs', label: 'Observations', sub: 'one shared scope per repo', shape: 'store', width: 220 },
            ],
          },
          { id: 'pages', label: 'Knowledge pages', shape: 'store', width: 220 },
        ],
      },
      {
        label: 'Hindsight server',
        direction: 'column',
        gap: 60,
        children: [
          { id: 'extract', label: 'Extraction', sub: 'LLM' },
          { id: 'consolidate', label: 'Consolidation' },
          { id: 'refresh', label: 'Page refresh', sub: 'hourly, delta' },
        ],
      },
    ],
  },
  edges: [
    { id: 'on-start', from: 'agent', to: 'start', label: 'session starts' },
    { id: 'on-prompt', from: 'agent', to: 'prompt', label: 'each prompt' },
    { id: 'on-tool', from: 'agent', to: 'tools', label: 'tool call' },
    { id: 'on-stop', from: 'agent', to: 'stop', label: 'reply done' },
    { id: 'spawn', from: 'start', to: 'deepen', label: 'spawns' },
    { id: 'read-git', from: 'git', to: 'deepen' },
    { id: 'read-sessions', from: 'sessions', to: 'deepen' },
    { id: 'backfill', from: 'deepen', to: 'docs', label: 'retain' },
    { id: 'seed-pages', from: 'deepen', to: 'pages', label: 'create pages', quiet: true },
    { id: 'inject', from: 'prompt', to: 'memories', label: 'reflect', quiet: true },
    { id: 'search', from: 'tools', to: 'pages', label: 'search' },
    { id: 'writeback', from: 'stop', to: 'docs', label: 'append' },
    { id: 'to-extract', from: 'docs', to: 'extract' },
    { id: 'to-facts', from: 'extract', to: 'facts' },
    { id: 'to-consolidate', from: 'facts', to: 'consolidate' },
    { id: 'to-obs', from: 'consolidate', to: 'obs' },
    { id: 'to-refresh', from: 'memories', to: 'refresh' },
    { id: 'to-pages', from: 'refresh', to: 'pages' },
  ],
  steps: [
    {
      label: 'first session',
      flow: [
        {
          edges: { edge: 'on-start', data: 'SessionStart' },
          show: { agent: [{ tag: 'shell', tone: 'gray', text: 'cd payments-api && claude', mono: true }] },
          say: 'You open a session in a repo. There is no setup command: the plugin’s SessionStart hook does the work.',
        },
        {
          show: {
            start: [
              { tag: 'bank', tone: 'gray', text: BANK, mono: true },
              { text: 'no git documents yet', mark: 'cold' },
            ],
          },
          say: 'It picks the repo’s bank — one per repository, shared by every agent and every worktree — and finds it empty.',
          ms: 3000,
        },
        {
          edges: { edge: 'spawn', data: 'detached' },
          show: {
            agent: [
              { tag: 'shell', tone: 'gray', text: 'cd payments-api && claude', mono: true },
              { tag: 'banner', tone: 'green', text: 'Hindsight is learning this repo' },
            ],
            start: [
              { tag: 'bank', tone: 'gray', text: BANK, mono: true },
              { text: 'no git documents yet', mark: 'cold' },
              { text: 'cold → codebase survey', meta: 'headless agent' },
            ],
            deepen: [
              { text: '1. configure the bank + create pages' },
              { text: '2. import past sessions' },
              { text: '3. seed the commit history' },
            ],
          },
          say: 'Every session start launches the backfill in the background; it only does what is missing, so the session is never blocked. A cold bank also gets a codebase survey by a headless agent (re-run every 20 commits).',
          ms: 3400,
        },
        {
          edges: { edge: 'seed-pages', data: '5 pages' },
          show: { pages: PAGES.map((p) => ({ ...p, meta: 'empty' })) },
          say: 'Its first job configures the bank and creates the repo’s knowledge pages, each a question about this project. They start empty.',
        },
        {
          edges: [
            { edge: 'read-sessions', data: '14 sessions' },
            { edge: 'read-git', data: 'last 300 commits' },
          ],
          show: {
            sessions: [{ text: '14 earlier sessions in this repo', meta: 'one document each' }],
            git: [
              { tag: 'a1b2c3d', tone: 'gray', text: 'Use banker’s rounding for invoice totals' },
              { tag: '9e8f7a6', tone: 'gray', text: 'Never round line items, only the total' },
            ],
          },
          say: 'It reads the agent’s own past conversations in this repo, and the commit messages of the last 300 commits.',
          ms: 2800,
        },
        {
          edges: { edge: 'backfill', data: 'retain' },
          show: {
            docs: [
              { tag: 'chat:…', tone: 'gray', text: '14 conversations', meta: 'source:chat' },
              { tag: 'gitlog:payments-api', tone: 'gray', text: 'all 300 commit messages, one document', meta: 'source:git' },
            ],
          },
          say: 'Each conversation becomes one document (skipped if already there); the commit history becomes one document, replaced when HEAD moves. Full diffs are opt-in (gitIngest: "full").',
          ms: 3600,
        },
        {
          edges: 'to-extract',
          ms: 1300,
        },
        {
          edges: 'to-facts',
          show: {
            facts: [
              DECISION,
              { tag: 'world', tone: 'blue', text: 'Line items are never rounded, only the total', meta: 'knowledge:convention' },
            ],
          },
          say: 'The server extracts facts and labels the durable ones for the page they belong to: a decision, a convention, a component…',
          ms: 3200,
        },
        {
          edges: 'to-consolidate',
          ms: 1300,
        },
        {
          edges: 'to-obs',
          show: { obs: [{ text: 'Money is rounded once, half-even, on the invoice total', meta: '2 sources' }] },
          say: 'Consolidation merges them into observations — one set per repo, whichever agent wrote the facts.',
        },
        {
          edges: 'to-refresh',
          ms: 1300,
        },
        {
          edges: 'to-pages',
          show: {
            pages: [
              ...PAGES.slice(0, 3),
              { text: 'Key decisions and rationale', meta: '“Rounding: half-even, total only”', mark: '↻' },
              PAGES[4],
            ],
          },
          say: 'Each page is written from the memories labelled for it, and keeps itself current on an hourly schedule (each page on its own minute, only when something changed).',
          ms: 3200,
        },
      ],
    },
    {
      label: 'first prompt',
      flow: [
        {
          edges: { edge: 'on-prompt', data: 'UserPromptSubmit' },
          show: { agent: [PROMPT] },
          say: 'On the first prompt of a session, the prompt hook fetches memory for the task at hand. (A session on an existing bank also got the page roster and a tool guide at start.)',
        },
        {
          edges: { edge: 'inject', data: 'reflect · budget low' },
          show: {
            prompt: [
              { tag: 'autoInject', tone: 'gray', text: 'reflect', meta: 'default', mark: '✓' },
              { text: 'pages · recall · none', meta: 'cheaper options' },
            ],
            obs: [{ text: 'Money is rounded once, half-even, on the invoice total', meta: '2 sources', mark: '✓' }],
            facts: [{ ...DECISION, mark: '✓' }],
          },
          say: 'By default it runs one bounded reflect over the bank. "pages" searches the knowledge pages and "recall" recalls memories instead — both retrieval-only.',
          ms: 3400,
        },
        {
          edges: { edge: 'on-prompt', back: true, data: 'injected context' },
          show: {
            agent: [
              PROMPT,
              {
                tag: 'injected',
                tone: 'green',
                text: 'Round once, half-even, on the invoice total — decided in a1b2c3d. Never round line items.',
              },
              { tag: 'notice', tone: 'gray', text: 'Hindsight · goal: recall this repo’s past decisions about “Invoice totals…”' },
            ],
          },
          say: 'The answer lands in the agent’s context before it starts, once per session. On a bank with no history or pages yet, it waits for the second prompt instead.',
          ms: 3400,
        },
      ],
    },
    {
      label: 'while working',
      flow: [
        {
          edges: { edge: 'on-tool', data: 'search_knowledge_pages("rounding")' },
          show: {
            agent: [PROMPT, { tag: 'tool', tone: 'gray', text: 'hindsight_search_knowledge_pages("rounding")', mono: true }],
            tools: TOOLS,
          },
          say: 'Mid-task the agent pulls memory itself through MCP tools. Pages are not pushed every turn; the page roster and tool guide are re-injected every 10 turns.',
          ms: 3200,
        },
        {
          edges: { edge: 'search', data: 'full text + semantic' },
          show: {
            pages: [
              { text: 'Key decisions and rationale', meta: '§ Money rounding', mark: '✓' },
              { text: 'Conventions and patterns', meta: '§ Money', mark: '✓' },
            ],
          },
          say: 'Page search ranks the pages and returns a snippet of each — fast, and visible as a tool call. hindsight_read_knowledge_page opens one in full.',
          ms: 2800,
        },
        {
          edges: { edge: 'on-tool', back: true, data: '2 pages' },
          show: {
            agent: [
              PROMPT,
              { tag: 'tool', tone: 'gray', text: 'hindsight_search_knowledge_pages("rounding")', mono: true },
              { tag: 'result', tone: 'green', text: 'Key decisions and rationale · Conventions and patterns' },
            ],
          },
          say: 'hindsight_reflect goes deeper when pages are not enough; capture_initiative turns a new plan into its own page.',
          ms: 2800,
        },
      ],
    },
    {
      label: 'each reply',
      flow: [
        {
          edges: { edge: 'on-stop', data: 'Stop' },
          show: {
            agent: [PROMPT, { tag: 'reply', tone: 'gray', text: 'Moved rounding to InvoiceTotal.finalize(); line items stay exact.' }],
            stop: [{ text: '2 new turns since the last write' }, { text: 'injected memory stripped', mark: '✓' }],
          },
          say: 'Every time the agent finishes a reply, the Stop hook reads the transcript and takes the turns it has not written yet.',
          ms: 3000,
        },
        {
          edges: { edge: 'writeback', data: 'append' },
          show: {
            docs: [
              {
                tag: 'conversation:7f3e…',
                tone: 'gray',
                text: 'this session',
                meta: 'source:chat · harness:claude-code',
                mark: '+2 turns',
              },
              { tag: 'gitlog:payments-api', tone: 'gray', text: 'all 300 commit messages, one document', meta: 'source:git' },
            ],
          },
          say: 'They are appended to the session’s document (its first reply created it), tagged with the agent that wrote it — that is where the control plane’s agent logo comes from.',
          ms: 3400,
        },
        {
          edges: 'to-extract',
          ms: 1300,
        },
        {
          edges: 'to-facts',
          show: {
            facts: [
              DECISION,
              {
                tag: 'experience',
                tone: 'purple',
                text: 'I moved rounding into InvoiceTotal.finalize()',
                meta: 'knowledge:decision',
                mark: 'new',
              },
            ],
          },
          say: 'The new turns are extracted like everything else…',
        },
        {
          edges: 'to-consolidate',
          ms: 1300,
        },
        {
          edges: 'to-obs',
          show: { obs: [{ text: 'Money is rounded once, half-even, in InvoiceTotal.finalize()', meta: '3 sources', mark: 'updated' }] },
          say: '…consolidated into the repo’s beliefs…',
        },
        {
          edges: 'to-refresh',
          ms: 1300,
        },
        {
          edges: 'to-pages',
          show: {
            pages: [
              ...PAGES.slice(0, 3),
              { text: 'Key decisions and rationale', meta: '+ rounding lives in finalize()', mark: '↻' },
              PAGES[4],
            ],
          },
          say: '…and the next scheduled refresh edits the page, so the next session starts from it.',
          ms: 3200,
        },
      ],
    },
  ],
};

export default {
  title: 'Coding Agents',
  source: 'hindsight-docs/docs-integrations/coding-agents.md',
  props,
} satisfies Figure;
