/**
 * Integration tests for the OpenClaw plugin hooks.
 *
 * Loads the plugin with a mock MoltbotPluginAPI in HTTP mode, then triggers
 * `before_agent_start` and `agent_end` hooks with realistic event payloads.
 * Client methods (recall / retain) are spied on to verify the plugin
 * orchestrates them correctly without requiring a full LLM pipeline.
 *
 * Requirements:
 *   Running Hindsight API at HINDSIGHT_API_URL (default: http://localhost:8888)
 *
 * Run:
 *   npm run test:integration
 */

import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from 'vitest';
import type { HindsightClient } from '../src/client.js';
import type { MoltbotPluginAPI, PluginConfig } from '../src/types.js';
import type { RecallResponse, RetainResponse } from '../src/types.js';

const HINDSIGHT_API_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function waitForApi(url: string, maxMs = 5000): Promise<boolean> {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(1000) });
      if (res.ok) return true;
    } catch {
      /* not ready yet */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

interface MockApiHandle {
  api: MoltbotPluginAPI;
  /** Trigger a registered hook and return the last handler's return value. */
  trigger(event: string, eventData: unknown, ctx?: unknown): Promise<unknown>;
  startServices(): Promise<void>;
  stopServices(): Promise<void>;
}

function createMockApi(pluginConfig: Partial<PluginConfig> = {}): MockApiHandle {
  const handlers = new Map<string, ((event: unknown, ctx?: unknown) => unknown)[]>();
  const services: { id: string; start(): Promise<void>; stop(): Promise<void> }[] = [];

  const api: MoltbotPluginAPI = {
    config: {
      plugins: {
        entries: {
          'hindsight-openclaw': { enabled: true, config: pluginConfig as PluginConfig },
        },
      },
    },
    registerService(svc: any) {
      services.push(svc);
    },
    on(event: string, handler: any) {
      const list = handlers.get(event) ?? [];
      list.push(handler);
      handlers.set(event, list);
    },
  };

  return {
    api,
    async trigger(event, eventData, ctx) {
      const list = handlers.get(event) ?? [];
      let result: unknown;
      for (const h of list) result = await h(eventData, ctx);
      return result;
    },
    async startServices() {
      for (const svc of services) await svc.start();
    },
    async stopServices() {
      for (const svc of services) await svc.stop();
    },
  };
}

const EMPTY_RECALL: RecallResponse = { results: [], entities: null, trace: null, chunks: null };
const OK_RETAIN: RetainResponse = { message: 'queued', document_id: 'test', memory_unit_ids: [] };

function makeMemoryResult(text: string) {
  return {
    id: `mem-${Math.random().toString(36).slice(2)}`,
    text,
    type: 'fact',
    entities: [],
    context: '',
    occurred_start: null,
    occurred_end: null,
    mentioned_at: null,
    document_id: null,
    metadata: null,
    chunk_id: null,
    tags: [],
  };
}

// ---------------------------------------------------------------------------
// Module-level state shared across all hook describe blocks
// ---------------------------------------------------------------------------

let apiReachable = false;
let triggerHook: MockApiHandle['trigger'];
let stopServicesFn: () => Promise<void>;
let recallSpy: ReturnType<typeof vi.spyOn<HindsightClient, 'recall'>>;
let retainSpy: ReturnType<typeof vi.spyOn<HindsightClient, 'retain'>>;

beforeAll(async () => {
  apiReachable = await waitForApi(HINDSIGHT_API_URL, 8000);
  if (!apiReachable) {
    console.warn(
      `[Hooks Integration] Hindsight API not reachable at ${HINDSIGHT_API_URL} – skipping hook tests.`,
    );
    return;
  }

  // Reset module registry so we get a fresh module with clean state.
  vi.resetModules();

  // Provide LLM config — used by plugin init even in HTTP mode.
  process.env.HINDSIGHT_API_LLM_PROVIDER = 'openai';
  process.env.HINDSIGHT_API_LLM_API_KEY = 'test-key-hooks';
  // Point the plugin at the running test API.
  process.env.HINDSIGHT_EMBED_API_URL = HINDSIGHT_API_URL;

  const mod = await import('../src/index.js');
  const pluginFn = mod.default;
  const getClient = mod.getClient;

  const handle = createMockApi({
    dynamicBankId: true,
    excludeProviders: ['slack'],
    // No bankMission — keeps init lean
  });
  triggerHook = handle.trigger;
  stopServicesFn = handle.stopServices;

  // Load the plugin — registers hooks and starts background init.
  pluginFn(handle.api);

  // service.start() awaits initPromise and health-checks the external API.
  await handle.startServices();

  // After startServices the client must be ready.
  const c = getClient();
  if (!c) throw new Error('[Hooks Integration] Client not initialized after service start');

  recallSpy = vi.spyOn(c, 'recall') as ReturnType<typeof vi.spyOn<HindsightClient, 'recall'>>;
  retainSpy = vi.spyOn(c, 'retain') as ReturnType<typeof vi.spyOn<HindsightClient, 'retain'>>;
}, 30_000);

afterAll(async () => {
  vi.restoreAllMocks();
  delete process.env.HINDSIGHT_API_LLM_PROVIDER;
  delete process.env.HINDSIGHT_API_LLM_API_KEY;
  delete process.env.HINDSIGHT_EMBED_API_URL;
  if (stopServicesFn) await stopServicesFn().catch(() => {});
}, 15_000);

afterEach(() => {
  // Reset spy call history between tests; don't remove the implementation.
  recallSpy?.mockReset();
  retainSpy?.mockReset();
});

// ---------------------------------------------------------------------------
// before_agent_start
// ---------------------------------------------------------------------------

describe('before_agent_start hook', () => {
  it('skips recall for excluded providers and returns undefined', async () => {
    if (!apiReachable) return;

    const result = await triggerHook(
      'before_agent_start',
      { rawMessage: 'What are my preferences?', prompt: 'What are my preferences?' },
      { messageProvider: 'slack', senderId: 'U001' },
    );

    expect(recallSpy).not.toHaveBeenCalled();
    expect(result).toBeUndefined();
  });

  it('skips recall when rawMessage is too short and returns undefined', async () => {
    if (!apiReachable) return;

    const result = await triggerHook(
      'before_agent_start',
      { rawMessage: 'Hi', prompt: 'Hi' },
      { messageProvider: 'telegram', senderId: 'U001' },
    );

    expect(recallSpy).not.toHaveBeenCalled();
    expect(result).toBeUndefined();
  });

  it('returns undefined when recall finds no results', async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue(EMPTY_RECALL);

    const result = await triggerHook(
      'before_agent_start',
      { rawMessage: 'What programming language do I like?', prompt: '' },
      { messageProvider: 'telegram', senderId: 'U002' },
    );

    expect(recallSpy).toHaveBeenCalledOnce();
    expect(result).toBeUndefined();
  });

  it('returns { prependContext } with <hindsight_memories> when recall returns results', async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue({
      results: [makeMemoryResult('User likes Python')],
      entities: null,
      trace: null,
      chunks: null,
    });

    const result = (await triggerHook(
      'before_agent_start',
      { rawMessage: 'What programming language do I prefer?', prompt: '' },
      { messageProvider: 'telegram', senderId: 'U003' },
    )) as { prependContext: string };

    expect(result).toBeDefined();
    expect(result.prependContext).toContain('<hindsight_memories>');
    expect(result.prependContext).toContain('User likes Python');
    expect(result.prependContext).toContain('</hindsight_memories>');
  });

  it('injects all memory result fields in the prependContext JSON', async () => {
    if (!apiReachable) return;
    const mem = makeMemoryResult('User prefers dark mode');
    mem.tags = ['preference'];
    mem.entities = ['dark_mode'];
    recallSpy.mockResolvedValue({
      results: [mem],
      entities: null,
      trace: null,
      chunks: null,
    });

    const result = (await triggerHook(
      'before_agent_start',
      { rawMessage: 'Do I prefer dark or light mode?', prompt: '' },
      { messageProvider: 'telegram', senderId: 'U004' },
    )) as { prependContext: string };

    // The prependContext should be valid JSON containing all MemoryResult fields
    const jsonStart = result.prependContext.indexOf('[');
    const jsonEnd = result.prependContext.lastIndexOf(']') + 1;
    const parsed = JSON.parse(result.prependContext.slice(jsonStart, jsonEnd)) as unknown[];
    expect(parsed).toHaveLength(1);
    const first = parsed[0] as Record<string, unknown>;
    expect(first.id).toBe(mem.id);
    expect(first.text).toBe('User prefers dark mode');
    expect(first.type).toBe('fact');
    expect(first.tags).toEqual(['preference']);
    expect(first.entities).toEqual(['dark_mode']);
  });

  it('extracts the inner query from an envelope-formatted prompt when rawMessage is absent', async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue(EMPTY_RECALL);

    const envelopePrompt = '[Telegram Chat]\nWhat is my favorite food?\n[from: Alice]';
    await triggerHook(
      'before_agent_start',
      { rawMessage: '', prompt: envelopePrompt },
      { messageProvider: 'telegram', senderId: 'U005' },
    );

    expect(recallSpy).toHaveBeenCalledOnce();
    const [callArgs] = recallSpy.mock.calls[0];
    // The query passed to recall must NOT contain envelope artifacts
    expect(callArgs.query).not.toContain('[Telegram');
    expect(callArgs.query).not.toContain('[from: Alice]');
    expect(callArgs.query).toContain('What is my favorite food?');
  });

  it('passes max_tokens to recall', async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue(EMPTY_RECALL);

    await triggerHook(
      'before_agent_start',
      { rawMessage: 'Tell me about my hobbies please.', prompt: '' },
      { messageProvider: 'telegram', senderId: 'U006' },
    );

    expect(recallSpy).toHaveBeenCalledOnce();
    const [callArgs] = recallSpy.mock.calls[0];
    expect(callArgs.max_tokens).toBeGreaterThan(0);
  });

  it('includes the user message in the prependContext block', async () => {
    if (!apiReachable) return;
    recallSpy.mockResolvedValue({
      results: [makeMemoryResult('User loves hiking')],
      entities: null,
      trace: null,
      chunks: null,
    });

    const result = (await triggerHook(
      'before_agent_start',
      { rawMessage: 'What outdoor activities do I enjoy?', prompt: '' },
      { messageProvider: 'telegram', senderId: 'U007' },
    )) as { prependContext: string };

    expect(result.prependContext).toContain('What outdoor activities do I enjoy?');
  });
});

// ---------------------------------------------------------------------------
// agent_end hook
// ---------------------------------------------------------------------------

describe('agent_end hook', () => {
  it('skips retain when success is false', async () => {
    if (!apiReachable) return;

    await triggerHook(
      'agent_end',
      { success: false, messages: [{ role: 'user', content: 'Hello there world!' }] },
      { messageProvider: 'telegram', senderId: 'U010' },
    );

    expect(retainSpy).not.toHaveBeenCalled();
  });

  it('skips retain when messages array is empty', async () => {
    if (!apiReachable) return;

    await triggerHook(
      'agent_end',
      { success: true, messages: [] },
      { messageProvider: 'telegram', senderId: 'U011' },
    );

    expect(retainSpy).not.toHaveBeenCalled();
  });

  it('skips retain for excluded providers', async () => {
    if (!apiReachable) return;

    await triggerHook(
      'agent_end',
      {
        success: true,
        messages: [{ role: 'user', content: 'I work as a software engineer.' }],
      },
      { messageProvider: 'slack', senderId: 'U012' },
    );

    expect(retainSpy).not.toHaveBeenCalled();
  });

  it('calls retain with correctly formatted transcript for string content', async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      'agent_end',
      {
        success: true,
        messages: [
          { role: 'user', content: 'I love TypeScript.' },
          { role: 'assistant', content: 'TypeScript is great!' },
        ],
      },
      { messageProvider: 'telegram', senderId: 'U013', sessionKey: 'sess-ts-test' },
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [req] = retainSpy.mock.calls[0];
    expect(req.content).toContain('[role: user]');
    expect(req.content).toContain('I love TypeScript.');
    expect(req.content).toContain('[user:end]');
    expect(req.content).toContain('[role: assistant]');
    expect(req.content).toContain('TypeScript is great!');
    expect(req.content).toContain('[assistant:end]');
  });

  it('includes session key in document_id', async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      'agent_end',
      {
        success: true,
        messages: [{ role: 'user', content: 'My favourite colour is blue.' }],
      },
      { messageProvider: 'telegram', senderId: 'U014', sessionKey: 'sess-colour' },
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [req] = retainSpy.mock.calls[0];
    expect(req.document_id).toContain('sess-colour');
  });

  it('populates metadata with channel_type, channel_id, and sender_id', async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      'agent_end',
      {
        success: true,
        messages: [{ role: 'user', content: 'My cat is named Whiskers.' }],
      },
      {
        messageProvider: 'telegram',
        channelId: 'chat-999',
        senderId: 'U015',
        sessionKey: 'sess-cat',
      },
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [req] = retainSpy.mock.calls[0];
    expect(req.metadata?.channel_type).toBe('telegram');
    expect(req.metadata?.channel_id).toBe('chat-999');
    expect(req.metadata?.sender_id).toBe('U015');
    expect(req.metadata?.retained_at).toBeDefined();
    expect(req.metadata?.message_count).toBe('1');
  });

  it('strips <hindsight_memories> tags from content before retaining', async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    const contentWithMemories =
      '<hindsight_memories>\nRelevant memories:\n[{"text":"old fact"}]\n</hindsight_memories>\nI enjoy reading science fiction.';

    await triggerHook(
      'agent_end',
      {
        success: true,
        messages: [{ role: 'user', content: contentWithMemories }],
      },
      { messageProvider: 'telegram', senderId: 'U016', sessionKey: 'sess-strip' },
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [req] = retainSpy.mock.calls[0];
    expect(req.content).not.toContain('<hindsight_memories>');
    expect(req.content).not.toContain('</hindsight_memories>');
    expect(req.content).not.toContain('old fact');
    expect(req.content).toContain('I enjoy reading science fiction.');
  });

  it('strips <relevant_memories> tags from content before retaining', async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    const contentWithLegacyTag =
      '<relevant_memories>\nSome old memories\n</relevant_memories>\nI am learning Rust.';

    await triggerHook(
      'agent_end',
      {
        success: true,
        messages: [{ role: 'user', content: contentWithLegacyTag }],
      },
      { messageProvider: 'telegram', senderId: 'U017', sessionKey: 'sess-legacy' },
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [req] = retainSpy.mock.calls[0];
    expect(req.content).not.toContain('<relevant_memories>');
    expect(req.content).toContain('I am learning Rust.');
  });

  it('handles array content blocks (structured message format)', async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      'agent_end',
      {
        success: true,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'I prefer dark mode in all my editors.' },
              { type: 'image', source: 'data:...' }, // non-text block — should be ignored
            ],
          },
        ],
      },
      { messageProvider: 'telegram', senderId: 'U018', sessionKey: 'sess-array' },
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [req] = retainSpy.mock.calls[0];
    expect(req.content).toContain('I prefer dark mode in all my editors.');
    // Image block text should not appear
    expect(req.content).not.toContain('data:');
  });

  it('retains a multi-turn conversation in the correct transcript format', async () => {
    if (!apiReachable) return;
    retainSpy.mockResolvedValue(OK_RETAIN);

    await triggerHook(
      'agent_end',
      {
        success: true,
        messages: [
          { role: 'user', content: 'My name is Carol.' },
          { role: 'assistant', content: 'Nice to meet you, Carol!' },
          { role: 'user', content: 'I work as a data scientist.' },
          { role: 'assistant', content: "That's a fascinating career!" },
        ],
      },
      { messageProvider: 'telegram', senderId: 'U019', sessionKey: 'sess-multi' },
    );

    expect(retainSpy).toHaveBeenCalledOnce();
    const [req] = retainSpy.mock.calls[0];

    // Each message should appear in the correct envelope format
    expect(req.content).toContain('[role: user]\nMy name is Carol.\n[user:end]');
    expect(req.content).toContain('[role: assistant]\nNice to meet you, Carol!\n[assistant:end]');
    expect(req.content).toContain('[role: user]\nI work as a data scientist.\n[user:end]');
    expect(req.metadata?.message_count).toBe('4');
  });
});
