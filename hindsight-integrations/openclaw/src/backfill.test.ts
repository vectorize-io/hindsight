import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, symlinkSync, writeFileSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { pathToFileURL } from 'url';
import type { BankStats, PluginConfig } from './types.js';
import type { BackfillCheckpoint, BackfillPlanEntry } from './backfill-lib.js';

const managerStart = vi.fn();
const managerStop = vi.fn();
const managerGetBaseUrl = vi.fn(() => 'http://127.0.0.1:9077');

vi.mock('./embed-manager.js', () => ({
  HindsightEmbedManager: vi.fn(class {
    start = managerStart;
    stop = managerStop;
    getBaseUrl = managerGetBaseUrl;
  }),
}));

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  managerStart.mockReset();
  managerStop.mockReset();
  managerGetBaseUrl.mockClear();
});

function makeEntry(bankId: string, sessionId: string): BackfillPlanEntry {
  return {
    filePath: `/tmp/${sessionId}.jsonl`,
    agentId: bankId,
    sessionId,
    bankId,
    documentId: `backfill::${bankId}::${sessionId}`,
    transcript: '[role: user]\nhello\n[user:end]',
    messageCount: 1,
  };
}

function makeStats(overrides: Partial<BankStats> = {}): BankStats {
  return {
    bank_id: 'bank',
    total_nodes: 0,
    total_links: 0,
    total_documents: 0,
    pending_operations: 0,
    failed_operations: 0,
    pending_consolidation: 0,
    last_consolidated_at: null,
    total_observations: 0,
    ...overrides,
  };
}

describe('backfill helpers', () => {
  it('resume skips only completed entries', async () => {
    const { filterEntriesForResume, splitResumeEntries } = await import('./backfill.js');
    const entries = [makeEntry('bank-a', '1'), makeEntry('bank-a', '2'), makeEntry('bank-a', '3')];
    const checkpoint: BackfillCheckpoint = {
      version: 1,
      entries: {
        'bank-a::backfill::bank-a::1': { status: 'completed', bankId: 'bank-a', filePath: '/tmp/1', sessionId: '1', updatedAt: 'now' },
        'bank-a::backfill::bank-a::2': { status: 'enqueued', bankId: 'bank-a', filePath: '/tmp/2', sessionId: '2', updatedAt: 'now' },
        'bank-a::backfill::bank-a::3': { status: 'failed', bankId: 'bank-a', filePath: '/tmp/3', sessionId: '3', updatedAt: 'now' },
      },
    };
    const resumable = filterEntriesForResume(entries, checkpoint, true);
    expect(resumable.map((entry) => entry.sessionId)).toEqual(['2', '3']);
    expect(splitResumeEntries(resumable, checkpoint, false).entriesToEnqueue.map((entry) => entry.sessionId)).toEqual(['2', '3']);
    expect(splitResumeEntries(resumable, checkpoint, true)).toEqual({
      entriesToEnqueue: [entries[2]],
      alreadyEnqueuedKeys: ['bank-a::backfill::bank-a::2'],
    });
  });

  it('normalizes legacy queued checkpoint entries', async () => {
    const { loadCheckpoint } = await import('./backfill-lib.js');
    const dir = mkdtempSync(join(tmpdir(), 'hindsight-backfill-'));
    const checkpointPath = join(dir, 'checkpoint.json');
    writeFileSync(checkpointPath, JSON.stringify({
      version: 1,
      entries: {
        legacy: { status: 'queued', bankId: 'bank-a', filePath: '/tmp/a', sessionId: 'a', updatedAt: 'now' },
      },
    }), 'utf8');

    const checkpoint = loadCheckpoint(checkpointPath);
    expect(checkpoint.entries.legacy.status).toBe('enqueued');
  });

  it('marks drained entries completed and leaves aggregate-failure banks enqueued', async () => {
    const { applyDrainResults } = await import('./backfill.js');
    const checkpoint: BackfillCheckpoint = {
      version: 1,
      entries: {
        a: { status: 'enqueued', bankId: 'bank-a', filePath: '/tmp/a', sessionId: 'a', updatedAt: 'now' },
        b: { status: 'enqueued', bankId: 'bank-b', filePath: '/tmp/b', sessionId: 'b', updatedAt: 'now' },
      },
    };
    const touchedEntriesByBank = new Map([
      ['bank-a', ['a']],
      ['bank-b', ['b']],
    ]);
    const finalStatsByBank = new Map<string, BankStats>([
      ['bank-a', makeStats({ bank_id: 'bank-a', pending_operations: 0, failed_operations: 0 })],
      ['bank-b', makeStats({ bank_id: 'bank-b', pending_operations: 0, failed_operations: 2 })],
    ]);
    const initialFailedByBank = new Map([
      ['bank-a', 0],
      ['bank-b', 0],
    ]);

    const result = applyDrainResults(checkpoint, touchedEntriesByBank, finalStatsByBank, initialFailedByBank);
    expect(result.completed).toBe(1);
    expect(result.unresolved).toBe(1);
    expect(result.warnings).toEqual([
      'bank bank-b reported 2 new failed operations during drain; leaving 1 checkpoint entries enqueued',
    ]);
    expect(checkpoint.entries.a.status).toBe('completed');
    expect(checkpoint.entries.b.status).toBe('enqueued');
  });

  it('starts local daemon when no external API is configured and health check fails', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error('offline'));
    vi.stubGlobal('fetch', fetchMock);
    const { createBackfillRuntime } = await import('./backfill.js');
    const pluginConfig: PluginConfig = {
      apiPort: 9077,
      llmProvider: 'openai-codex',
      llmModel: 'gpt-5.4',
    };
    const runtime = await createBackfillRuntime(pluginConfig);
    expect(managerStart).toHaveBeenCalledTimes(1);
    expect(runtime.apiUrl).toBe('http://127.0.0.1:9077');
    await runtime.stop();
    expect(managerStop).toHaveBeenCalledTimes(1);
  });

  it('treats a symlinked bin path as direct execution', async () => {
    const { isDirectExecution } = await import('./backfill.js');
    const dir = mkdtempSync(join(tmpdir(), 'hindsight-backfill-bin-'));
    const modulePath = join(process.cwd(), 'dist', 'backfill.js');
    const symlinkPath = join(dir, 'hindsight-openclaw-backfill');
    symlinkSync(modulePath, symlinkPath);

    const moduleUrl = pathToFileURL(modulePath).href;
    expect(isDirectExecution(symlinkPath, moduleUrl)).toBe(true);
    expect(isDirectExecution(join(dir, 'other-entrypoint'), moduleUrl)).toBe(false);
  });
});
