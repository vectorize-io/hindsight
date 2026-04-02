import { afterEach, describe, expect, it } from 'vitest';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import {
  buildBackfillPlan,
  loadPluginConfigFromOpenClawRoot,
  stableDocumentId,
} from './backfill-lib.js';

const tempDirs: string[] = [];

function makeTempRoot(): string {
  const dir = mkdtempSync(join(tmpdir(), 'hindsight-openclaw-backfill-'));
  tempDirs.push(dir);
  return dir;
}

function writeOpenClawConfig(root: string, config: Record<string, unknown>) {
  writeFileSync(join(root, 'openclaw.json'), JSON.stringify(config, null, 2));
}

function writeSession(root: string, agentId: string, fileName: string, lines: unknown[], archive = false) {
  const dir = archive
    ? join(root, 'agents', agentId, 'sessions-archive-from-migration_backup')
    : join(root, 'agents', agentId, 'sessions');
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, fileName), lines.map((line) => JSON.stringify(line)).join('\n') + '\n');
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe('backfill planning', () => {
  it('mirrors plugin bank routing from config by default', () => {
    const root = makeTempRoot();
    writeOpenClawConfig(root, {
      plugins: {
        entries: {
          'hindsight-openclaw': {
            config: {
              dynamicBankId: true,
              dynamicBankGranularity: ['agent', 'provider', 'channel'],
            },
          },
        },
      },
    });
    writeSession(root, 'proj-run', 'one.jsonl', [
      { type: 'session', id: 'session-1', sessionKey: 'agent:proj-run:discord:channel:123' },
      { type: 'message', message: { role: 'user', content: 'hello' } },
      { type: 'message', message: { role: 'assistant', content: 'world' } },
    ]);

    const config = loadPluginConfigFromOpenClawRoot(root);
    const result = buildBackfillPlan(config, {
      openclawRoot: root,
      includeArchive: true,
      bankStrategy: 'mirror-config',
    });

    expect(result.discoveredSessions).toBe(1);
    expect(result.entries).toHaveLength(1);
    expect(result.entries[0].bankId).toBe('proj-run::discord::channel%3A123');
    expect(result.entries[0].documentId).toBe(stableDocumentId({
      filePath: result.entries[0].filePath,
      agentId: 'proj-run',
      sessionId: 'session-1',
      sessionKey: 'agent:proj-run:discord:channel:123',
      messages: [],
    }, result.entries[0].bankId));
  });

  it('supports migration overrides for agent-only banks', () => {
    const root = makeTempRoot();
    writeOpenClawConfig(root, { plugins: { entries: { 'hindsight-openclaw': { config: {} } } } });
    writeSession(root, 'proj-debug', 'two.jsonl', [
      { type: 'session', id: 'session-2', sessionKey: 'agent:proj-debug:discord:group:abc' },
      { type: 'message', message: { role: 'user', content: 'hello' } },
      { type: 'message', message: { role: 'assistant', content: 'world' } },
    ]);

    const config = loadPluginConfigFromOpenClawRoot(root);
    const result = buildBackfillPlan(config, {
      openclawRoot: root,
      includeArchive: true,
      bankStrategy: 'agent',
    });

    expect(result.entries).toHaveLength(1);
    expect(result.entries[0].bankId).toBe('proj-debug');
  });

  it('can exclude archive sessions', () => {
    const root = makeTempRoot();
    writeOpenClawConfig(root, { plugins: { entries: { 'hindsight-openclaw': { config: {} } } } });
    writeSession(root, 'main', 'live.jsonl', [
      { type: 'session', id: 'live' },
      { type: 'message', message: { role: 'user', content: 'live' } },
      { type: 'message', message: { role: 'assistant', content: 'reply' } },
    ]);
    writeSession(root, 'main', 'archive.jsonl', [
      { type: 'session', id: 'archive' },
      { type: 'message', message: { role: 'user', content: 'archived' } },
      { type: 'message', message: { role: 'assistant', content: 'reply' } },
    ], true);

    const config = loadPluginConfigFromOpenClawRoot(root);
    const result = buildBackfillPlan(config, {
      openclawRoot: root,
      includeArchive: false,
      bankStrategy: 'mirror-config',
    });

    expect(result.discoveredSessions).toBe(1);
    expect(result.entries).toHaveLength(1);
    expect(result.entries[0].sessionId).toBe('live');
  });
});
