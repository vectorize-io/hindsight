import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, readFileSync, rmSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { listHindsightPublicArtifacts } from './index.js';
import type { MoltbotConfig } from './types.js';

let tempDir: string | null = null;

afterEach(() => {
  vi.restoreAllMocks();
  if (tempDir) {
    rmSync(tempDir, { recursive: true, force: true });
    tempDir = null;
  }
});

describe('public artifacts', () => {
  it('materializes Hindsight documents as daily-note artifacts', async () => {
    tempDir = mkdtempSync(join(tmpdir(), 'hindsight-artifacts-'));

    const cfg: MoltbotConfig = {
      agents: {
        defaults: { workspace: tempDir },
        list: [{ id: 'main', default: true, workspace: tempDir }],
      },
      plugins: {
        entries: {
          'hindsight-openclaw': {
            config: {
              hindsightApiUrl: 'https://api.example.com',
            },
          },
        },
      },
    };

    const fetchMock = vi.fn(async (input: unknown) => {
      const url = String(input);
      if (url === 'https://api.example.com/v1/default/banks') {
        return {
          ok: true,
          json: async () => ({
            banks: [{ bank_id: 'bank-a', name: 'Bank A', mission: 'Remember things' }],
          }),
        };
      }
      if (url === 'https://api.example.com/v1/default/banks/bank-a/documents?limit=100&offset=0') {
        return {
          ok: true,
          json: async () => ({
            items: [{ id: 'doc/1', bank_id: 'bank-a', updated_at: '2024-01-02T00:00:00Z' }],
          }),
        };
      }
      if (url === 'https://api.example.com/v1/default/banks/bank-a/documents/doc%2F1') {
        return {
          ok: true,
          json: async () => ({
            id: 'doc/1',
            bank_id: 'bank-a',
            original_text: 'hello from stored document',
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-02T00:00:00Z',
            memory_unit_count: 2,
            tags: ['alpha'],
            document_metadata: { source: 'telegram' },
            retain_params: { context: 'chat' },
          }),
        };
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const artifacts = await listHindsightPublicArtifacts(cfg);

    expect(artifacts).toHaveLength(1);
    expect(artifacts[0]?.kind).toBe('daily-note');
    expect(artifacts[0]?.relativePath).toContain('memory/hindsight-bridge/bank-a/');

    const written = readFileSync(artifacts[0]!.absolutePath, 'utf8');
    expect(written).toContain('hello from stored document');
    expect(written).toContain('Remember things');
  });

  it('skips failed per-document fetches and continues export', async () => {
    tempDir = mkdtempSync(join(tmpdir(), 'hindsight-artifacts-'));

    const cfg: MoltbotConfig = {
      agents: {
        defaults: { workspace: tempDir },
        list: [{ id: 'main', default: true, workspace: tempDir }],
      },
      plugins: {
        entries: {
          'hindsight-openclaw': {
            config: {
              hindsightApiUrl: 'https://api.example.com',
            },
          },
        },
      },
    };

    const fetchMock = vi.fn(async (input: unknown) => {
      const url = String(input);
      if (url === 'https://api.example.com/v1/default/banks') {
        return {
          ok: true,
          json: async () => ({
            banks: [{ bank_id: 'bank-b', name: 'Bank B', mission: 'Test isolation' }],
          }),
        };
      }
      if (url === 'https://api.example.com/v1/default/banks/bank-b/documents?limit=100&offset=0') {
        return {
          ok: true,
          json: async () => ({
            items: [
              { id: 'doc/good', bank_id: 'bank-b', updated_at: '2024-01-02T00:00:00Z' },
              { id: 'doc/bad', bank_id: 'bank-b', updated_at: '2024-01-02T00:00:00Z' },
            ],
          }),
        };
      }
      if (url === 'https://api.example.com/v1/default/banks/bank-b/documents/doc%2Fgood') {
        return {
          ok: true,
          json: async () => ({
            id: 'doc/good',
            bank_id: 'bank-b',
            original_text: 'good document content',
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-02T00:00:00Z',
            memory_unit_count: 1,
            tags: [],
            document_metadata: {},
            retain_params: {},
          }),
        };
      }
      if (url === 'https://api.example.com/v1/default/banks/bank-b/documents/doc%2Fbad') {
        return { ok: false, json: async () => ({}) };
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const artifacts = await listHindsightPublicArtifacts(cfg);

    // Only the good document should produce an artifact; the bad one is skipped
    expect(artifacts).toHaveLength(1);
    expect(artifacts[0]?.relativePath).toContain('doc-good');
    const written = readFileSync(artifacts[0]!.absolutePath, 'utf8');
    expect(written).toContain('good document content');
  });

  it('paginates through documents when a page is full', async () => {
    tempDir = mkdtempSync(join(tmpdir(), 'hindsight-artifacts-'));

    const cfg: MoltbotConfig = {
      agents: {
        defaults: { workspace: tempDir },
        list: [{ id: 'main', default: true, workspace: tempDir }],
      },
      plugins: {
        entries: {
          'hindsight-openclaw': {
            config: {
              hindsightApiUrl: 'https://api.example.com',
            },
          },
        },
      },
    };

    // Build a full page of 100 items to trigger a second request
    const fullPage = Array.from({ length: 100 }, (_, i) => ({
      id: `doc/${i}`,
      bank_id: 'bank-p',
      updated_at: '2024-01-02T00:00:00Z',
    }));

    const fetchMock = vi.fn(async (input: unknown) => {
      const url = String(input);
      if (url === 'https://api.example.com/v1/default/banks') {
        return {
          ok: true,
          json: async () => ({
            banks: [{ bank_id: 'bank-p', name: 'Bank P', mission: 'Paginate' }],
          }),
        };
      }
      if (url === 'https://api.example.com/v1/default/banks/bank-p/documents?limit=100&offset=0') {
        return { ok: true, json: async () => ({ items: fullPage }) };
      }
      if (url === 'https://api.example.com/v1/default/banks/bank-p/documents?limit=100&offset=100') {
        return {
          ok: true,
          json: async () => ({
            items: [{ id: 'doc/extra', bank_id: 'bank-p', updated_at: '2024-01-03T00:00:00Z' }],
          }),
        };
      }
      // Per-document fetches — return minimal valid documents
      const docMatch = url.match(/\/documents\/(.+)$/);
      if (docMatch) {
        const docId = decodeURIComponent(docMatch[1]);
        return {
          ok: true,
          json: async () => ({
            id: docId,
            bank_id: 'bank-p',
            original_text: `content of ${docId}`,
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-02T00:00:00Z',
            memory_unit_count: 1,
            tags: [],
            document_metadata: {},
            retain_params: {},
          }),
        };
      }
      throw new Error(`Unexpected fetch URL: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    const artifacts = await listHindsightPublicArtifacts(cfg);

    // 100 from first page + 1 from second page
    expect(artifacts).toHaveLength(101);

    // Verify the second page was actually requested
    const calls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(calls).toContain(
      'https://api.example.com/v1/default/banks/bank-p/documents?limit=100&offset=100',
    );
  });
});
