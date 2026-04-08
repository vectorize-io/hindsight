import { describe, it, expect, vi, beforeEach } from 'vitest';
import { recall } from '../src/recall.js';
import { loadConfig } from '../src/config.js';

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

function makeRecallResponse(results: Array<{ text: string; type?: string; mentionedAt?: string }>) {
  return {
    ok: true,
    status: 200,
    json: async () => ({ results }),
    text: async () => '',
  } as unknown as Response;
}

function makeErrorResponse(status: number, body = '') {
  return {
    ok: false,
    status,
    json: async () => { throw new Error('not json'); },
    text: async () => body,
  } as unknown as Response;
}

beforeEach(() => {
  mockFetch.mockReset();
});

const config = loadConfig({ hindsightApiUrl: 'http://fake:9077' });
const input = { companyId: 'co-1', agentId: 'ag-1', query: 'what did I work on?' };

describe('recall()', () => {
  it('returns empty string for blank query', async () => {
    const result = await recall({ ...input, query: '   ' }, config);
    expect(result).toBe('');
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('formats memories as bullet list', async () => {
    mockFetch.mockResolvedValue(makeRecallResponse([
      { text: 'Fixed the login bug', type: 'experience' },
      { text: 'Prefers TypeScript', type: 'preference' },
    ]));
    const result = await recall(input, config);
    expect(result).toContain('- Fixed the login bug [experience]');
    expect(result).toContain('- Prefers TypeScript [preference]');
  });

  it('includes mentionedAt date when present', async () => {
    mockFetch.mockResolvedValue(makeRecallResponse([
      { text: 'Deployed to prod', mentionedAt: '2024-01-15' },
    ]));
    const result = await recall(input, config);
    expect(result).toContain('(2024-01-15)');
  });

  it('returns empty string when no results', async () => {
    mockFetch.mockResolvedValue(makeRecallResponse([]));
    const result = await recall(input, config);
    expect(result).toBe('');
  });

  it('gracefully degrades on HTTP error', async () => {
    mockFetch.mockResolvedValue(makeErrorResponse(500, 'Internal Server Error'));
    const result = await recall(input, config);
    expect(result).toBe('');
  });

  it('gracefully degrades on network error', async () => {
    mockFetch.mockRejectedValue(new Error('ECONNREFUSED'));
    const result = await recall(input, config);
    expect(result).toBe('');
  });

  it('calls the correct API path with bank ID', async () => {
    mockFetch.mockResolvedValue(makeRecallResponse([]));
    await recall(input, config);
    const [url] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/v1/default/banks/paperclip%3A%3Aco-1%3A%3Aag-1/memories/recall');
  });

  it('sends query and budget in request body', async () => {
    mockFetch.mockResolvedValue(makeRecallResponse([]));
    await recall(input, config);
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.query).toBe('what did I work on?');
    expect(body.budget).toBe('mid');
    expect(body.max_tokens).toBe(1024);
  });

  it('uses custom budget and max_tokens from config', async () => {
    const customConfig = loadConfig({
      hindsightApiUrl: 'http://fake:9077',
      recallBudget: 'high',
      recallMaxTokens: 2048,
    });
    mockFetch.mockResolvedValue(makeRecallResponse([]));
    await recall(input, customConfig);
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.budget).toBe('high');
    expect(body.max_tokens).toBe(2048);
  });

  it('sends Authorization header when token is set', async () => {
    const authConfig = loadConfig({
      hindsightApiUrl: 'http://fake:9077',
      hindsightApiToken: 'hsk_test123',
    });
    mockFetch.mockResolvedValue(makeRecallResponse([]));
    await recall(input, authConfig);
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>)['Authorization']).toBe('Bearer hsk_test123');
  });
});
