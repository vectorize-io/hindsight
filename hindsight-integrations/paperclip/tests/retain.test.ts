import { describe, it, expect, vi, beforeEach } from 'vitest';
import { retain } from '../src/retain.js';
import { loadConfig } from '../src/config.js';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

function makeRetainResponse() {
  return {
    ok: true,
    status: 200,
    json: async () => ({ success: true }),
    text: async () => '',
  } as unknown as Response;
}

function makeErrorResponse(status: number) {
  return {
    ok: false,
    status,
    json: async () => { throw new Error('not json'); },
    text: async () => 'error',
  } as unknown as Response;
}

beforeEach(() => {
  mockFetch.mockReset();
});

const config = loadConfig({ hindsightApiUrl: 'http://fake:9077' });
const input = {
  companyId: 'co-1',
  agentId: 'ag-1',
  content: 'Fixed the authentication bug in login.ts',
  documentId: 'run-abc123',
};

describe('retain()', () => {
  it('does nothing for blank content', async () => {
    await retain({ ...input, content: '   ' }, config);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('calls the correct API path', async () => {
    mockFetch.mockResolvedValue(makeRetainResponse());
    await retain(input, config);
    const [url] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/v1/default/banks/paperclip%3A%3Aco-1%3A%3Aag-1/memories');
  });

  it('sends content in request body items array', async () => {
    mockFetch.mockResolvedValue(makeRetainResponse());
    await retain(input, config);
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.items).toHaveLength(1);
    expect(body.items[0].content).toBe('Fixed the authentication bug in login.ts');
  });

  it('sends document_id to prevent duplicates', async () => {
    mockFetch.mockResolvedValue(makeRetainResponse());
    await retain(input, config);
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.items[0].document_id).toBe('run-abc123');
  });

  it('includes companyId and agentId in metadata', async () => {
    mockFetch.mockResolvedValue(makeRetainResponse());
    await retain(input, config);
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.items[0].metadata.companyId).toBe('co-1');
    expect(body.items[0].metadata.agentId).toBe('ag-1');
  });

  it('merges custom metadata with default metadata', async () => {
    mockFetch.mockResolvedValue(makeRetainResponse());
    await retain({ ...input, metadata: { taskId: 'task-99' } }, config);
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.items[0].metadata.taskId).toBe('task-99');
    expect(body.items[0].metadata.companyId).toBe('co-1');
  });

  it('sets context to retainContext from config', async () => {
    mockFetch.mockResolvedValue(makeRetainResponse());
    await retain(input, config);
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.items[0].context).toBe('paperclip');
  });

  it('gracefully degrades on HTTP error', async () => {
    mockFetch.mockResolvedValue(makeErrorResponse(503));
    // Should not throw
    await expect(retain(input, config)).resolves.toBeUndefined();
  });

  it('gracefully degrades on network error', async () => {
    mockFetch.mockRejectedValue(new Error('Network failure'));
    await expect(retain(input, config)).resolves.toBeUndefined();
  });

  it('sends async flag in request body', async () => {
    mockFetch.mockResolvedValue(makeRetainResponse());
    await retain(input, config);
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.async).toBe(true);
  });
});
