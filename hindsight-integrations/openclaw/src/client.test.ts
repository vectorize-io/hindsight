import { afterEach, describe, it, expect, vi } from 'vitest';
import { HindsightClient } from './client.js';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('HindsightClient', () => {
  it('should create instance with model', () => {
    const client = new HindsightClient({ llmModel: 'gpt-4' });
    expect(client).toBeInstanceOf(HindsightClient);
  });

  it('should set bank ID', () => {
    const client = new HindsightClient({});
    expect(() => client.setBankId('test-bank')).not.toThrow();
  });

  it('should create instance with embed package path', () => {
    const client = new HindsightClient({ llmModel: 'gpt-4', embedPackagePath: '/path/to/hindsight' });
    expect(client).toBeInstanceOf(HindsightClient);
  });

  it('should create instance in HTTP mode', () => {
    const client = new HindsightClient({
      apiUrl: 'https://api.example.com/',
      apiToken: 'bearer-token',
    });
    expect(client).toBeInstanceOf(HindsightClient);
  });

  it('should ensure bank mission in HTTP mode', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      text: async () => '',
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = new HindsightClient({
      apiUrl: 'https://api.example.com/',
      apiToken: 'bearer-token',
    });
    client.setBankId('demo');
    await expect(client.ensureBankMission('mission')).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith(
      'https://api.example.com/v1/default/banks/demo',
      expect.objectContaining({ method: 'PUT' }),
    );
  });

  it('should throw when strict bank mission setup fails in HTTP mode', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => 'boom',
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = new HindsightClient({
      apiUrl: 'https://api.example.com/',
    });
    client.setBankId('demo');
    await expect(client.ensureBankMission('mission')).rejects.toThrow('Failed to set bank mission');
  });
});
