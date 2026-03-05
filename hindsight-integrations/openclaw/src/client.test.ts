import { describe, it, expect } from 'vitest';
import { HindsightClient } from './client.js';

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
});
