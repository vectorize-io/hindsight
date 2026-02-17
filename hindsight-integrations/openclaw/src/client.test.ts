import { describe, it, expect } from 'vitest';
import { HindsightClient } from './client.js';

describe('HindsightClient', () => {
  it('should create instance with provider and API key', () => {
    const client = new HindsightClient({ llmProvider: 'openai', llmApiKey: 'test-key', llmModel: 'gpt-4' });
    expect(client).toBeInstanceOf(HindsightClient);
  });

  it('should set bank ID', () => {
    const client = new HindsightClient({ llmProvider: 'openai', llmApiKey: 'test-key' });
    client.setBankId('test-bank');
    // No error thrown means success
    expect(true).toBe(true);
  });

  it('should create instance with embed package path', () => {
    const client = new HindsightClient({ llmProvider: 'openai', llmApiKey: 'test-key', llmModel: 'gpt-4', embedPackagePath: '/path/to/hindsight' });
    expect(client).toBeInstanceOf(HindsightClient);
  });

  it('should create instance in HTTP mode', () => {
    const client = new HindsightClient({
      llmProvider: 'openai',
      llmApiKey: 'test-key',
      apiUrl: 'https://api.example.com/',
      apiToken: 'bearer-token',
    });
    expect(client).toBeInstanceOf(HindsightClient);
  });
});
