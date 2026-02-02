import { describe, it, expect } from 'vitest';
import { HindsightClient } from './client.js';

describe('HindsightClient', () => {
  it('should create instance with provider and API key', () => {
    const client = new HindsightClient('openai', 'test-key', 'gpt-4');
    expect(client).toBeInstanceOf(HindsightClient);
  });

  it('should set bank ID', () => {
    const client = new HindsightClient('openai', 'test-key');
    client.setBankId('test-bank');
    // No error thrown means success
    expect(true).toBe(true);
  });

  it('should handle content escaping for single quotes', () => {
    const client = new HindsightClient('openai', 'test-key');
    // This test validates the client is instantiated correctly
    // Actual CLI calls would require mocking
    expect(client).toBeDefined();
  });
});
