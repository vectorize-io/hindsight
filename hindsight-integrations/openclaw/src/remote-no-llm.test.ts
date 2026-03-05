import { describe, it, expect } from 'vitest';
import { HindsightClient } from './client.js';

describe('HindsightClient remote mode without LLM config', () => {
  it('should initialize successfully with only apiUrl and apiToken', () => {
    const client = new HindsightClient({
      apiUrl: 'https://api.example.com',
      apiToken: 'secret-token',
    });

    expect(client).toBeDefined();
    expect(client).toBeInstanceOf(HindsightClient);
  });

  it('should allow initialization with partial config', () => {
    const client = new HindsightClient({
      apiUrl: 'https://api.example.com',
      llmModel: 'gpt-4o-mini',
    });
    expect(client).toBeDefined();
  });
});
