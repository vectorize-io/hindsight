import { describe, it, expect } from 'vitest';
import { HindsightServer } from './server.js';

describe('HindsightServer construction', () => {
  it('defaults base URL to http://127.0.0.1:8888', () => {
    const server = new HindsightServer();
    expect(server.getBaseUrl()).toBe('http://127.0.0.1:8888');
    expect(server.getProfile()).toBe('default');
  });

  it('honours custom profile, port, and host', () => {
    const server = new HindsightServer({ profile: 'app', port: 9077, host: '0.0.0.0' });
    expect(server.getProfile()).toBe('app');
    expect(server.getBaseUrl()).toBe('http://0.0.0.0:9077');
  });

  it('accepts open env pass-through without complaining about unknown keys', () => {
    const server = new HindsightServer({
      env: {
        HINDSIGHT_API_LLM_PROVIDER: 'openai',
        HINDSIGHT_API_LLM_MODEL: 'gpt-4o-mini',
        // A field that does not exist today — should still be accepted
        HINDSIGHT_FUTURE_FLAG: 'enabled',
      },
    });
    expect(server).toBeInstanceOf(HindsightServer);
  });

  it('exposes checkHealth that returns false when no daemon is running', async () => {
    // Random high port that nothing is listening on.
    const server = new HindsightServer({ port: 1, readyTimeoutMs: 100 });
    const healthy = await server.checkHealth();
    expect(healthy).toBe(false);
  });
});
