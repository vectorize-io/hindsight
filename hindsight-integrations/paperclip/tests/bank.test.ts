import { describe, it, expect } from 'vitest';
import { deriveBankId } from '../src/bank.js';
import { loadConfig } from '../src/config.js';

describe('deriveBankId', () => {
  const ctx = { companyId: 'co-123', agentId: 'ag-456' };

  const baseUrl = 'http://fake:9077';

  it('default: paperclip::companyId::agentId', () => {
    const config = loadConfig({ hindsightApiUrl: baseUrl });
    expect(deriveBankId(ctx, config)).toBe('paperclip::co-123::ag-456');
  });

  it('company-only granularity', () => {
    const config = loadConfig({ hindsightApiUrl: baseUrl, bankGranularity: ['company'] });
    expect(deriveBankId(ctx, config)).toBe('paperclip::co-123');
  });

  it('agent-only granularity', () => {
    const config = loadConfig({ hindsightApiUrl: baseUrl, bankGranularity: ['agent'] });
    expect(deriveBankId(ctx, config)).toBe('paperclip::ag-456');
  });

  it('custom prefix', () => {
    const config = loadConfig({ hindsightApiUrl: baseUrl, bankIdPrefix: 'myapp' });
    expect(deriveBankId(ctx, config)).toBe('myapp::co-123::ag-456');
  });

  it('empty prefix with default granularity', () => {
    const config = loadConfig({ hindsightApiUrl: baseUrl, bankIdPrefix: '' });
    expect(deriveBankId(ctx, config)).toBe('co-123::ag-456');
  });

  it('throws when bank ID would be empty', () => {
    const config = loadConfig({ hindsightApiUrl: baseUrl, bankIdPrefix: '', bankGranularity: [] });
    expect(() => deriveBankId(ctx, config)).toThrow('Bank ID cannot be empty');
  });

  it('reversed granularity order', () => {
    const config = loadConfig({ hindsightApiUrl: baseUrl, bankGranularity: ['agent', 'company'] });
    expect(deriveBankId(ctx, config)).toBe('paperclip::ag-456::co-123');
  });
});
