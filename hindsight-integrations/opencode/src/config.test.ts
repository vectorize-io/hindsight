import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { loadConfig, type HindsightConfig } from './config.js';

describe('loadConfig', () => {
    const originalEnv = { ...process.env };

    beforeEach(() => {
        // Clear all HINDSIGHT_ env vars
        for (const key of Object.keys(process.env)) {
            if (key.startsWith('HINDSIGHT_')) {
                delete process.env[key];
            }
        }
    });

    afterEach(() => {
        process.env = { ...originalEnv };
    });

    it('returns defaults when no config sources exist', () => {
        const config = loadConfig();
        expect(config.autoRecall).toBe(true);
        expect(config.autoRetain).toBe(true);
        expect(config.recallBudget).toBe('mid');
        expect(config.recallMaxTokens).toBe(1024);
        expect(config.retainContext).toBe('opencode');
        expect(config.agentName).toBe('opencode');
        expect(config.dynamicBankId).toBe(false);
        expect(config.debug).toBe(false);
        expect(config.hindsightApiUrl).toBeNull();
        expect(config.hindsightApiToken).toBeNull();
        expect(config.bankId).toBeNull();
    });

    it('env vars override defaults', () => {
        process.env.HINDSIGHT_API_URL = 'https://example.com';
        process.env.HINDSIGHT_API_TOKEN = 'secret-token';
        process.env.HINDSIGHT_BANK_ID = 'my-bank';
        process.env.HINDSIGHT_AUTO_RECALL = 'false';
        process.env.HINDSIGHT_AUTO_RETAIN = '0';
        process.env.HINDSIGHT_RECALL_MAX_TOKENS = '2048';
        process.env.HINDSIGHT_DEBUG = 'true';

        const config = loadConfig();
        expect(config.hindsightApiUrl).toBe('https://example.com');
        expect(config.hindsightApiToken).toBe('secret-token');
        expect(config.bankId).toBe('my-bank');
        expect(config.autoRecall).toBe(false);
        expect(config.autoRetain).toBe(false);
        expect(config.recallMaxTokens).toBe(2048);
        expect(config.debug).toBe(true);
    });

    it('plugin options override defaults', () => {
        const config = loadConfig({
            bankId: 'plugin-bank',
            autoRecall: false,
            recallBudget: 'high',
        });
        expect(config.bankId).toBe('plugin-bank');
        expect(config.autoRecall).toBe(false);
        expect(config.recallBudget).toBe('high');
    });

    it('env vars override plugin options', () => {
        process.env.HINDSIGHT_BANK_ID = 'env-bank';
        const config = loadConfig({ bankId: 'plugin-bank' });
        expect(config.bankId).toBe('env-bank');
    });

    it('boolean env var parsing', () => {
        process.env.HINDSIGHT_AUTO_RECALL = 'true';
        expect(loadConfig().autoRecall).toBe(true);

        process.env.HINDSIGHT_AUTO_RECALL = '1';
        expect(loadConfig().autoRecall).toBe(true);

        process.env.HINDSIGHT_AUTO_RECALL = 'yes';
        expect(loadConfig().autoRecall).toBe(true);

        process.env.HINDSIGHT_AUTO_RECALL = 'false';
        expect(loadConfig().autoRecall).toBe(false);

        process.env.HINDSIGHT_AUTO_RECALL = 'no';
        expect(loadConfig().autoRecall).toBe(false);
    });

    it('integer env var parsing', () => {
        process.env.HINDSIGHT_RECALL_MAX_TOKENS = '4096';
        expect(loadConfig().recallMaxTokens).toBe(4096);

        // Invalid integer keeps default
        process.env.HINDSIGHT_RECALL_MAX_TOKENS = 'not-a-number';
        expect(loadConfig().recallMaxTokens).toBe(1024);
    });

    it('null plugin options are ignored', () => {
        const config = loadConfig({ bankId: null, debug: undefined });
        expect(config.bankId).toBeNull(); // stays default null
        expect(config.debug).toBe(false); // stays default
    });

    it('invalid retainMode falls back to full-session with warning', () => {
        const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
        const config = loadConfig({ retainMode: 'full_session' });
        expect(config.retainMode).toBe('full-session');
        expect(spy).toHaveBeenCalledWith(expect.stringContaining('Unknown retainMode'));
        spy.mockRestore();
    });

    it('invalid recallBudget falls back to mid with warning', () => {
        const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
        const config = loadConfig({ recallBudget: 'maximum' });
        expect(config.recallBudget).toBe('mid');
        expect(spy).toHaveBeenCalledWith(expect.stringContaining('Unknown recallBudget'));
        spy.mockRestore();
    });

    it('valid retainMode and recallBudget pass without warning', () => {
        const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
        const config = loadConfig({ retainMode: 'last-turn', recallBudget: 'high' });
        expect(config.retainMode).toBe('last-turn');
        expect(config.recallBudget).toBe('high');
        expect(spy).not.toHaveBeenCalled();
        spy.mockRestore();
    });
});
