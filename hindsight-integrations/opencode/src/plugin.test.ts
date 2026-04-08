import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock the HindsightClient before importing the plugin
vi.mock('@vectorize-io/hindsight-client', () => {
    const MockHindsightClient = vi.fn(function (this: any) {
        this.retain = vi.fn().mockResolvedValue({});
        this.recall = vi.fn().mockResolvedValue({ results: [] });
        this.reflect = vi.fn().mockResolvedValue({ text: '' });
        this.createBank = vi.fn().mockResolvedValue({});
    });
    return { HindsightClient: MockHindsightClient };
});

import { HindsightPlugin } from './index.js';
import { HindsightClient } from '@vectorize-io/hindsight-client';

const mockPluginInput = {
    client: {
        session: {
            messages: vi.fn().mockResolvedValue({ data: [] }),
        },
    },
    project: { id: 'test-project', worktree: '/tmp/test', vcs: 'git' },
    directory: '/tmp/test-project',
    worktree: '/tmp/test-project',
    serverUrl: new URL('http://localhost:3000'),
    $: {} as any,
};

describe('HindsightPlugin', () => {
    const originalEnv = { ...process.env };

    beforeEach(() => {
        for (const key of Object.keys(process.env)) {
            if (key.startsWith('HINDSIGHT_')) delete process.env[key];
        }
        vi.clearAllMocks();
    });

    afterEach(() => {
        process.env = { ...originalEnv };
    });

    it('returns empty hooks when no API URL configured', async () => {
        const result = await HindsightPlugin(mockPluginInput as any);
        expect(result).toEqual({});
        expect(HindsightClient).not.toHaveBeenCalled();
    });

    it('returns tools and hooks when configured', async () => {
        process.env.HINDSIGHT_API_URL = 'http://localhost:8888';

        const result = await HindsightPlugin(mockPluginInput as any);

        expect(HindsightClient).toHaveBeenCalledWith({
            baseUrl: 'http://localhost:8888',
            apiKey: undefined,
        });

        expect(result.tool).toBeDefined();
        expect(result.tool!.hindsight_retain).toBeDefined();
        expect(result.tool!.hindsight_recall).toBeDefined();
        expect(result.tool!.hindsight_reflect).toBeDefined();
        expect(result.event).toBeDefined();
        expect(result['experimental.session.compacting']).toBeDefined();
        expect(result['experimental.chat.system.transform']).toBeDefined();
    });

    it('passes API key when configured', async () => {
        process.env.HINDSIGHT_API_URL = 'http://localhost:8888';
        process.env.HINDSIGHT_API_TOKEN = 'my-token';

        await HindsightPlugin(mockPluginInput as any);

        expect(HindsightClient).toHaveBeenCalledWith({
            baseUrl: 'http://localhost:8888',
            apiKey: 'my-token',
        });
    });

    it('accepts plugin options', async () => {
        const result = await HindsightPlugin(mockPluginInput as any, {
            hindsightApiUrl: 'http://example.com',
            bankId: 'custom-bank',
        });

        expect(result.tool).toBeDefined();
        expect(HindsightClient).toHaveBeenCalledWith({
            baseUrl: 'http://example.com',
            apiKey: undefined,
        });
    });
});

describe('PluginModule default export', () => {
    it('exports correct module shape', async () => {
        const mod = await import('./index.js');
        expect(mod.default).toBeDefined();
        expect(mod.default.id).toBe('hindsight');
        expect(typeof mod.default.server).toBe('function');
    });
});
