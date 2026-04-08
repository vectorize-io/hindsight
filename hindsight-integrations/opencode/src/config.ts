/**
 * Configuration management for the Hindsight OpenCode plugin.
 *
 * Loading order (later entries win):
 *   1. Built-in defaults
 *   2. User config file (~/.hindsight/opencode.json)
 *   3. Plugin options (from opencode.json plugin tuple)
 *   4. Environment variable overrides
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

export interface HindsightConfig {
    // Recall
    autoRecall: boolean;
    recallBudget: string;
    recallMaxTokens: number;
    recallTypes: string[];
    recallContextTurns: number;
    recallMaxQueryChars: number;
    recallPromptPreamble: string;

    // Retain
    autoRetain: boolean;
    retainMode: string;
    retainEveryNTurns: number;
    retainOverlapTurns: number;
    retainContext: string;
    retainTags: string[];
    retainMetadata: Record<string, string>;

    // Connection
    hindsightApiUrl: string | null;
    hindsightApiToken: string | null;

    // Bank
    bankId: string | null;
    bankIdPrefix: string;
    dynamicBankId: boolean;
    dynamicBankGranularity: string[];
    bankMission: string;
    retainMission: string | null;
    agentName: string;

    // Misc
    debug: boolean;
}

const DEFAULTS: HindsightConfig = {
    // Recall
    autoRecall: true,
    recallBudget: 'mid',
    recallMaxTokens: 1024,
    recallTypes: ['world', 'experience'],
    recallContextTurns: 1,
    recallMaxQueryChars: 800,
    recallPromptPreamble:
        'Relevant memories from past conversations (prioritize recent when ' +
        'conflicting). Only use memories that are directly useful to continue ' +
        'this conversation; ignore the rest:',

    // Retain
    autoRetain: true,
    retainMode: 'full-session',
    retainEveryNTurns: 10,
    retainOverlapTurns: 2,
    retainContext: 'opencode',
    retainTags: [],
    retainMetadata: {},

    // Connection
    hindsightApiUrl: null,
    hindsightApiToken: null,

    // Bank
    bankId: null,
    bankIdPrefix: '',
    dynamicBankId: false,
    dynamicBankGranularity: ['agent', 'project'],
    bankMission: '',
    retainMission: null,
    agentName: 'opencode',

    // Misc
    debug: false,
};

/** Env var → config key + type mapping */
const ENV_OVERRIDES: Record<string, [keyof HindsightConfig, 'string' | 'bool' | 'int']> = {
    HINDSIGHT_API_URL: ['hindsightApiUrl', 'string'],
    HINDSIGHT_API_TOKEN: ['hindsightApiToken', 'string'],
    HINDSIGHT_BANK_ID: ['bankId', 'string'],
    HINDSIGHT_AGENT_NAME: ['agentName', 'string'],
    HINDSIGHT_AUTO_RECALL: ['autoRecall', 'bool'],
    HINDSIGHT_AUTO_RETAIN: ['autoRetain', 'bool'],
    HINDSIGHT_RETAIN_MODE: ['retainMode', 'string'],
    HINDSIGHT_RECALL_BUDGET: ['recallBudget', 'string'],
    HINDSIGHT_RECALL_MAX_TOKENS: ['recallMaxTokens', 'int'],
    HINDSIGHT_RECALL_MAX_QUERY_CHARS: ['recallMaxQueryChars', 'int'],
    HINDSIGHT_RECALL_CONTEXT_TURNS: ['recallContextTurns', 'int'],
    HINDSIGHT_DYNAMIC_BANK_ID: ['dynamicBankId', 'bool'],
    HINDSIGHT_BANK_MISSION: ['bankMission', 'string'],
    HINDSIGHT_DEBUG: ['debug', 'bool'],
};

function castEnv(value: string, typ: 'string' | 'bool' | 'int'): string | boolean | number | null {
    if (typ === 'bool') return ['true', '1', 'yes'].includes(value.toLowerCase());
    if (typ === 'int') {
        const n = parseInt(value, 10);
        return isNaN(n) ? null : n;
    }
    return value;
}

function loadSettingsFile(path: string): Record<string, unknown> {
    try {
        const raw = readFileSync(path, 'utf-8');
        return JSON.parse(raw);
    } catch {
        return {};
    }
}

export function loadConfig(pluginOptions?: Record<string, unknown>): HindsightConfig {
    // 1. Start with defaults
    const config: Record<string, unknown> = { ...DEFAULTS };

    // 2. User config file (~/.hindsight/opencode.json)
    const userConfigPath = join(homedir(), '.hindsight', 'opencode.json');
    const fileConfig = loadSettingsFile(userConfigPath);
    for (const [key, value] of Object.entries(fileConfig)) {
        if (value !== null && value !== undefined) {
            config[key] = value;
        }
    }

    // 3. Plugin options (from opencode.json: ["@vectorize-io/opencode-hindsight", { ... }])
    if (pluginOptions) {
        for (const [key, value] of Object.entries(pluginOptions)) {
            if (value !== null && value !== undefined) {
                config[key] = value;
            }
        }
    }

    // 4. Environment variable overrides (highest priority)
    for (const [envName, [key, typ]] of Object.entries(ENV_OVERRIDES)) {
        const val = process.env[envName];
        if (val !== undefined) {
            const castVal = castEnv(val, typ);
            if (castVal !== null) {
                config[key] = castVal;
            }
        }
    }

    const result = config as unknown as HindsightConfig;

    // Validate enum-like fields to catch typos early
    const VALID_RETAIN_MODES = ['full-session', 'last-turn'];
    if (!VALID_RETAIN_MODES.includes(result.retainMode)) {
        console.error(
            `[Hindsight] Unknown retainMode "${result.retainMode}" — ` +
                `valid: ${VALID_RETAIN_MODES.join(', ')}. Falling back to "full-session".`,
        );
        result.retainMode = 'full-session';
    }

    const VALID_BUDGETS = ['low', 'mid', 'high'];
    if (!VALID_BUDGETS.includes(result.recallBudget)) {
        console.error(
            `[Hindsight] Unknown recallBudget "${result.recallBudget}" — ` +
                `valid: ${VALID_BUDGETS.join(', ')}. Falling back to "mid".`,
        );
        result.recallBudget = 'mid';
    }

    return result;
}

export function debugLog(config: HindsightConfig, ...args: unknown[]): void {
    if (config.debug) {
        console.error('[Hindsight]', ...args);
    }
}
