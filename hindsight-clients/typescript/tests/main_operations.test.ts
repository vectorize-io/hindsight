/**
 * Tests for Hindsight TypeScript client.
 *
 * These tests require a running Hindsight API server.
 */

import { HindsightClient } from '../src';

// Test configuration
const HINDSIGHT_API_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

let client: HindsightClient;

beforeAll(() => {
    client = new HindsightClient({ baseUrl: HINDSIGHT_API_URL });
});

function randomBankId(): string {
    return `test_bank_${Math.random().toString(36).slice(2, 14)}`;
}

describe('TestRetain', () => {
    test('retain single memory', async () => {
        const bankId = randomBankId();
        const response = await client.retain(
            bankId,
            'Alice loves artificial intelligence and machine learning'
        );

        expect(response).not.toBeNull();
        expect(response.success).toBe(true);
    });

    test('retain memory with context', async () => {
        const bankId = randomBankId();
        const response = await client.retain(bankId, 'Bob went hiking in the mountains', {
            timestamp: new Date('2024-01-15T10:30:00'),
            context: 'outdoor activities',
        });

        expect(response).not.toBeNull();
        expect(response.success).toBe(true);
    });

    test('retain batch memories', async () => {
        const bankId = randomBankId();
        const response = await client.retainBatch(bankId, [
            { content: 'Charlie enjoys reading science fiction books' },
            { content: 'Diana is learning to play the guitar', context: 'hobbies' },
            { content: 'Eve completed a marathon last month', timestamp: '2024-10-15' },
        ]);

        expect(response).not.toBeNull();
        expect(response.success).toBe(true);
        expect(response.items_count).toBe(3);
    });
});

describe('TestRecall', () => {
    let bankId: string;

    beforeAll(async () => {
        bankId = randomBankId();
        // Setup: Store some test memories before recall tests
        await client.retainBatch(bankId, [
            { content: 'Alice loves programming in Python' },
            { content: 'Bob enjoys hiking and outdoor adventures' },
            { content: 'Charlie is interested in quantum physics' },
            { content: 'Diana plays the violin beautifully' },
        ]);
    });

    test('recall basic', async () => {
        const response = await client.recall(bankId, 'What does Alice like?');

        expect(response).not.toBeNull();
        expect(response.results).toBeDefined();
        expect(response.results!.length).toBeGreaterThan(0);

        // Check that at least one result contains relevant information
        const resultTexts = response.results!.map((r) => r.text || '');
        const hasRelevant = resultTexts.some(
            (text: string) => text.includes('Alice') || text.includes('Python') || text.includes('programming')
        );
        expect(hasRelevant).toBe(true);
    });

    test('recall with max tokens', async () => {
        const response = await client.recall(bankId, 'outdoor activities', {
            maxTokens: 1024,
        });

        expect(response).not.toBeNull();
        expect(response.results).toBeDefined();
        expect(Array.isArray(response.results)).toBe(true);
    });

    test('recall with types filter', async () => {
        const response = await client.recall(bankId, "What are people's hobbies?", {
            types: ['world'],
            maxTokens: 2048,
            trace: true,
        });

        expect(response).not.toBeNull();
        expect(response.results).toBeDefined();
    });
});

describe('TestReflect', () => {
    let bankId: string;

    beforeAll(async () => {
        bankId = randomBankId();
        // Setup: Create bank and store test memories
        await client.createBank(bankId, {
            background: 'I am a helpful AI assistant interested in technology and science.',
        });

        await client.retainBatch(bankId, [
            { content: 'The Python programming language is great for data science' },
            { content: 'Machine learning models can recognize patterns in data' },
            { content: 'Neural networks are inspired by biological neurons' },
        ]);
    });

    test('reflect basic', async () => {
        const response = await client.reflect(
            bankId,
            'What do you think about artificial intelligence?'
        );

        expect(response).not.toBeNull();
        expect(response.text).toBeDefined();
        expect(response.text!.length).toBeGreaterThan(0);
    });

    test('reflect with context', async () => {
        const response = await client.reflect(bankId, 'Should I learn Python?', {
            context: "I'm interested in starting a career in data science",
            budget: 'low',
        });

        expect(response).not.toBeNull();
        expect(response.text).toBeDefined();
        expect(response.text!.length).toBeGreaterThan(0);
    });
});

describe('TestListMemories', () => {
    let bankId: string;

    beforeAll(async () => {
        bankId = randomBankId();
        // Setup: Store some test memories synchronously
        await client.retainBatch(bankId, [
            { content: 'Alice likes topic number 0' },
            { content: 'Alice likes topic number 1' },
            { content: 'Alice likes topic number 2' },
            { content: 'Alice likes topic number 3' },
            { content: 'Alice likes topic number 4' },
        ]);
    });

    test('list all memories', async () => {
        const response = await client.listMemories(bankId);

        expect(response).not.toBeNull();
        expect(response.items).toBeDefined();
        expect(response.total).toBeDefined();
        expect(response.items!.length).toBeGreaterThan(0);
    });

    test('list with pagination', async () => {
        const response = await client.listMemories(bankId, {
            limit: 2,
            offset: 0,
        });

        expect(response).not.toBeNull();
        expect(response.items).toBeDefined();
        expect(response.items!.length).toBeLessThanOrEqual(2);
    });
});

describe('TestEndToEndWorkflow', () => {
    test('complete workflow', async () => {
        const workflowBankId = randomBankId();

        // 1. Create bank
        await client.createBank(workflowBankId, {
            background: 'I am a software engineer who loves Python programming.',
        });

        // 2. Store memories
        const retainResponse = await client.retainBatch(workflowBankId, [
            { content: 'I completed a project using FastAPI' },
            { content: 'I learned about async programming in Python' },
            { content: 'I enjoy working on open source projects' },
        ]);
        expect(retainResponse.success).toBe(true);

        // 3. Search for relevant memories
        const recallResponse = await client.recall(
            workflowBankId,
            'What programming technologies do I use?'
        );
        expect(recallResponse.results!.length).toBeGreaterThan(0);

        // 4. Generate contextual answer
        const reflectResponse = await client.reflect(
            workflowBankId,
            'What are my professional interests?'
        );
        expect(reflectResponse.text).toBeDefined();
        expect(reflectResponse.text!.length).toBeGreaterThan(0);
    });
});
