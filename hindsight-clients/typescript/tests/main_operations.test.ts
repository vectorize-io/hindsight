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

describe('TestBankProfile', () => {
    test('get bank profile', async () => {
        const bankId = randomBankId();

        // Create bank with background
        await client.createBank(bankId, {
            name: 'Test Agent',
            background: 'I am a helpful assistant for testing.',
        });

        // Get bank profile
        const profile = await client.getBankProfile(bankId);

        expect(profile).not.toBeNull();
        expect(profile.bank_id).toBe(bankId);
        expect(profile.name).toBe('Test Agent');
        expect(profile.background).toBe('I am a helpful assistant for testing.');
    });
});

describe('TestBankStats', () => {
    let bankId: string;

    beforeAll(async () => {
        bankId = randomBankId();
        // Setup: Store some test memories
        await client.retainBatch(bankId, [
            { content: 'Alice likes Python programming' },
            { content: 'Bob enjoys hiking in the mountains' },
        ]);
    });

    test('get bank stats', async () => {
        const { sdk, createClient, createConfig } = await import('../src');
        const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_API_URL }));

        const { data: stats } = await sdk.getAgentStats({
            client: apiClient,
            path: { bank_id: bankId },
        });

        expect(stats).not.toBeNull();
        expect(stats!.bank_id).toBe(bankId);
        expect(stats!.total_nodes).toBeGreaterThanOrEqual(0);
        expect(stats!.total_links).toBeGreaterThanOrEqual(0);
        expect(stats!.total_documents).toBeGreaterThanOrEqual(0);
        expect(typeof stats!.nodes_by_fact_type).toBe('object');
        expect(typeof stats!.links_by_link_type).toBe('object');
    });
});

describe('TestOperations', () => {
    test('list operations', async () => {
        const bankId = randomBankId();
        const { sdk, createClient, createConfig } = await import('../src');

        // First create an async operation
        await client.retain(bankId, 'Test content for async operation', {
            async: true,
        });

        const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_API_URL }));
        const { data: response } = await sdk.listOperations({
            client: apiClient,
            path: { bank_id: bankId },
        });

        expect(response).not.toBeNull();
        expect(response!.bank_id).toBe(bankId);
        expect(Array.isArray(response!.operations)).toBe(true);
    });
});

describe('TestDocuments', () => {
    test('delete document', async () => {
        const bankId = randomBankId();
        const docId = `test-doc-${Math.random().toString(36).slice(2, 10)}`;
        const { sdk, createClient, createConfig } = await import('../src');

        // First create a document
        const retainResponse = await client.retain(bankId, 'Test document content for deletion', {
            documentId: docId,
        });
        expect(retainResponse.success).toBe(true);

        const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_API_URL }));
        const { data: response } = await sdk.deleteDocument({
            client: apiClient,
            path: { bank_id: bankId, document_id: docId },
        });

        expect(response).not.toBeNull();
        expect(response!.success).toBe(true);
        expect(response!.document_id).toBe(docId);
        expect(response!.memory_units_deleted).toBeGreaterThanOrEqual(0);
    });

    test('get document', async () => {
        const bankId = randomBankId();
        const docId = `test-doc-${Math.random().toString(36).slice(2, 10)}`;
        const { sdk, createClient, createConfig } = await import('../src');

        // First create a document
        await client.retain(bankId, 'Test document content for retrieval', {
            documentId: docId,
        });

        const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_API_URL }));
        const { data: document } = await sdk.getDocument({
            client: apiClient,
            path: { bank_id: bankId, document_id: docId },
        });

        expect(document).not.toBeNull();
        expect(document!.id).toBe(docId);
        expect(document!.original_text).toContain('Test document content');
    });
});

describe('TestEntities', () => {
    let bankId: string;

    beforeAll(async () => {
        bankId = randomBankId();
        // Create memories that will generate entities
        await client.retainBatch(bankId, [
            { content: 'Alice works at Google as a software engineer' },
            { content: 'Bob is friends with Alice and works at Microsoft' },
        ]);
    });

    test('list entities', async () => {
        const { sdk, createClient, createConfig } = await import('../src');
        const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_API_URL }));

        const { data: response } = await sdk.listEntities({
            client: apiClient,
            path: { bank_id: bankId },
        });

        expect(response).not.toBeNull();
        expect(response!.items).toBeDefined();
        expect(Array.isArray(response!.items)).toBe(true);
    });

    test('get entity', async () => {
        const { sdk, createClient, createConfig } = await import('../src');
        const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_API_URL }));

        // First list entities to get an ID
        const { data: listResponse } = await sdk.listEntities({
            client: apiClient,
            path: { bank_id: bankId },
        });

        if (listResponse?.items && listResponse.items.length > 0) {
            const entityId = listResponse.items[0].id;

            const { data: entity } = await sdk.getEntity({
                client: apiClient,
                path: { bank_id: bankId, entity_id: entityId },
            });

            expect(entity).not.toBeNull();
            expect(entity!.id).toBe(entityId);
        }
    });
});

describe('TestDeleteBank', () => {
    test('delete bank', async () => {
        const bankId = randomBankId();
        const { sdk, createClient, createConfig } = await import('../src');

        // First create a bank with some data
        await client.createBank(bankId, {
            name: 'Bank to delete',
            background: 'This bank will be deleted',
        });
        await client.retain(bankId, 'Some memory to store');

        const apiClient = createClient(createConfig({ baseUrl: HINDSIGHT_API_URL }));
        const { data: response } = await sdk.deleteBank({
            client: apiClient,
            path: { bank_id: bankId },
        });

        expect(response).not.toBeNull();
        expect(response!.success).toBe(true);

        // Verify bank data is deleted - memories should be gone
        const memories = await client.listMemories(bankId);
        expect(memories.total).toBe(0);
    });
});

describe('TestRecallIncludeOptions', () => {
    let bankId: string;

    beforeAll(async () => {
        bankId = randomBankId();
        await client.retainBatch(bankId, [
            { content: 'Alice works at Google as a software engineer' },
            { content: 'Bob is a researcher at OpenAI' },
        ]);
    });

    test('entities included by default', async () => {
        const response = await client.recall(bankId, 'Where does Alice work?');

        expect(response).not.toBeNull();
        expect(response.results!.length).toBeGreaterThan(0);
        // entities should be present when includeEntities is not specified (default: true)
        expect(response.entities).toBeDefined();
    });

    test('entities excluded when includeEntities is false', async () => {
        const response = await client.recall(bankId, 'Where does Alice work?', {
            includeEntities: false,
        });

        expect(response).not.toBeNull();
        expect(response.results!.length).toBeGreaterThan(0);
        // entities should be absent when explicitly disabled
        expect(response.entities).toBeFalsy();
    });

    test('entities included when includeEntities is true', async () => {
        const response = await client.recall(bankId, 'Where does Alice work?', {
            includeEntities: true,
        });

        expect(response).not.toBeNull();
        expect(response.results!.length).toBeGreaterThan(0);
        expect(response.entities).toBeDefined();
    });
});

describe('TestMission', () => {
    test('set mission', async () => {
        const bankId = randomBankId();
        const response = await client.setMission(
            bankId,
            'Be a helpful PM tracking sprint progress and team capacity'
        );

        expect(response).not.toBeNull();
        expect(response.bank_id).toBe(bankId);
        expect(response.mission).toBe('Be a helpful PM tracking sprint progress and team capacity');
    });
});
