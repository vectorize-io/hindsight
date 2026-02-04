import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createHindsightTools, type HindsightClient } from './index.js';

describe('createHindsightTools', () => {
  let mockClient: HindsightClient;

  beforeEach(() => {
    mockClient = {
      retain: vi.fn(),
      recall: vi.fn(),
      reflect: vi.fn(),
    };
  });

  describe('tool creation', () => {
    it('should create all three tools', () => {
      const tools = createHindsightTools({ client: mockClient });

      expect(tools).toHaveProperty('retain');
      expect(tools).toHaveProperty('recall');
      expect(tools).toHaveProperty('reflect');
      expect(typeof tools.retain.execute).toBe('function');
      expect(typeof tools.recall.execute).toBe('function');
      expect(typeof tools.reflect.execute).toBe('function');
    });

    it('should use default descriptions when not provided', () => {
      const tools = createHindsightTools({ client: mockClient });

      expect(tools.retain.description).toContain('Store information in long-term memory');
      expect(tools.recall.description).toContain('Search memory for relevant information');
      expect(tools.reflect.description).toContain('Analyze memories to form insights');
    });

    it('should use custom descriptions when provided', () => {
      const tools = createHindsightTools({
        client: mockClient,
        retainDescription: 'Custom retain description',
        recallDescription: 'Custom recall description',
        reflectDescription: 'Custom reflect description',
      });

      expect(tools.retain.description).toBe('Custom retain description');
      expect(tools.recall.description).toBe('Custom recall description');
      expect(tools.reflect.description).toBe('Custom reflect description');
    });
  });

  describe('retain tool', () => {
    it('should call client.retain with correct parameters', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.retain).mockResolvedValue({
        success: true,
        bank_id: 'test-bank',
        items_count: 5,
        async: false,
      });

      const result = await tools.retain.execute({
        bankId: 'test-bank',
        content: 'Test content',
      });

      expect(mockClient.retain).toHaveBeenCalledWith('test-bank', 'Test content', {
        documentId: undefined,
        timestamp: undefined,
        context: undefined,
      });
      expect(result).toEqual({ success: true, itemsCount: 5 });
    });

    it('should pass optional parameters to client.retain', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.retain).mockResolvedValue({
        success: true,
        bank_id: 'test-bank',
        items_count: 3,
        async: false,
      });

      await tools.retain.execute({
        bankId: 'test-bank',
        content: 'Test content',
        documentId: 'doc-123',
        timestamp: '2024-01-01T00:00:00Z',
        context: 'Test context',
      });

      expect(mockClient.retain).toHaveBeenCalledWith('test-bank', 'Test content', {
        documentId: 'doc-123',
        timestamp: '2024-01-01T00:00:00Z',
        context: 'Test context',
      });
    });

    it('should transform response correctly', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.retain).mockResolvedValue({
        success: true,
        bank_id: 'test-bank',
        items_count: 10,
        async: false,
      });

      const result = await tools.retain.execute({
        bankId: 'test-bank',
        content: 'Test content',
      });

      expect(result).toEqual({ success: true, itemsCount: 10 });
    });
  });

  describe('recall tool', () => {
    it('should call client.recall with correct parameters', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.recall).mockResolvedValue({
        results: [
          {
            id: 'fact-1',
            text: 'Test fact',
            type: 'preference',
          },
        ],
      });

      const result = await tools.recall.execute({
        bankId: 'test-bank',
        query: 'Test query',
      });

      expect(mockClient.recall).toHaveBeenCalledWith('test-bank', 'Test query', {
        types: undefined,
        maxTokens: undefined,
        budget: undefined,
        queryTimestamp: undefined,
        includeEntities: undefined,
        includeChunks: undefined,
      });
      expect(result.results).toHaveLength(1);
      expect(result.results[0].id).toBe('fact-1');
    });

    it('should pass all optional parameters to client.recall', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.recall).mockResolvedValue({
        results: [],
      });

      await tools.recall.execute({
        bankId: 'test-bank',
        query: 'Test query',
        types: ['preference', 'fact'],
        maxTokens: 1000,
        budget: 'high',
        queryTimestamp: '2024-01-01T00:00:00Z',
        includeEntities: true,
        includeChunks: true,
      });

      expect(mockClient.recall).toHaveBeenCalledWith('test-bank', 'Test query', {
        types: ['preference', 'fact'],
        maxTokens: 1000,
        budget: 'high',
        queryTimestamp: '2024-01-01T00:00:00Z',
        includeEntities: true,
        includeChunks: true,
      });
    });

    it('should handle empty results', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.recall).mockResolvedValue({
        results: undefined as any,
      });

      const result = await tools.recall.execute({
        bankId: 'test-bank',
        query: 'Test query',
      });

      expect(result.results).toEqual([]);
    });

    it('should include entities when present', async () => {
      const tools = createHindsightTools({ client: mockClient });
      const entities = {
        'entity-1': {
          entity_id: 'entity-1',
          canonical_name: 'Alice',
          observations: [{ text: 'Alice loves hiking' }],
        },
      };

      vi.mocked(mockClient.recall).mockResolvedValue({
        results: [],
        entities,
      });

      const result = await tools.recall.execute({
        bankId: 'test-bank',
        query: 'Test query',
        includeEntities: true,
      });

      expect(result.entities).toEqual(entities);
    });
  });

  describe('reflect tool', () => {
    it('should call client.reflect with correct parameters', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.reflect).mockResolvedValue({
        text: 'Reflection result',
        based_on: [
          {
            id: 'fact-1',
            text: 'Supporting fact',
          },
        ],
      });

      const result = await tools.reflect.execute({
        bankId: 'test-bank',
        query: 'What are my preferences?',
      });

      expect(mockClient.reflect).toHaveBeenCalledWith('test-bank', 'What are my preferences?', {
        context: undefined,
        budget: undefined,
      });
      expect(result.text).toBe('Reflection result');
      expect(result.basedOn).toHaveLength(1);
    });

    it('should pass optional parameters to client.reflect', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.reflect).mockResolvedValue({
        text: 'Reflection result',
      });

      await tools.reflect.execute({
        bankId: 'test-bank',
        query: 'What are my preferences?',
        context: 'User context',
        budget: 'mid',
      });

      expect(mockClient.reflect).toHaveBeenCalledWith('test-bank', 'What are my preferences?', {
        context: 'User context',
        budget: 'mid',
      });
    });

    it('should handle empty text response with fallback', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.reflect).mockResolvedValue({
        text: undefined as any,
      });

      const result = await tools.reflect.execute({
        bankId: 'test-bank',
        query: 'Test query',
      });

      expect(result.text).toBe('No insights available yet.');
    });

    it('should include basedOn facts when present', async () => {
      const tools = createHindsightTools({ client: mockClient });
      const basedOn = [
        {
          id: 'fact-1',
          text: 'User prefers spicy food',
          type: 'preference',
        },
        {
          id: 'fact-2',
          text: 'User is allergic to nuts',
          type: 'health',
        },
      ];

      vi.mocked(mockClient.reflect).mockResolvedValue({
        text: 'Based on your history, you prefer spicy Asian cuisine',
        based_on: basedOn,
      });

      const result = await tools.reflect.execute({
        bankId: 'test-bank',
        query: 'What do I like?',
      });

      expect(result.basedOn).toEqual(basedOn);
    });
  });

  describe('error handling', () => {
    it('should propagate errors from client.retain', async () => {
      const tools = createHindsightTools({ client: mockClient });
      const error = new Error('Retain failed');
      vi.mocked(mockClient.retain).mockRejectedValue(error);

      await expect(
        tools.retain.execute({
          bankId: 'test-bank',
          content: 'Test content',
        })
      ).rejects.toThrow('Retain failed');
    });

    it('should propagate errors from client.recall', async () => {
      const tools = createHindsightTools({ client: mockClient });
      const error = new Error('Recall failed');
      vi.mocked(mockClient.recall).mockRejectedValue(error);

      await expect(
        tools.recall.execute({
          bankId: 'test-bank',
          query: 'Test query',
        })
      ).rejects.toThrow('Recall failed');
    });

    it('should propagate errors from client.reflect', async () => {
      const tools = createHindsightTools({ client: mockClient });
      const error = new Error('Reflect failed');
      vi.mocked(mockClient.reflect).mockRejectedValue(error);

      await expect(
        tools.reflect.execute({
          bankId: 'test-bank',
          query: 'Test query',
        })
      ).rejects.toThrow('Reflect failed');
    });
  });

  describe('budget schema', () => {
    it('should accept valid budget values', async () => {
      const tools = createHindsightTools({ client: mockClient });
      vi.mocked(mockClient.recall).mockResolvedValue({ results: [] });

      for (const budget of ['low', 'mid', 'high'] as const) {
        await tools.recall.execute({
          bankId: 'test-bank',
          query: 'Test',
          budget,
        });

        expect(mockClient.recall).toHaveBeenCalledWith('test-bank', 'Test', {
          types: undefined,
          maxTokens: undefined,
          budget,
          queryTimestamp: undefined,
          includeEntities: undefined,
          includeChunks: undefined,
        });
      }
    });
  });
});
