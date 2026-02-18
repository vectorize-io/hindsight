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
    it('should create all tools', () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });

      expect(tools).toHaveProperty('retain');
      expect(tools).toHaveProperty('recall');
      expect(tools).toHaveProperty('reflect');
      expect(tools).toHaveProperty('getMentalModel');
      expect(tools).toHaveProperty('getDocument');
      expect(typeof tools.retain.execute).toBe('function');
      expect(typeof tools.recall.execute).toBe('function');
      expect(typeof tools.reflect.execute).toBe('function');
    });

    it('should use default descriptions when not provided', () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });

      expect(tools.retain.description).toContain('Store information in long-term memory');
      expect(tools.recall.description).toContain('Search memory for relevant information');
      expect(tools.reflect.description).toContain('Analyze memories to form insights');
    });

    it('should use custom descriptions from nested options', () => {
      const tools = createHindsightTools({
        client: mockClient,
        bankId: 'test-bank',
        retain: { description: 'Custom retain description' },
        recall: { description: 'Custom recall description' },
        reflect: { description: 'Custom reflect description' },
      });

      expect(tools.retain.description).toBe('Custom retain description');
      expect(tools.recall.description).toBe('Custom recall description');
      expect(tools.reflect.description).toBe('Custom reflect description');
    });
  });

  describe('retain tool', () => {
    it('should call client.retain with agent inputs and constructor defaults', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.retain).mockResolvedValue({
        success: true,
        bank_id: 'test-bank',
        items_count: 5,
        async: false,
      });

      const result = await tools.retain.execute({ content: 'Test content' });

      expect(mockClient.retain).toHaveBeenCalledWith('test-bank', 'Test content', {
        documentId: undefined,
        timestamp: undefined,
        context: undefined,
        tags: undefined,
        metadata: undefined,
        async: false,
      });
      expect(result).toEqual({ success: true, itemsCount: 5 });
    });

    it('should pass agent-provided optional inputs', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.retain).mockResolvedValue({
        success: true,
        bank_id: 'test-bank',
        items_count: 3,
        async: false,
      });

      await tools.retain.execute({
        content: 'Test content',
        documentId: 'doc-123',
        timestamp: '2024-01-01T00:00:00Z',
        context: 'Test context',
      });

      expect(mockClient.retain).toHaveBeenCalledWith('test-bank', 'Test content', {
        documentId: 'doc-123',
        timestamp: '2024-01-01T00:00:00Z',
        context: 'Test context',
        tags: undefined,
        metadata: undefined,
        async: false,
      });
    });

    it('should apply constructor-level retain options', async () => {
      const tools = createHindsightTools({
        client: mockClient,
        bankId: 'test-bank',
        retain: {
          async: true,
          tags: ['env:prod', 'app:support'],
          metadata: { version: '1.0' },
        },
      });
      vi.mocked(mockClient.retain).mockResolvedValue({
        success: true,
        bank_id: 'test-bank',
        items_count: 1,
        async: true,
      });

      await tools.retain.execute({ content: 'Test content' });

      expect(mockClient.retain).toHaveBeenCalledWith('test-bank', 'Test content', {
        documentId: undefined,
        timestamp: undefined,
        context: undefined,
        tags: ['env:prod', 'app:support'],
        metadata: { version: '1.0' },
        async: true,
      });
    });
  });

  describe('recall tool', () => {
    it('should call client.recall with agent inputs and constructor defaults', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.recall).mockResolvedValue({
        results: [{ id: 'fact-1', text: 'Test fact', type: 'preference' }],
      });

      const result = await tools.recall.execute({ query: 'Test query' });

      expect(mockClient.recall).toHaveBeenCalledWith('test-bank', 'Test query', {
        types: undefined,
        maxTokens: undefined,
        budget: 'mid',
        queryTimestamp: undefined,
        includeEntities: false,
        includeChunks: false,
      });
      expect(result.results).toHaveLength(1);
      expect(result.results[0].id).toBe('fact-1');
    });

    it('should pass agent-provided queryTimestamp', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.recall).mockResolvedValue({ results: [] });

      await tools.recall.execute({
        query: 'Test query',
        queryTimestamp: '2024-01-01T00:00:00Z',
      });

      expect(mockClient.recall).toHaveBeenCalledWith('test-bank', 'Test query', {
        types: undefined,
        maxTokens: undefined,
        budget: 'mid',
        queryTimestamp: '2024-01-01T00:00:00Z',
        includeEntities: false,
        includeChunks: false,
      });
    });

    it('should apply constructor-level recall options', async () => {
      const tools = createHindsightTools({
        client: mockClient,
        bankId: 'test-bank',
        recall: {
          types: ['preference', 'fact'],
          maxTokens: 1000,
          budget: 'high',
          includeEntities: true,
          includeChunks: true,
        },
      });
      vi.mocked(mockClient.recall).mockResolvedValue({ results: [] });

      await tools.recall.execute({ query: 'Test query' });

      expect(mockClient.recall).toHaveBeenCalledWith('test-bank', 'Test query', {
        types: ['preference', 'fact'],
        maxTokens: 1000,
        budget: 'high',
        queryTimestamp: undefined,
        includeEntities: true,
        includeChunks: true,
      });
    });

    it('should handle empty results', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.recall).mockResolvedValue({ results: undefined as any });

      const result = await tools.recall.execute({ query: 'Test query' });

      expect(result.results).toEqual([]);
    });

    it('should include entities when present', async () => {
      const tools = createHindsightTools({
        client: mockClient,
        bankId: 'test-bank',
        recall: { includeEntities: true },
      });
      const entities = {
        'entity-1': {
          entity_id: 'entity-1',
          canonical_name: 'Alice',
          observations: [{ text: 'Alice loves hiking' }],
        },
      };
      vi.mocked(mockClient.recall).mockResolvedValue({ results: [], entities });

      const result = await tools.recall.execute({ query: 'Test query' });

      expect(result.entities).toEqual(entities);
    });
  });

  describe('reflect tool', () => {
    it('should call client.reflect with agent inputs and constructor defaults', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.reflect).mockResolvedValue({
        text: 'Reflection result',
        based_on: [{ id: 'fact-1', text: 'Supporting fact' }],
      });

      const result = await tools.reflect.execute({ query: 'What are my preferences?' });

      expect(mockClient.reflect).toHaveBeenCalledWith('test-bank', 'What are my preferences?', {
        context: undefined,
        budget: 'mid',
      });
      expect(result.text).toBe('Reflection result');
      expect(result.basedOn).toHaveLength(1);
    });

    it('should pass agent-provided context', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.reflect).mockResolvedValue({ text: 'Reflection result' });

      await tools.reflect.execute({
        query: 'What are my preferences?',
        context: 'User context',
      });

      expect(mockClient.reflect).toHaveBeenCalledWith('test-bank', 'What are my preferences?', {
        context: 'User context',
        budget: 'mid',
      });
    });

    it('should apply constructor-level reflect budget', async () => {
      const tools = createHindsightTools({
        client: mockClient,
        bankId: 'test-bank',
        reflect: { budget: 'low' },
      });
      vi.mocked(mockClient.reflect).mockResolvedValue({ text: 'Reflection result' });

      await tools.reflect.execute({ query: 'Test query' });

      expect(mockClient.reflect).toHaveBeenCalledWith('test-bank', 'Test query', {
        context: undefined,
        budget: 'low',
      });
    });

    it('should handle empty text response with fallback', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.reflect).mockResolvedValue({ text: undefined as any });

      const result = await tools.reflect.execute({ query: 'Test query' });

      expect(result.text).toBe('No insights available yet.');
    });

    it('should include basedOn facts when present', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      const basedOn = [
        { id: 'fact-1', text: 'User prefers spicy food', type: 'preference' },
        { id: 'fact-2', text: 'User is allergic to nuts', type: 'health' },
      ];
      vi.mocked(mockClient.reflect).mockResolvedValue({
        text: 'Based on your history, you prefer spicy Asian cuisine',
        based_on: basedOn,
      });

      const result = await tools.reflect.execute({ query: 'What do I like?' });

      expect(result.basedOn).toEqual(basedOn);
    });
  });

  describe('error handling', () => {
    it('should propagate errors from client.retain', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.retain).mockRejectedValue(new Error('Retain failed'));

      await expect(tools.retain.execute({ content: 'Test content' })).rejects.toThrow('Retain failed');
    });

    it('should propagate errors from client.recall', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.recall).mockRejectedValue(new Error('Recall failed'));

      await expect(tools.recall.execute({ query: 'Test query' })).rejects.toThrow('Recall failed');
    });

    it('should propagate errors from client.reflect', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.reflect).mockRejectedValue(new Error('Reflect failed'));

      await expect(tools.reflect.execute({ query: 'Test query' })).rejects.toThrow('Reflect failed');
    });
  });

  describe('budget defaults', () => {
    it('should default recall budget to mid', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.recall).mockResolvedValue({ results: [] });

      await tools.recall.execute({ query: 'Test' });

      expect(mockClient.recall).toHaveBeenCalledWith('test-bank', 'Test', expect.objectContaining({ budget: 'mid' }));
    });

    it('should default reflect budget to mid', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank' });
      vi.mocked(mockClient.reflect).mockResolvedValue({ text: 'ok' });

      await tools.reflect.execute({ query: 'Test' });

      expect(mockClient.reflect).toHaveBeenCalledWith('test-bank', 'Test', expect.objectContaining({ budget: 'mid' }));
    });

    it('should accept low/mid/high budget values', async () => {
      vi.mocked(mockClient.recall).mockResolvedValue({ results: [] });

      for (const budget of ['low', 'mid', 'high'] as const) {
        const tools = createHindsightTools({ client: mockClient, bankId: 'test-bank', recall: { budget } });
        await tools.recall.execute({ query: 'Test' });
        expect(mockClient.recall).toHaveBeenCalledWith('test-bank', 'Test', expect.objectContaining({ budget }));
      }
    });
  });

  describe('bankId enforcement', () => {
    it('should always use the bankId from constructor options', async () => {
      const tools = createHindsightTools({ client: mockClient, bankId: 'forced-bank' });
      vi.mocked(mockClient.retain).mockResolvedValue({
        success: true,
        bank_id: 'forced-bank',
        items_count: 1,
        async: false,
      });

      await tools.retain.execute({ content: 'Test' });

      expect(mockClient.retain).toHaveBeenCalledWith('forced-bank', 'Test', expect.anything());
    });
  });
});
