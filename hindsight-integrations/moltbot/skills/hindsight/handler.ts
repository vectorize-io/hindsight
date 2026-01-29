// Handler for memory_search tool
// This will be called when the agent invokes memory_search

import { getClient } from '../../src/index.js';

export interface ToolContext {
  query: string;
  args: Record<string, unknown>;
}

export async function handle(ctx: ToolContext): Promise<string> {
  try {
    const { query } = ctx;

    const client = getClient();
    if (!client) {
      throw new Error('Hindsight client not initialized');
    }

    // Call Hindsight recall API
    const response = await client.recall({
      query,
      limit: 10,
    });

    // Format results for the agent
    if (!response.results || response.results.length === 0) {
      return 'No relevant memories found for this query.';
    }

    const formatted = response.results
      .map((result: any, idx: number) => {
        const score = result.score ? ` (relevance: ${result.score.toFixed(2)})` : '';
        const date = result.metadata?.created_at
          ? ` [${new Date(result.metadata.created_at).toLocaleDateString()}]`
          : '';
        return `${idx + 1}. ${result.content}${score}${date}`;
      })
      .join('\n\n');

    return `Found ${response.results.length} relevant memories:\n\n${formatted}`;
  } catch (error) {
    console.error('[Hindsight] memory_search error:', error);
    return `Error searching memories: ${error instanceof Error ? error.message : String(error)}`;
  }
}
