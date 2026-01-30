import fetch from 'node-fetch';
import { exec } from 'child_process';
import { promisify } from 'util';
import type {
  RetainRequest,
  RetainResponse,
  RecallRequest,
  RecallResponse,
} from './types.js';

const execAsync = promisify(exec);

export class HindsightClient {
  private bankId: string = 'default'; // Always use default bank
  private llmProvider: string;
  private llmApiKey: string;
  private llmModel?: string;

  constructor(llmProvider: string, llmApiKey: string, llmModel?: string) {
    this.llmProvider = llmProvider;
    this.llmApiKey = llmApiKey;
    this.llmModel = llmModel;
  }

  setBankId(bankId: string): void {
    this.bankId = bankId;
  }

  private getEnv(): Record<string, string> {
    const env: Record<string, string> = {
      ...process.env,
      HINDSIGHT_EMBED_LLM_PROVIDER: this.llmProvider,
      HINDSIGHT_EMBED_LLM_API_KEY: this.llmApiKey,
    };

    if (this.llmModel) {
      env.HINDSIGHT_EMBED_LLM_MODEL = this.llmModel;
    }

    return env;
  }

  async retain(request: RetainRequest): Promise<RetainResponse> {
    const content = request.content.replace(/'/g, "'\\''"); // Escape single quotes
    const docId = request.document_id || 'conversation';

    const cmd = `uvx hindsight-embed memory retain ${this.bankId} '${content}' --doc-id '${docId}' --async`;

    try {
      const { stdout } = await execAsync(cmd, { env: this.getEnv() });
      console.log(`[Hindsight] Retained (async): ${stdout.trim()}`);

      // Return a simple response
      return {
        message: 'Memory queued for background processing',
        document_id: docId,
        memory_unit_ids: [],
      };
    } catch (error) {
      throw new Error(`Failed to retain memory: ${error}`);
    }
  }

  async recall(request: RecallRequest): Promise<RecallResponse> {
    const query = request.query.replace(/'/g, "'\\''"); // Escape single quotes
    const maxTokens = request.max_tokens || 1024;

    const cmd = `uvx hindsight-embed memory recall ${this.bankId} '${query}' --output json --max-tokens ${maxTokens}`;

    try {
      const { stdout } = await execAsync(cmd, { env: this.getEnv() });

      // Parse JSON output - returns { entities: {...}, results: [...] }
      const response = JSON.parse(stdout);
      const results = response.results || [];

      return {
        results: results.map((r: any) => ({
          content: r.text || r.content || '',
          score: 1.0, // CLI doesn't return scores
          metadata: {
            document_id: r.document_id,
            chunk_id: r.chunk_id,
            ...r.metadata,
          },
        })),
      };
    } catch (error) {
      throw new Error(`Failed to recall memories: ${error}`);
    }
  }
}
