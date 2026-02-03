import { exec } from 'child_process';
import { promisify } from 'util';
import type {
  RetainRequest,
  RetainResponse,
  RecallRequest,
  RecallResponse,
} from './types.js';

const execAsync = promisify(exec);

/**
 * Escape a string for use as a single-quoted shell argument.
 *
 * In POSIX shells, single-quoted strings treat ALL characters literally
 * except for the single quote itself. To include a literal single quote,
 * we use the pattern: end quote + escaped quote + start quote = '\''
 *
 * Example: "It's $100" becomes 'It'\''s $100'
 * Shell interprets: 'It' + \' + 's $100' = It's $100
 *
 * This handles ALL shell-special characters including:
 * - $ (variable expansion)
 * - ` (command substitution)
 * - ! (history expansion)
 * - ? * [ ] (glob patterns)
 * - ( ) { } (subshell/brace expansion)
 * - < > | & ; (redirection/control)
 * - \ " # ~ newlines
 *
 * @param arg - The string to escape
 * @returns The escaped string (without surrounding quotes - caller adds those)
 */
export function escapeShellArg(arg: string): string {
  // Replace single quotes with the escape sequence: '\''
  // This ends the current single-quoted string, adds an escaped literal quote,
  // and starts a new single-quoted string.
  return arg.replace(/'/g, "'\\''");
}

export class HindsightClient {
  private bankId: string = 'default'; // Always use default bank
  private llmProvider: string;
  private llmApiKey: string;
  private llmModel?: string;
  private embedVersion: string;
  private embedPackagePath?: string;

  constructor(llmProvider: string, llmApiKey: string, llmModel?: string, embedVersion: string = 'latest', embedPackagePath?: string) {
    this.llmProvider = llmProvider;
    this.llmApiKey = llmApiKey;
    this.llmModel = llmModel;
    this.embedVersion = embedVersion || 'latest';
    this.embedPackagePath = embedPackagePath;
  }

  /**
   * Get the command prefix to run hindsight-embed (either local or from PyPI)
   */
  private getEmbedCommandPrefix(): string {
    if (this.embedPackagePath) {
      // Local package: uv run --directory <path> hindsight-embed
      return `uv run --directory ${this.embedPackagePath} hindsight-embed`;
    } else {
      // PyPI package: uvx hindsight-embed@version
      const embedPackage = this.embedVersion ? `hindsight-embed@${this.embedVersion}` : 'hindsight-embed@latest';
      return `uvx ${embedPackage}`;
    }
  }

  setBankId(bankId: string): void {
    this.bankId = bankId;
  }

  async setBankMission(mission: string): Promise<void> {
    if (!mission || mission.trim().length === 0) {
      return;
    }

    const escapedMission = escapeShellArg(mission);
    const embedCmd = this.getEmbedCommandPrefix();
    const cmd = `${embedCmd} --profile openclaw bank mission ${this.bankId} '${escapedMission}'`;

    try {
      const { stdout } = await execAsync(cmd);
      console.log(`[Hindsight] Bank mission set: ${stdout.trim()}`);
    } catch (error) {
      // Don't fail if mission set fails - bank might not exist yet, will be created on first retain
      console.warn(`[Hindsight] Could not set bank mission (bank may not exist yet): ${error}`);
    }
  }

  async retain(request: RetainRequest): Promise<RetainResponse> {
    const content = escapeShellArg(request.content);
    const docId = escapeShellArg(request.document_id || 'conversation');

    const embedCmd = this.getEmbedCommandPrefix();
    const cmd = `${embedCmd} --profile openclaw memory retain ${this.bankId} '${content}' --doc-id '${docId}' --async`;

    try {
      const { stdout } = await execAsync(cmd);
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
    const query = escapeShellArg(request.query);
    const maxTokens = request.max_tokens || 1024;

    const embedCmd = this.getEmbedCommandPrefix();
    const cmd = `${embedCmd} --profile openclaw memory recall ${this.bankId} '${query}' --output json --max-tokens ${maxTokens}`;

    try {
      const { stdout } = await execAsync(cmd);

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
