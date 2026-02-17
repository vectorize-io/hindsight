import { execFile } from 'child_process';
import { promisify } from 'util';
import { writeFile, mkdir, rm } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { randomBytes } from 'crypto';
import type {
  RetainRequest,
  RetainResponse,
  RecallRequest,
  RecallResponse,
} from './types.js';

const execFileAsync = promisify(execFile);

const MAX_BUFFER = 5 * 1024 * 1024; // 5 MB — large transcripts can exceed default 1 MB
const DEFAULT_TIMEOUT_MS = 15_000;

/** Strip null bytes from strings — Node 22 rejects them in execFile() args */
const sanitize = (s: string) => s.replace(/\0/g, '');

/**
 * Sanitize a string for use as a cross-platform filename.
 * Replaces characters illegal on Windows or Unix with underscores.
 */
function sanitizeFilename(name: string): string {
  // Replace characters illegal on Windows (\/:*?"<>|) and control chars
  return name.replace(/[\\/:*?"<>|\x00-\x1f]/g, '_').slice(0, 200) || 'content';
}

export interface HindsightClientOptions {
  llmProvider: string;
  llmApiKey: string;
  llmModel?: string;
  embedVersion?: string;
  embedPackagePath?: string;
  apiUrl?: string;   // Direct HTTP mode — bypass subprocess
  apiToken?: string; // Auth header for HTTP mode
}

export class HindsightClient {
  private bankId: string = 'default';
  private llmProvider: string;
  private llmApiKey: string;
  private llmModel?: string;
  private embedVersion: string;
  private embedPackagePath?: string;
  private apiUrl?: string;
  private apiToken?: string;

  constructor(opts: HindsightClientOptions) {
    this.llmProvider = opts.llmProvider;
    this.llmApiKey = opts.llmApiKey;
    this.llmModel = opts.llmModel;
    this.embedVersion = opts.embedVersion || 'latest';
    this.embedPackagePath = opts.embedPackagePath;
    this.apiUrl = opts.apiUrl?.replace(/\/$/, ''); // strip trailing slash
    this.apiToken = opts.apiToken;
  }

  private get httpMode(): boolean {
    return !!this.apiUrl;
  }

  /**
   * Get the command and base args to run hindsight-embed.
   * Returns [command, ...baseArgs] for use with execFile/spawn (no shell).
   */
  private getEmbedCommand(): string[] {
    if (this.embedPackagePath) {
      return ['uv', 'run', '--directory', this.embedPackagePath, 'hindsight-embed'];
    }
    const embedPackage = this.embedVersion ? `hindsight-embed@${this.embedVersion}` : 'hindsight-embed@latest';
    return ['uvx', embedPackage];
  }

  private httpHeaders(): Record<string, string> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.apiToken) {
      headers['Authorization'] = `Bearer ${this.apiToken}`;
    }
    return headers;
  }

  setBankId(bankId: string): void {
    this.bankId = bankId;
  }

  // --- setBankMission ---

  async setBankMission(mission: string): Promise<void> {
    if (!mission || mission.trim().length === 0) {
      return;
    }

    if (this.httpMode) {
      return this.setBankMissionHttp(mission);
    }
    return this.setBankMissionSubprocess(mission);
  }

  private async setBankMissionHttp(mission: string): Promise<void> {
    try {
      const url = `${this.apiUrl}/v1/default/banks/${encodeURIComponent(this.bankId)}`;
      const res = await fetch(url, {
        method: 'PUT',
        headers: this.httpHeaders(),
        body: JSON.stringify({ mission }),
        signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
      });
      if (!res.ok) {
        const body = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status}: ${body}`);
      }
      console.log(`[Hindsight] Bank mission set via HTTP`);
    } catch (error) {
      console.warn(`[Hindsight] Could not set bank mission (bank may not exist yet): ${error}`);
    }
  }

  private async setBankMissionSubprocess(mission: string): Promise<void> {
    const [cmd, ...baseArgs] = this.getEmbedCommand();
    const args = [...baseArgs, '--profile', 'openclaw', 'bank', 'mission', this.bankId, sanitize(mission)];
    try {
      const { stdout } = await execFileAsync(cmd, args, { maxBuffer: MAX_BUFFER });
      console.log(`[Hindsight] Bank mission set: ${stdout.trim()}`);
    } catch (error) {
      // Don't fail if mission set fails - bank might not exist yet, will be created on first retain
      console.warn(`[Hindsight] Could not set bank mission (bank may not exist yet): ${error}`);
    }
  }

  // --- retain ---

  async retain(request: RetainRequest): Promise<RetainResponse> {
    if (this.httpMode) {
      return this.retainHttp(request);
    }
    return this.retainSubprocess(request);
  }

  private async retainHttp(request: RetainRequest): Promise<RetainResponse> {
    const url = `${this.apiUrl}/v1/default/banks/${encodeURIComponent(this.bankId)}/memories`;
    const body = {
      items: [{
        content: request.content,
        document_id: request.document_id || 'conversation',
        metadata: request.metadata,
      }],
      async: true,
    };

    const res = await fetch(url, {
      method: 'POST',
      headers: this.httpHeaders(),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`Failed to retain memory (HTTP ${res.status}): ${text}`);
    }

    const data = await res.json();
    console.log(`[Hindsight] Retained via HTTP (async): ${JSON.stringify(data).substring(0, 200)}`);

    return {
      message: 'Memory queued for background processing',
      document_id: request.document_id || 'conversation',
      memory_unit_ids: [],
    };
  }

  private async retainSubprocess(request: RetainRequest): Promise<RetainResponse> {
    const docId = request.document_id || 'conversation';

    // Write content to a temp file to avoid E2BIG (ARG_MAX) errors when passing
    // large conversations as arguments.
    const tempDir = join(tmpdir(), `hindsight_${randomBytes(8).toString('hex')}`);
    const safeFilename = sanitizeFilename(docId);
    const tempFile = join(tempDir, `${safeFilename}.txt`);

    try {
      await mkdir(tempDir, { recursive: true });
      await writeFile(tempFile, sanitize(request.content), 'utf8');

      const [cmd, ...baseArgs] = this.getEmbedCommand();
      const args = [...baseArgs, '--profile', 'openclaw', 'memory', 'retain-files', this.bankId, tempFile, '--async'];

      const { stdout } = await execFileAsync(cmd, args, { maxBuffer: MAX_BUFFER });
      console.log(`[Hindsight] Retained (async): ${stdout.trim()}`);

      return {
        message: 'Memory queued for background processing',
        document_id: docId,
        memory_unit_ids: [],
      };
    } catch (error) {
      throw new Error(`Failed to retain memory: ${error}`, { cause: error });
    } finally {
      await rm(tempDir, { recursive: true, force: true }).catch(() => {});
    }
  }

  // --- recall ---

  async recall(request: RecallRequest, timeoutMs?: number): Promise<RecallResponse> {
    if (this.httpMode) {
      return this.recallHttp(request, timeoutMs);
    }
    return this.recallSubprocess(request, timeoutMs);
  }

  private async recallHttp(request: RecallRequest, timeoutMs?: number): Promise<RecallResponse> {
    const url = `${this.apiUrl}/v1/default/banks/${encodeURIComponent(this.bankId)}/memories/recall`;
    // Defense-in-depth: truncate query to stay under API's 500-token limit
    const MAX_QUERY_CHARS = 800;
    const query = request.query.length > MAX_QUERY_CHARS
      ? (console.warn(`[Hindsight] Truncating recall query from ${request.query.length} to ${MAX_QUERY_CHARS} chars`),
         request.query.substring(0, MAX_QUERY_CHARS))
      : request.query;
    const body = {
      query,
      max_tokens: request.max_tokens || 1024,
    };

    const res = await fetch(url, {
      method: 'POST',
      headers: this.httpHeaders(),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs ?? DEFAULT_TIMEOUT_MS),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`Failed to recall memories (HTTP ${res.status}): ${text}`);
    }

    const response = await res.json() as { results?: any[] };
    const results = response.results || [];

    return {
      results: results.map((r: any) => ({
        content: r.text || r.content || '',
        score: r.score ?? 1.0,
        metadata: {
          document_id: r.document_id,
          chunk_id: r.chunk_id,
          ...r.metadata,
        },
      })),
    };
  }

  private async recallSubprocess(request: RecallRequest, timeoutMs?: number): Promise<RecallResponse> {
    const query = sanitize(request.query);
    const maxTokens = request.max_tokens || 1024;
    const [cmd, ...baseArgs] = this.getEmbedCommand();
    const args = [...baseArgs, '--profile', 'openclaw', 'memory', 'recall', this.bankId, query, '--output', 'json', '--max-tokens', String(maxTokens)];

    try {
      const { stdout } = await execFileAsync(cmd, args, {
        maxBuffer: MAX_BUFFER,
        timeout: timeoutMs ?? 30_000, // subprocess gets a longer default
      });

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
      throw new Error(`Failed to recall memories: ${error}`, { cause: error });
    }
  }
}
