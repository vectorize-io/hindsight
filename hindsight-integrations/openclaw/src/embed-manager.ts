import { spawn, ChildProcess } from 'child_process';
import { promises as fs } from 'fs';
import { join } from 'path';
import { homedir } from 'os';
import { execSync } from 'child_process';

export class HindsightEmbedManager {
  private process: ChildProcess | null = null;
  private port: number;
  private baseUrl: string;
  private embedDir: string;
  private llmProvider: string;
  private llmApiKey: string;
  private llmModel?: string;
  private llmBaseUrl?: string;
  private daemonIdleTimeout: number;
  private embedVersion: string;

  constructor(
    port: number,
    llmProvider: string,
    llmApiKey: string,
    llmModel?: string,
    llmBaseUrl?: string,
    daemonIdleTimeout: number = 0, // Default: never timeout
    embedVersion: string = 'latest' // Default: latest
  ) {
    this.port = 8888; // hindsight-embed daemon uses same port as API
    this.baseUrl = `http://127.0.0.1:8888`;
    this.embedDir = join(homedir(), '.openclaw', 'hindsight-embed');
    this.llmProvider = llmProvider;
    this.llmApiKey = llmApiKey;
    this.llmModel = llmModel;
    this.llmBaseUrl = llmBaseUrl;
    this.daemonIdleTimeout = daemonIdleTimeout;
    this.embedVersion = embedVersion || 'latest';
  }

  async start(): Promise<void> {
    console.log(`[Hindsight] Starting hindsight-embed daemon...`);

    // Build environment variables
    const env: NodeJS.ProcessEnv = {
      ...process.env,
      HINDSIGHT_EMBED_LLM_PROVIDER: this.llmProvider,
      HINDSIGHT_EMBED_LLM_API_KEY: this.llmApiKey,
      HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT: this.daemonIdleTimeout.toString(),
    };

    if (this.llmModel) {
      env['HINDSIGHT_EMBED_LLM_MODEL'] = this.llmModel;
    }

    // Pass through base URL for OpenAI-compatible providers (OpenRouter, etc.)
    if (this.llmBaseUrl) {
      env['HINDSIGHT_API_LLM_BASE_URL'] = this.llmBaseUrl;
    }

    // On macOS, force CPU for embeddings/reranker to avoid MPS/Metal issues in daemon mode
    if (process.platform === 'darwin') {
      env['HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU'] = '1';
      env['HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU'] = '1';
    }

    // Write env vars to ~/.hindsight/config.env for daemon persistence
    await this.writeConfigEnv(env);

    // Start hindsight-embed daemon (it manages itself)
    const embedPackage = this.embedVersion ? `hindsight-embed@${this.embedVersion}` : 'hindsight-embed@latest';
    const startDaemon = spawn(
      'uvx',
      [embedPackage, 'daemon', 'start'],
      {
        env,
        stdio: 'pipe',
      }
    );

    // Collect output
    let output = '';
    startDaemon.stdout?.on('data', (data) => {
      const text = data.toString();
      output += text;
      console.log(`[Hindsight] ${text.trim()}`);
    });

    startDaemon.stderr?.on('data', (data) => {
      const text = data.toString();
      output += text;
      console.error(`[Hindsight] ${text.trim()}`);
    });

    // Wait for daemon start command to complete
    await new Promise<void>((resolve, reject) => {
      startDaemon.on('exit', (code) => {
        if (code === 0) {
          console.log('[Hindsight] Daemon start command completed');
          resolve();
        } else {
          reject(new Error(`Daemon start failed with code ${code}: ${output}`));
        }
      });

      startDaemon.on('error', (error) => {
        reject(error);
      });
    });

    // Wait for server to be ready
    await this.waitForReady();
    console.log('[Hindsight] Daemon is ready');
  }

  async stop(): Promise<void> {
    console.log('[Hindsight] Stopping hindsight-embed daemon...');

    const embedPackage = this.embedVersion ? `hindsight-embed@${this.embedVersion}` : 'hindsight-embed@latest';
    const stopDaemon = spawn('uvx', [embedPackage, 'daemon', 'stop'], {
      stdio: 'pipe',
    });

    await new Promise<void>((resolve) => {
      stopDaemon.on('exit', () => {
        console.log('[Hindsight] Daemon stopped');
        resolve();
      });

      stopDaemon.on('error', (error) => {
        console.error('[Hindsight] Error stopping daemon:', error);
        resolve(); // Resolve anyway
      });

      // Timeout after 5 seconds
      setTimeout(() => {
        console.log('[Hindsight] Daemon stop timeout');
        resolve();
      }, 5000);
    });
  }

  private async waitForReady(maxAttempts = 30): Promise<void> {
    console.log('[Hindsight] Waiting for daemon to be ready...');
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const response = await fetch(`${this.baseUrl}/health`);
        if (response.ok) {
          console.log('[Hindsight] Daemon health check passed');
          return;
        }
      } catch {
        // Not ready yet
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    throw new Error('Hindsight daemon failed to become ready within 30 seconds');
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  isRunning(): boolean {
    return this.process !== null;
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(2000) });
      return response.ok;
    } catch {
      return false;
    }
  }

  private async writeConfigEnv(env: NodeJS.ProcessEnv): Promise<void> {
    const hindsightDir = join(homedir(), '.hindsight');
    const embedConfigPath = join(hindsightDir, 'embed');

    // Ensure directory exists
    await fs.mkdir(hindsightDir, { recursive: true });

    // Read existing config to preserve extra settings
    let existingContent = '';
    let extraSettings: string[] = [];
    try {
      existingContent = await fs.readFile(embedConfigPath, 'utf-8');
      // Extract non-LLM settings (like FORCE_CPU flags)
      const lines = existingContent.split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith('#') &&
            !trimmed.startsWith('HINDSIGHT_EMBED_LLM_') &&
            !trimmed.startsWith('HINDSIGHT_API_LLM_') &&
            !trimmed.startsWith('HINDSIGHT_EMBED_BANK_ID') &&
            !trimmed.startsWith('HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT')) {
          extraSettings.push(line);
        }
      }
    } catch {
      // File doesn't exist yet, that's fine
    }

    // Build config file with header
    const configLines: string[] = [
      '# Hindsight Embed Configuration',
      '# Generated by OpenClaw Hindsight plugin',
      '',
    ];

    // Add LLM config
    if (env.HINDSIGHT_EMBED_LLM_PROVIDER) {
      configLines.push(`HINDSIGHT_EMBED_LLM_PROVIDER=${env.HINDSIGHT_EMBED_LLM_PROVIDER}`);
    }
    if (env.HINDSIGHT_EMBED_LLM_MODEL) {
      configLines.push(`HINDSIGHT_EMBED_LLM_MODEL=${env.HINDSIGHT_EMBED_LLM_MODEL}`);
    }
    if (env.HINDSIGHT_EMBED_LLM_API_KEY) {
      configLines.push(`HINDSIGHT_EMBED_LLM_API_KEY=${env.HINDSIGHT_EMBED_LLM_API_KEY}`);
    }
    if (env.HINDSIGHT_API_LLM_BASE_URL) {
      configLines.push(`HINDSIGHT_API_LLM_BASE_URL=${env.HINDSIGHT_API_LLM_BASE_URL}`);
    }
    if (env.HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT) {
      configLines.push(`HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=${env.HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT}`);
    }

    // Add platform-specific config (macOS FORCE_CPU flags)
    if (env.HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU) {
      configLines.push(`HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU=${env.HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU}`);
    }
    if (env.HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU) {
      configLines.push(`HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU=${env.HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU}`);
    }

    // Add extra settings if they exist
    if (extraSettings.length > 0) {
      configLines.push('');
      configLines.push('# Additional settings');
      configLines.push(...extraSettings);
    }

    // Write to file
    await fs.writeFile(embedConfigPath, configLines.join('\n') + '\n', 'utf-8');
    console.log(`[Hindsight] Wrote config to ${embedConfigPath}`);
  }
}
