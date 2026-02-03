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
  private embedPackagePath?: string;

  constructor(
    port: number,
    llmProvider: string,
    llmApiKey: string,
    llmModel?: string,
    llmBaseUrl?: string,
    daemonIdleTimeout: number = 0, // Default: never timeout
    embedVersion: string = 'latest', // Default: latest
    embedPackagePath?: string // Local path to hindsight package
  ) {
    // Use the configured port (default: 9077 from config)
    this.port = port;
    this.baseUrl = `http://127.0.0.1:${port}`;
    this.embedDir = join(homedir(), '.openclaw', 'hindsight-embed');
    this.llmProvider = llmProvider;
    this.llmApiKey = llmApiKey;
    this.llmModel = llmModel;
    this.llmBaseUrl = llmBaseUrl;
    this.daemonIdleTimeout = daemonIdleTimeout;
    this.embedVersion = embedVersion || 'latest';
    this.embedPackagePath = embedPackagePath;
  }

  /**
   * Get the command to run hindsight-embed (either local or from PyPI)
   */
  private getEmbedCommand(): string[] {
    if (this.embedPackagePath) {
      // Local package: uv run --directory <path> hindsight-embed
      return ['uv', 'run', '--directory', this.embedPackagePath, 'hindsight-embed'];
    } else {
      // PyPI package: uvx hindsight-embed@version
      const embedPackage = this.embedVersion ? `hindsight-embed@${this.embedVersion}` : 'hindsight-embed@latest';
      return ['uvx', embedPackage];
    }
  }

  async start(): Promise<void> {
    console.log(`[Hindsight] Starting hindsight-embed daemon...`);

    // Build environment variables using standard HINDSIGHT_API_LLM_* variables
    const env: NodeJS.ProcessEnv = {
      ...process.env,
      HINDSIGHT_API_LLM_PROVIDER: this.llmProvider,
      HINDSIGHT_API_LLM_API_KEY: this.llmApiKey,
      HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT: this.daemonIdleTimeout.toString(),
    };

    if (this.llmModel) {
      env['HINDSIGHT_API_LLM_MODEL'] = this.llmModel;
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

    // Configure "openclaw" profile using hindsight-embed configure (non-interactive)
    console.log('[Hindsight] Configuring "openclaw" profile...');
    await this.configureProfile(env);

    // Start hindsight-embed daemon with openclaw profile
    const embedCmd = this.getEmbedCommand();
    const startDaemon = spawn(
      embedCmd[0],
      [...embedCmd.slice(1), 'daemon', '--profile', 'openclaw', 'start'],
      {
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

    const embedCmd = this.getEmbedCommand();
    const stopDaemon = spawn(embedCmd[0], [...embedCmd.slice(1), 'daemon', '--profile', 'openclaw', 'stop'], {
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

  private async configureProfile(env: NodeJS.ProcessEnv): Promise<void> {
    // Build profile create command args with --merge, --port and --env flags
    // Use --merge to allow updating existing profile
    const createArgs = ['profile', 'create', 'openclaw', '--merge', '--port', this.port.toString()];

    // Add all environment variables as --env flags
    const envVars = [
      'HINDSIGHT_API_LLM_PROVIDER',
      'HINDSIGHT_API_LLM_MODEL',
      'HINDSIGHT_API_LLM_API_KEY',
      'HINDSIGHT_API_LLM_BASE_URL',
      'HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT',
      'HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU',
      'HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU',
    ];

    for (const envVar of envVars) {
      if (env[envVar]) {
        createArgs.push('--env', `${envVar}=${env[envVar]}`);
      }
    }

    // Run profile create command (non-interactive, overwrites if exists)
    const embedCmd = this.getEmbedCommand();
    const create = spawn(embedCmd[0], [...embedCmd.slice(1), ...createArgs], {
      stdio: 'pipe',
    });

    let output = '';
    create.stdout?.on('data', (data) => {
      const text = data.toString();
      output += text;
      console.log(`[Hindsight] ${text.trim()}`);
    });

    create.stderr?.on('data', (data) => {
      const text = data.toString();
      output += text;
      console.error(`[Hindsight] ${text.trim()}`);
    });

    await new Promise<void>((resolve, reject) => {
      create.on('exit', (code) => {
        if (code === 0) {
          console.log('[Hindsight] Profile "openclaw" configured successfully');
          resolve();
        } else {
          reject(new Error(`Profile create failed with code ${code}: ${output}`));
        }
      });

      create.on('error', (error) => {
        reject(error);
      });
    });
  }
}
