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
  private daemonIdleTimeout: number;
  private embedVersion: string;

  constructor(
    port: number,
    llmProvider: string,
    llmApiKey: string,
    llmModel?: string,
    daemonIdleTimeout: number = 0, // Default: never timeout
    embedVersion: string = 'latest' // Default: latest
  ) {
    this.port = 8889; // hindsight-embed uses fixed port 8889
    this.baseUrl = `http://127.0.0.1:8889`;
    this.embedDir = join(homedir(), '.openclaw', 'hindsight-embed');
    this.llmProvider = llmProvider;
    this.llmApiKey = llmApiKey;
    this.llmModel = llmModel;
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
}
