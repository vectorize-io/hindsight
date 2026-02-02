// Type definitions for moltbot plugin SDK
// These are minimal types based on the documentation

declare module 'moltbot/plugin-sdk' {
  export interface HookEvent {
    type: 'command' | 'session' | 'agent' | 'gateway' | 'tool_result_persist';
    action?: string;
    sessionKey?: string;
    timestamp?: string;
    messages?: string[];
    context?: {
      sessionEntry?: {
        messages?: Array<{
          role: string;
          content: string;
        }>;
      };
      sessionId?: string;
      sessionKey?: string;
      sessionFile?: string;
      commandSource?: string;
      senderId?: string;
      workspaceDir?: string;
      bootstrapFiles?: string[];
      cfg?: any;
    };
  }

  export type HookHandler = (event: HookEvent) => Promise<void>;

  export function registerPluginHooksFromDir(api: any, dir: string): void;
}
