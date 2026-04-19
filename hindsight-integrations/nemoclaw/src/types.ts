export interface CliArgs {
  sandbox: string;
  apiUrl: string;
  apiToken: string;
  bankPrefix: string;
  skipPolicy: boolean;
  skipPluginInstall: boolean;
  dryRun: boolean;
}

export interface PolicyEndpointRule {
  allow: {
    method: string;
    path: string;
  };
}

export interface PolicyEndpoint {
  host: string;
  port: number;
  protocol?: string;
  tls?: string;
  enforcement?: string;
  access?: string;
  rules?: PolicyEndpointRule[];
}

export interface PolicyBinary {
  path: string;
}

export interface NetworkPolicy {
  name: string;
  endpoints: PolicyEndpoint[];
  binaries?: PolicyBinary[];
}

export interface FilesystemPolicy {
  include_workdir?: boolean;
  read_only?: string[];
  read_write?: string[];
}

export interface Landlock {
  compatibility?: string;
}

export interface ProcessPolicy {
  run_as_user?: string;
  run_as_group?: string;
}

export interface SandboxPolicy {
  version?: number;
  filesystem_policy?: FilesystemPolicy;
  landlock?: Landlock;
  process?: ProcessPolicy;
  network_policies?: Record<string, NetworkPolicy>;
}

export const HINDSIGHT_POLICY_NAME = "hindsight";
export const HINDSIGHT_HOST = "api.hindsight.vectorize.io";
export const OPENCLAW_BINARY = "/usr/local/bin/openclaw";
