/**
 * Per-agent HindsightClient registry.
 *
 * OpenCode plugins are loaded once per server process, but multiple agents
 * (build, code-reviewer, custom agents, …) can drive sessions within that
 * process. When `hindsightApiTokens` maps agent names to distinct API tokens,
 * each agent must talk to Hindsight with its own `HindsightClient` instance.
 *
 * The registry lazily constructs and caches one `HindsightClient` per resolved
 * token (so agents that share a token reuse a single client). A default client
 * — built eagerly from the static `hindsightApiToken` / `agentName` fallback —
 * is preserved so that legacy single-token setups (and the plugin tests that
 * assert eager construction) keep working unchanged.
 */

import { HindsightClient } from "@vectorize-io/hindsight-client";
import type { HindsightConfig } from "./config.js";
import { resolveApiKey } from "./config.js";
import { Logger } from "./logger.js";

/** Anything that can resolve a `HindsightClient` for a given agent name. */
export interface ClientResolver {
  forAgent(agentName?: string | null): HindsightClient;
}

/**
 * Normalize a `HindsightClient | ClientResolver` into a `ClientResolver`.
 * A bare `HindsightClient` (e.g. a mock in unit tests) is wrapped so that
 * `forAgent()` always returns the same instance, regardless of agent — this
 * keeps the existing call sites and tests unchanged.
 */
export function toResolver(
  clientOrResolver: HindsightClient | ClientResolver
): ClientResolver {
  const maybe = clientOrResolver as Partial<ClientResolver>;
  if (typeof maybe.forAgent === "function") {
    return clientOrResolver as ClientResolver;
  }
  const client = clientOrResolver as HindsightClient;
  return { forAgent: () => client };
}

export interface ClientRegistryOptions {
  baseUrl: string;
  config: HindsightConfig;
  /** Eagerly-constructed default client (constructed from the fallback token). */
  defaultClient: HindsightClient;
  logger?: Logger;
  /** Injectable for tests; defaults to the real constructor. */
  clientFactory?: (options: { baseUrl: string; apiKey?: string }) => HindsightClient;
}

export class ClientRegistry implements ClientResolver {
  private readonly baseUrl: string;
  private readonly config: HindsightConfig;
  private readonly defaultClient: HindsightClient;
  private readonly logger: Logger;
  private readonly clientFactory: (options: {
    baseUrl: string;
    apiKey?: string;
  }) => HindsightClient;
  /** Cache by resolved token so agents sharing a token share one client. */
  private readonly clientsByToken = new Map<string, HindsightClient>();

  constructor(options: ClientRegistryOptions) {
    this.baseUrl = options.baseUrl;
    this.config = options.config;
    this.defaultClient = options.defaultClient;
    this.logger = options.logger ?? new Logger({ silent: true });
    this.clientFactory =
      options.clientFactory ??
      ((opts) => new HindsightClient({ baseUrl: opts.baseUrl, apiKey: opts.apiKey }));
  }

  /**
   * Return the `HindsightClient` to use for `agentName`. When the resolved
   * token equals the default client's token, the pre-built default client is
   * returned (reference-stable), avoiding extra construction for the common
   * single-token case.
   */
  forAgent(agentName?: string | null): HindsightClient {
    const token = resolveApiKey(this.config, agentName) ?? undefined;
    const defaultToken = resolveApiKey(this.config) ?? undefined;

    if (token === defaultToken) {
      return this.defaultClient;
    }

    const cacheKey = token ?? "__no_token__";
    const existing = this.clientsByToken.get(cacheKey);
    if (existing) return existing;

    const client = this.clientFactory({ baseUrl: this.baseUrl, apiKey: token });
    this.clientsByToken.set(cacheKey, client);
    this.logger.info("Hindsight client created for agent", {
      agent: agentName ?? "(unknown)",
      authenticated: Boolean(token),
    });
    return client;
  }
}