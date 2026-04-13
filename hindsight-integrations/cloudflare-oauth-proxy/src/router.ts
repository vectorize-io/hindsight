/**
 * Outer worker fetch handler. Wraps the OAuth provider to restrict CORS,
 * intercept OAuth metadata for PKCE hardening, and enforce an origin
 * allowlist on preflight requests.
 *
 * This is a factory so tests can inject a mock provider without depending
 * on `@cloudflare/workers-oauth-provider` or the Workers runtime.
 */

import { applyCors, preflightResponse } from "./cors";
import type { Env } from "./env";

export interface ProviderLike {
  fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response>;
}

export interface WorkerRouter {
  fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response>;
}

export function createWorker(provider: ProviderLike): WorkerRouter {
  return {
    async fetch(request, env, ctx) {
      const origin = request.headers.get("Origin");
      const url = new URL(request.url);

      if (request.method === "OPTIONS") {
        return preflightResponse(origin);
      }

      if (url.pathname === "/.well-known/oauth-authorization-server") {
        const upstream = await provider.fetch(request, env, ctx);
        const metadata = (await upstream.json()) as Record<string, unknown>;
        metadata.code_challenge_methods_supported = ["S256"];
        const response = new Response(JSON.stringify(metadata), {
          headers: { "Content-Type": "application/json" },
        });
        return applyCors(response, origin);
      }

      const response = await provider.fetch(request, env, ctx);
      return applyCors(response, origin);
    },
  };
}
