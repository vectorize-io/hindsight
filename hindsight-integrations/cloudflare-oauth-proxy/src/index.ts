import { WorkerEntrypoint } from "cloudflare:workers";
import OAuthProvider from "@cloudflare/workers-oauth-provider";

import { handleDefaultRequest } from "./auth";
import type { Env } from "./env";
import { proxyRequest } from "./proxy";
import { createWorker } from "./router";

export type { Env, OAuthHelpers, OAuthReqInfo } from "./env";

/**
 * MCP reverse-proxy. The OAuth provider routes authenticated requests for
 * `/mcp` into this class; everything else goes to `defaultHandler` below.
 */
export class HindsightProxy extends WorkerEntrypoint<Env> {
  async fetch(request: Request): Promise<Response> {
    return proxyRequest(request, this.env);
  }
}

// The OAuth provider library types `defaultHandler` as `ExportedHandler`
// (unparameterised), so we cast through `unknown` rather than try to make the
// generic parameter line up with `ExportedHandler<unknown>` invariantly.
const defaultHandler = {
  async fetch(request: Request, env: Env): Promise<Response> {
    return handleDefaultRequest(request, env);
  },
} as unknown as ExportedHandler;

const provider = new OAuthProvider({
  apiRoute: "/mcp",
  apiHandler: HindsightProxy,
  defaultHandler,
  authorizeEndpoint: "/authorize",
  tokenEndpoint: "/token",
  clientRegistrationEndpoint: "/register",
});

export default createWorker(provider);
