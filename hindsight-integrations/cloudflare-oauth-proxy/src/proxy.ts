/**
 * MCP reverse-proxy handler. Takes an authenticated request, rewrites the
 * target URL, swaps the `Authorization` header for the server's bearer token,
 * and forwards to the Hindsight origin.
 */

import type { Env } from "./env";

type ProxyEnv = Pick<Env, "HINDSIGHT_ORIGIN" | "PROXY_SECRET" | "HINDSIGHT_API_TOKEN">;

/**
 * Hop-by-hop headers that must not be forwarded (RFC 7230 §6.1) plus the
 * client's `Authorization` and any attempt to forge `X-Proxy-Secret`.
 */
const STRIPPED_REQUEST_HEADERS: ReadonlySet<string> = new Set([
  "authorization",
  "x-proxy-secret",
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
]);

/**
 * Whitelist of response headers we're willing to forward back to the client.
 * Anything the upstream sets beyond this (cookies, its own CORS headers, etc.)
 * is dropped so it can't leak through the proxy.
 */
const FORWARDED_RESPONSE_HEADERS: ReadonlySet<string> = new Set([
  "content-type",
  "content-length",
  "content-encoding",
  "cache-control",
  "etag",
  "last-modified",
  "expires",
  "mcp-session-id",
  "retry-after",
]);

function sanitizeRequestHeaders(headers: Headers): Headers {
  const out = new Headers();
  for (const [key, value] of headers.entries()) {
    if (!STRIPPED_REQUEST_HEADERS.has(key.toLowerCase())) {
      out.set(key, value);
    }
  }
  return out;
}

function sanitizeUpstreamResponse(response: Response): Response {
  const headers = new Headers();
  for (const [key, value] of response.headers.entries()) {
    if (FORWARDED_RESPONSE_HEADERS.has(key.toLowerCase())) {
      headers.set(key, value);
    }
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

export interface ProxyOptions {
  /** Injected for tests; defaults to the global `fetch`. */
  fetchImpl?: typeof fetch;
}

export async function proxyRequest(
  request: Request,
  env: ProxyEnv,
  options: ProxyOptions = {}
): Promise<Response> {
  const fetchImpl = options.fetchImpl ?? fetch;

  const url = new URL(request.url);
  const originUrl = new URL(env.HINDSIGHT_ORIGIN);
  url.hostname = originUrl.hostname;
  url.port = originUrl.port;
  url.protocol = originUrl.protocol;

  const headers = sanitizeRequestHeaders(request.headers);
  headers.set("X-Proxy-Secret", env.PROXY_SECRET);
  headers.set("Authorization", `Bearer ${env.HINDSIGHT_API_TOKEN}`);

  // Buffer the body to avoid needing `duplex: "half"` when forwarding a
  // ReadableStream. MCP JSON-RPC payloads are small, so this is fine.
  const hasBody = request.method !== "GET" && request.method !== "HEAD";
  const body = hasBody ? await request.arrayBuffer() : null;

  try {
    const upstream = await fetchImpl(url.toString(), {
      method: request.method,
      headers,
      body,
    });
    return sanitizeUpstreamResponse(upstream);
  } catch (err) {
    console.error("Failed to proxy to Hindsight:", err);
    return new Response(JSON.stringify({ error: "Backend unavailable" }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }
}
