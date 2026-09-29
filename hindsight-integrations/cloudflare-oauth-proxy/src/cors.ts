/**
 * CORS helpers. The proxy only advertises CORS for an allowlist of origins;
 * responses for other origins are stripped of any CORS headers the underlying
 * OAuth provider library may have added.
 */

export const ALLOWED_ORIGINS: ReadonlySet<string> = new Set([
  "https://claude.ai",
  "https://www.claude.ai",
]);

const ALLOWED_METHODS = "GET, POST, OPTIONS";
const ALLOWED_HEADERS = "Authorization, Content-Type, Mcp-Session-Id";

export function corsHeaders(origin: string): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": ALLOWED_METHODS,
    "Access-Control-Allow-Headers": ALLOWED_HEADERS,
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

export function stripCorsHeaders(response: Response): Response {
  const cleaned = new Response(response.body, response);
  cleaned.headers.delete("Access-Control-Allow-Origin");
  cleaned.headers.delete("Access-Control-Allow-Methods");
  cleaned.headers.delete("Access-Control-Allow-Headers");
  cleaned.headers.delete("Access-Control-Max-Age");
  cleaned.headers.delete("Access-Control-Allow-Credentials");
  cleaned.headers.delete("Access-Control-Expose-Headers");
  return cleaned;
}

export function applyCors(response: Response, origin: string | null): Response {
  if (!origin || !ALLOWED_ORIGINS.has(origin)) {
    if (response.headers.has("Access-Control-Allow-Origin")) {
      return stripCorsHeaders(response);
    }
    return response;
  }
  const patched = stripCorsHeaders(response);
  for (const [key, value] of Object.entries(corsHeaders(origin))) {
    patched.headers.set(key, value);
  }
  return patched;
}

export function preflightResponse(origin: string | null): Response {
  if (origin && ALLOWED_ORIGINS.has(origin)) {
    return new Response(null, {
      status: 204,
      headers: { "Content-Length": "0", ...corsHeaders(origin) },
    });
  }
  return new Response(null, { status: 403 });
}
