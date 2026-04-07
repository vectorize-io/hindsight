import { WorkerEntrypoint } from "cloudflare:workers";
import OAuthProvider from "@cloudflare/workers-oauth-provider";

interface Env {
  OAUTH_KV: KVNamespace;
  OAUTH_PROVIDER: any;
  HINDSIGHT_ORIGIN: string;
  ALLOWED_EMAIL: string;
  SESSION_SECRET: string;
  PROXY_SECRET: string;        // add as Wrangler secret
  HINDSIGHT_API_TOKEN: string;  // add as Wrangler secret
}

// --- HTML escaping ---
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// --- CORS Allowlist ---
const ALLOWED_ORIGINS = new Set([
  "https://claude.ai",
  "https://www.claude.ai",
]);

function corsHeaders(origin: string): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "*",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}

function stripCorsHeaders(response: Response): Response {
  const cleaned = new Response(response.body, response);
  cleaned.headers.delete("Access-Control-Allow-Origin");
  cleaned.headers.delete("Access-Control-Allow-Methods");
  cleaned.headers.delete("Access-Control-Allow-Headers");
  cleaned.headers.delete("Access-Control-Max-Age");
  return cleaned;
}

function applyCors(response: Response, origin: string | null): Response {
  if (!origin || !ALLOWED_ORIGINS.has(origin)) {
    // Strip any CORS headers the library may have added
    if (response.headers.has("Access-Control-Allow-Origin")) {
      return stripCorsHeaders(response);
    }
    return response;
  }
  const patched = stripCorsHeaders(response);
  for (const [k, v] of Object.entries(corsHeaders(origin))) {
    patched.headers.set(k, v);
  }
  return patched;
}

// --- MCP Proxy Handler ---
export class HindsightProxy extends WorkerEntrypoint<Env> {
  async fetch(request: Request): Promise<Response> {
    const props = (this.ctx as any).props || {};
    console.log(`MCP request from ${props.email}: ${request.method} ${new URL(request.url).pathname}`);

    const url = new URL(request.url);
    const originUrl = new URL(this.env.HINDSIGHT_ORIGIN);
    url.hostname = originUrl.hostname;
    url.port = originUrl.port;
    url.protocol = originUrl.protocol;

    const headers = new Headers(request.headers);
    headers.delete("Authorization");
    headers.set("X-Proxy-Secret", this.env.PROXY_SECRET);
    headers.set("Authorization", `Bearer ${this.env.HINDSIGHT_API_TOKEN}`);

    try {
      const response = await fetch(url.toString(), {
        method: request.method,
        headers,
        body: request.body ?? null,
      });
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    } catch (err) {
      console.error("Failed to proxy to Hindsight:", err);
      return new Response(JSON.stringify({ error: "Backend unavailable" }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }
  }
}

function loginPage(stateKey: string, error?: string): string {
  return '<!DOCTYPE html>' +
    '<html><head><title>Hindsight MCP</title>' +
    '<style>' +
    'body{font-family:system-ui;background:#0a0a0a;color:#e0e0e0;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}' +
    '.card{background:#1a1a1a;border:1px solid #333;border-radius:12px;padding:2rem;width:320px}' +
    'h2{margin-top:0}' +
    'input{width:100%;padding:10px;margin:8px 0;border:1px solid #444;border-radius:6px;background:#0a0a0a;color:#e0e0e0;box-sizing:border-box;font-size:16px}' +
    'button{width:100%;padding:10px;margin-top:12px;border:none;border-radius:6px;background:#3b82f6;color:white;font-size:16px;cursor:pointer}' +
    'button:hover{background:#2563eb}' +
    '.error{color:#ef4444;font-size:14px}' +
    '.info{color:#888;font-size:13px;margin-top:12px}' +
    '</style></head><body>' +
    '<div class="card">' +
    '<h2>Hindsight MCP</h2>' +
    '<p>Authorize Claude to access your memory.</p>' +
    (error ? '<p class="error">' + escapeHtml(error) + '</p>' : '') +
    '<form method="POST" action="/authorize">' +
    '<input type="hidden" name="stateKey" value="' + escapeHtml(stateKey) + '" />' +
    '<input type="password" name="password" placeholder="Password" autofocus required />' +
    '<button type="submit">Authorize</button>' +
    '</form>' +
    '<p class="info">You only need to do this once per session.</p>' +
    '</div></body></html>';
}

const defaultHandler = {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return new Response(JSON.stringify({ status: "ok" }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/authorize" && request.method === "GET") {
      const oauthReqInfo = await env.OAUTH_PROVIDER.parseAuthRequest(request);
      console.log("GET /authorize - parsed oauthReqInfo, clientId:", oauthReqInfo.clientId);

      const stateKey = crypto.randomUUID();
      await env.OAUTH_KV.put("auth_state:" + stateKey, JSON.stringify(oauthReqInfo), { expirationTtl: 300 });

      return new Response(loginPage(stateKey), {
        headers: { "Content-Type": "text/html" },
      });
    }

    if (url.pathname === "/authorize" && request.method === "POST") {
      const formData = await request.formData();
      const password = formData.get("password") as string;
      const stateKey = formData.get("stateKey") as string;

      if (!password || password !== env.SESSION_SECRET) {
        return new Response(loginPage(stateKey || "", "Incorrect password."), {
          status: 401,
          headers: { "Content-Type": "text/html" },
        });
      }

      if (!stateKey) {
        return new Response("Missing state. Please try connecting again from Claude.", { status: 400 });
      }

      const stored = await env.OAUTH_KV.get("auth_state:" + stateKey);
      await env.OAUTH_KV.delete("auth_state:" + stateKey);

      if (!stored) {
        return new Response("Authorization expired. Please try connecting again from Claude.", { status: 400 });
      }

      const oauthReqInfo = JSON.parse(stored);
      console.log("POST /authorize - completing authorization for client:", oauthReqInfo.clientId);

      try {
        const { redirectTo } = await env.OAUTH_PROVIDER.completeAuthorization({
          request: oauthReqInfo,
          userId: env.ALLOWED_EMAIL,
          metadata: { label: env.ALLOWED_EMAIL },
          scope: oauthReqInfo.scope ? oauthReqInfo.scope : ["mcp:full"],
          props: {
            email: env.ALLOWED_EMAIL,
            authenticatedAt: Date.now(),
          },
        });

        console.log("Authorization complete, redirecting to Claude");
        return Response.redirect(redirectTo, 302);
      } catch (err) {
        console.error("completeAuthorization error:", err);
        return new Response("Authorization failed. Please try again.", { status: 500 });
      }
    }

    if (url.pathname === "/") {
      return new Response(JSON.stringify({ service: "Hindsight MCP OAuth Proxy" }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response("Not Found", { status: 404 });
  },
};

// --- Inner provider (not exported directly) ---
const provider = new OAuthProvider({
  apiRoute: "/mcp",
  apiHandler: HindsightProxy,
  defaultHandler: defaultHandler,
  authorizeEndpoint: "/authorize",
  tokenEndpoint: "/token",
  clientRegistrationEndpoint: "/register",
});

// --- Wrapped export: restricts CORS + hardens metadata ---
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const origin = request.headers.get("Origin");
    const url = new URL(request.url);

    // Intercept OPTIONS: only allow preflight for allowlisted origins
    if (request.method === "OPTIONS") {
      if (origin && ALLOWED_ORIGINS.has(origin)) {
        return new Response(null, {
          status: 204,
          headers: { "Content-Length": "0", ...corsHeaders(origin) },
        });
      }
      return new Response(null, { status: 403 });
    }

    // Override metadata to advertise S256-only PKCE
    if (url.pathname === "/.well-known/oauth-authorization-server") {
      const response = await provider.fetch(request, env, ctx);
      const metadata = await response.json() as Record<string, unknown>;
      metadata.code_challenge_methods_supported = ["S256"];
      const newResponse = new Response(JSON.stringify(metadata), {
        headers: { "Content-Type": "application/json" },
      });
      return applyCors(newResponse, origin);
    }

    // All other requests: pass to provider, then fix CORS
    const response = await provider.fetch(request, env, ctx);
    return applyCors(response, origin);
  },
};
