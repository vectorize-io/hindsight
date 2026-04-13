/**
 * Default request handler: /health, /, and the GET/POST /authorize flow.
 */

import type { Env, OAuthReqInfo } from "./env";
import { loginPage } from "./html";

const AUTH_STATE_PREFIX = "auth_state:";
const AUTH_STATE_TTL_SECONDS = 300;

/**
 * Constant-time password comparison. Both inputs are hashed with SHA-256 first
 * so the comparison length is fixed regardless of input length.
 */
export async function verifyPassword(
  provided: string,
  expected: string,
): Promise<boolean> {
  const encoder = new TextEncoder();
  const [a, b] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(provided)),
    crypto.subtle.digest("SHA-256", encoder.encode(expected)),
  ]);
  const viewA = new Uint8Array(a);
  const viewB = new Uint8Array(b);
  let diff = 0;
  for (let i = 0; i < viewA.length; i++) {
    diff |= viewA[i] ^ viewB[i];
  }
  return diff === 0;
}

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
  });
}

function htmlResponse(body: string, init: ResponseInit = {}): Response {
  return new Response(body, {
    ...init,
    headers: { "Content-Type": "text/html", ...(init.headers ?? {}) },
  });
}

async function handleAuthorizeGet(request: Request, env: Env): Promise<Response> {
  const oauthReqInfo = await env.OAUTH_PROVIDER.parseAuthRequest(request);
  const stateKey = crypto.randomUUID();
  await env.OAUTH_KV.put(
    AUTH_STATE_PREFIX + stateKey,
    JSON.stringify(oauthReqInfo),
    { expirationTtl: AUTH_STATE_TTL_SECONDS },
  );
  return htmlResponse(loginPage(stateKey));
}

async function handleAuthorizePost(request: Request, env: Env): Promise<Response> {
  const formData = await request.formData();
  const passwordField = formData.get("password");
  const stateKeyField = formData.get("stateKey");

  const password = typeof passwordField === "string" ? passwordField : "";
  const stateKey = typeof stateKeyField === "string" ? stateKeyField : "";

  if (!password || !(await verifyPassword(password, env.SESSION_SECRET))) {
    return htmlResponse(loginPage(stateKey, "Incorrect password."), { status: 401 });
  }

  if (!stateKey) {
    return new Response("Missing state. Please try connecting again from Claude.", {
      status: 400,
    });
  }

  const stored = await env.OAUTH_KV.get(AUTH_STATE_PREFIX + stateKey);
  await env.OAUTH_KV.delete(AUTH_STATE_PREFIX + stateKey);

  if (!stored) {
    return new Response(
      "Authorization expired. Please try connecting again from Claude.",
      { status: 400 },
    );
  }

  const oauthReqInfo = JSON.parse(stored) as OAuthReqInfo;

  try {
    const { redirectTo } = await env.OAUTH_PROVIDER.completeAuthorization({
      request: oauthReqInfo,
      userId: env.ALLOWED_EMAIL,
      metadata: { label: env.ALLOWED_EMAIL },
      scope: oauthReqInfo.scope ?? "mcp:full",
      props: {
        email: env.ALLOWED_EMAIL,
        authenticatedAt: Date.now(),
      },
    });
    return Response.redirect(redirectTo, 302);
  } catch (err) {
    console.error("completeAuthorization error:", err);
    return new Response("Authorization failed. Please try again.", { status: 500 });
  }
}

export async function handleDefaultRequest(
  request: Request,
  env: Env,
): Promise<Response> {
  const url = new URL(request.url);

  if (url.pathname === "/health") {
    return jsonResponse({ status: "ok" });
  }

  if (url.pathname === "/authorize" && request.method === "GET") {
    return handleAuthorizeGet(request, env);
  }

  if (url.pathname === "/authorize" && request.method === "POST") {
    return handleAuthorizePost(request, env);
  }

  if (url.pathname === "/") {
    return jsonResponse({ service: "Hindsight MCP OAuth Proxy" });
  }

  return new Response("Not Found", { status: 404 });
}
