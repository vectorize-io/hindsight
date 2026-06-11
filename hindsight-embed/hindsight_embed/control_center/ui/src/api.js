// Token + fetch wrapper + deep-link params. The token comes from the launch URL
// once, then lives in localStorage (same-origin only) so the bare URL works
// afterward. It's sent as a custom header (the CSRF defense), never a cookie.
const TOKEN_KEY = "hs_control_token";
const params = new URLSearchParams(location.search);
// Strip whitespace a copy-paste may have injected into the token.
const urlToken = (params.get("token") || "").replace(/\s/g, "");

let TOKEN = urlToken || localStorage.getItem(TOKEN_KEY) || "";
export const profileParam = params.get("profile");
export const tabParam = params.get("tab");

if (params.has("token")) {
  localStorage.setItem(TOKEN_KEY, urlToken);
  params.delete("token"); // keep ?profile/?tab; drop only the secret
  history.replaceState(null, "", location.pathname + (params.toString() ? "?" + params : ""));
}

export function forgetToken() {
  localStorage.removeItem(TOKEN_KEY);
  TOKEN = "";
}

export async function api(method, path, body) {
  const resp = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", "X-Hindsight-Control-Token": TOKEN },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (resp.status === 401) {
    const err = new Error("unauthorized");
    err.unauthorized = true;
    throw err;
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || "HTTP " + resp.status);
  return data;
}

// Reflect the selected profile in the address bar (shareable; desktop deep-links here).
export function syncProfileUrl(name) {
  const sp = new URLSearchParams(location.search);
  sp.set("profile", name || "default");
  history.replaceState(null, "", location.pathname + "?" + sp);
}

export const pn = (name) => encodeURIComponent(name || "default");
