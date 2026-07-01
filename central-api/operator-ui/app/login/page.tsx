"use client";

import { api } from "@/lib/api";
import { useAsync } from "@/lib/ui";

// v0.1 placeholder auth: the Central API uses a dev identity in development and
// will front Authentik (OIDC) later. This page just surfaces who the API
// currently resolves the caller as — no credential handling in the GUI.
export default function Login() {
  const me = useAsync(() => api.me(), []);
  return (
    <div>
      <h2>Login</h2>
      <div className="banner">
        v0.1 uses the Central API&apos;s dev identity. Production will redirect to Authentik (OIDC).
        The GUI never stores credentials or tokens.
      </div>
      <div className="card">
        {me.error && <p className="error">Not authenticated: {me.error}</p>}
        {me.data && (
          <p>
            Authenticated as <code>{me.data.user?.email}</code> via{" "}
            <code>{me.data.auth_method}</code>.
          </p>
        )}
      </div>
    </div>
  );
}
