"use client";

import { api } from "@/lib/api";
import { useAsync, WorkspaceBar } from "@/lib/ui";

export default function Dashboard() {
  const me = useAsync(() => api.me(), []);
  return (
    <div>
      <h2>Operator Dashboard</h2>
      <WorkspaceBar />
      <div className="card">
        <h3>Identity</h3>
        {me.error && <p className="error">{me.error}</p>}
        {me.data ? (
          <p>
            Signed in as <code>{me.data.user?.email}</code>{" "}
            {me.data.user?.is_operator ? "(operator)" : ""} · auth:{" "}
            <code>{me.data.auth_method}</code>
          </p>
        ) : (
          <p className="muted">Loading… (is the Central API running at {api.base}?)</p>
        )}
      </div>
      <div className="card">
        <h3>Governed flow (v0.1)</h3>
        <p className="muted">
          Connect Google Drive → discover file metadata → snapshot permissions → queue ingestion →
          process jobs → inspect audit. All access is permission-checked and fails closed. No chat
          over Drive yet.
        </p>
      </div>
    </div>
  );
}
