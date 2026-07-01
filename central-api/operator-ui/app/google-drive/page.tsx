"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { Status, useAsync, WorkspaceBar } from "@/lib/ui";
import { useWorkspace } from "@/lib/useWorkspace";

export default function GoogleDrive() {
  const [wid] = useWorkspace();
  const [folders, setFolders] = useState("");
  const [email, setEmail] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const status = useAsync(
    () => (wid ? api.gdriveStatus(wid) : Promise.resolve({ status: "—" })),
    [wid],
  );
  const cfg = useAsync(() => api.gdriveOauthConfig(), []);

  const act = async (fn: () => Promise<any>) => {
    setMsg(null);
    try {
      const r = await fn();
      setMsg(JSON.stringify(r));
      status.reload();
    } catch (e: any) {
      setMsg(`Error: ${e?.message ?? e}`);
    }
  };

  const connect = () =>
    act(() =>
      api.gdriveConnect({
        workspace_id: wid,
        folder_ids: folders.split(",").map((s) => s.trim()).filter(Boolean),
        account_email: email || undefined,
      }),
    );

  return (
    <div>
      <h2>Google Drive Connection</h2>
      <WorkspaceBar />
      {!wid && <p className="muted">Select a workspace first.</p>}

      <div className="card">
        <h3>OAuth (read-only)</h3>
        {cfg.data && (
          <p className="muted">
            Configured: {String(cfg.data.configured)} · scopes:{" "}
            {(cfg.data.scopes ?? []).map((s: string) => (
              <code key={s} style={{ marginRight: 6 }}>{s.split("/auth/")[1]}</code>
            ))}
          </p>
        )}
        <p className="muted">Read-only metadata + content scopes only. No write/delete scopes.</p>
      </div>

      <div className="card">
        <h3>Status: {status.data ? <Status value={status.data.status} /> : "…"}</h3>
        <div className="row">
          <input placeholder="Folder IDs (comma-separated)" value={folders}
                 onChange={(e) => setFolders(e.target.value)} style={{ minWidth: 280 }} />
          <input placeholder="account email (optional)" value={email}
                 onChange={(e) => setEmail(e.target.value)} />
          <button disabled={!wid} onClick={connect}>Connect</button>
          <button className="secondary" disabled={!wid} onClick={() => act(() => api.gdriveSync(wid))}>
            Manual sync
          </button>
          <button className="danger" disabled={!wid} onClick={() => act(() => api.gdriveDisconnect(wid))}>
            Disconnect
          </button>
        </div>
        {msg && <p className="muted" style={{ marginTop: 12 }}><code>{msg}</code></p>}
      </div>
    </div>
  );
}
