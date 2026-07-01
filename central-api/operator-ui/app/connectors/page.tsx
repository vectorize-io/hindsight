"use client";

import { api } from "@/lib/api";
import { Status, useAsync, WorkspaceBar } from "@/lib/ui";
import { useWorkspace } from "@/lib/useWorkspace";

export default function Connectors() {
  const [wid] = useWorkspace();
  const list = useAsync(() => (wid ? api.listConnectors(wid) : Promise.resolve({ connectors: [] })), [wid]);

  return (
    <div>
      <h2>Connectors</h2>
      <WorkspaceBar />
      {!wid && <p className="muted">Select a workspace first.</p>}
      {list.error && <p className="error">{list.error}</p>}
      <table>
        <thead>
          <tr><th>Provider</th><th>Status</th><th>Account</th><th>ID</th></tr>
        </thead>
        <tbody>
          {(list.data?.connectors ?? []).map((c: any) => (
            <tr key={c.id}>
              <td>{c.provider}</td>
              <td><Status value={c.status} /></td>
              <td>{c.account_email ?? "—"}</td>
              <td><code>{c.id}</code></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
