"use client";

import { api } from "@/lib/api";
import { Status, useAsync, WorkspaceBar } from "@/lib/ui";
import { useWorkspace } from "@/lib/useWorkspace";

export default function Audit() {
  const [wid] = useWorkspace();
  const audit = useAsync(
    () => (wid ? api.auditEvents(wid) : Promise.resolve({ events: [] })),
    [wid],
  );

  return (
    <div>
      <h2>Audit Log</h2>
      <WorkspaceBar />
      {!wid && <p className="muted">Select a workspace first.</p>}
      <button className="secondary" onClick={audit.reload}>Refresh</button>
      {audit.error && <p className="error">{audit.error}</p>}
      <table>
        <thead>
          <tr><th>When</th><th>Action</th><th>Source</th><th>File</th><th>Status</th></tr>
        </thead>
        <tbody>
          {(audit.data?.events ?? []).map((e: any) => (
            <tr key={e.id}>
              <td className="muted">{e.created_at}</td>
              <td><code>{e.action}</code></td>
              <td>{e.source ?? "—"}</td>
              <td className="muted">{e.source_file_id ?? "—"}</td>
              <td><Status value={e.status} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
