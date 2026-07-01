"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { useAsync, WorkspaceBar } from "@/lib/ui";
import { useWorkspace } from "@/lib/useWorkspace";

export default function Permissions() {
  const [wid] = useWorkspace();
  const [selected, setSelected] = useState<string>("");
  const files = useAsync(
    () => (wid ? api.gdriveFiles(wid) : Promise.resolve({ files: [] })),
    [wid],
  );
  const perms = useAsync(
    () =>
      selected && wid
        ? api.gdriveFilePermissions(selected, wid)
        : Promise.resolve({ permissions: [] }),
    [selected, wid],
  );

  return (
    <div>
      <h2>Permissions View</h2>
      <WorkspaceBar />
      {!wid && <p className="muted">Select a workspace first.</p>}
      <div className="banner">
        Retrieval is permission-checked against these Drive snapshots and fails closed: no snapshot,
        disabled doc, or inactive connector → access denied.
      </div>
      <div className="row">
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          <option value="">— pick a file —</option>
          {(files.data?.files ?? []).map((f: any) => (
            <option key={f.id} value={f.id}>{f.name}</option>
          ))}
        </select>
      </div>
      {perms.error && <p className="error">{perms.error}</p>}
      <table>
        <thead>
          <tr><th>Type</th><th>Role</th><th>Email</th><th>Domain</th><th>Discoverable</th></tr>
        </thead>
        <tbody>
          {(perms.data?.permissions ?? []).map((p: any) => (
            <tr key={p.id}>
              <td>{p.ptype}</td>
              <td>{p.role}</td>
              <td>{p.email_address ?? "—"}</td>
              <td>{p.domain ?? "—"}</td>
              <td>{String(p.allow_file_discovery ?? "—")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
