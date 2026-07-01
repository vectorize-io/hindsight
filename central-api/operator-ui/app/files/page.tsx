"use client";

import { api } from "@/lib/api";
import { Status, useAsync, WorkspaceBar } from "@/lib/ui";
import { useWorkspace } from "@/lib/useWorkspace";

export default function Files() {
  const [wid] = useWorkspace();
  const files = useAsync(
    () => (wid ? api.gdriveFiles(wid) : Promise.resolve({ files: [] })),
    [wid],
  );

  const disable = async (docId: string) => {
    await api.disableDocument(docId, wid);
    files.reload();
  };

  return (
    <div>
      <h2>Indexed Files</h2>
      <WorkspaceBar />
      {!wid && <p className="muted">Select a workspace first.</p>}
      {files.error && <p className="error">{files.error}</p>}
      <table>
        <thead>
          <tr><th>Name</th><th>MIME</th><th>Sync</th><th>Enabled</th><th>Drive ID</th><th></th></tr>
        </thead>
        <tbody>
          {(files.data?.files ?? []).map((f: any) => (
            <tr key={f.id}>
              <td>{f.name}</td>
              <td className="muted">{(f.mime_type ?? "").replace("application/vnd.google-apps.", "gapp:")}</td>
              <td><Status value={f.sync_status} /></td>
              <td>{f.enabled ? "✓" : <span className="badge bad">disabled</span>}</td>
              <td><code>{f.external_id}</code></td>
              <td>
                {f.enabled && (
                  <button className="danger" onClick={() => disable(f.id)}>Disable</button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
