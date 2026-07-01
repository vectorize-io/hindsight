"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/ui";
import { useWorkspace } from "@/lib/useWorkspace";

export default function Workspaces() {
  const [active, setActive] = useWorkspace();
  const [name, setName] = useState("");
  const list = useAsync(() => api.listWorkspaces(), []);

  const create = async () => {
    if (!name.trim()) return;
    await api.createWorkspace(name.trim());
    setName("");
    list.reload();
  };

  return (
    <div>
      <h2>Workspaces</h2>
      <div className="card">
        <div className="row">
          <input placeholder="New workspace name" value={name}
                 onChange={(e) => setName(e.target.value)} />
          <button onClick={create}>Create</button>
          <button className="secondary" onClick={list.reload}>Refresh</button>
        </div>
      </div>
      {list.error && <p className="error">{list.error}</p>}
      <table>
        <thead>
          <tr><th>Name</th><th>ID</th><th>Active</th><th></th></tr>
        </thead>
        <tbody>
          {(list.data?.workspaces ?? []).map((w: any) => (
            <tr key={w.id}>
              <td>{w.name}</td>
              <td><code>{w.id}</code></td>
              <td>{active === w.id ? "✓" : ""}</td>
              <td><button className="secondary" onClick={() => setActive(w.id)}>Set active</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
