"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { Status, useAsync, WorkspaceBar } from "@/lib/ui";
import { useWorkspace } from "@/lib/useWorkspace";

export default function Ingestion() {
  const [wid] = useWorkspace();
  const [msg, setMsg] = useState<string | null>(null);
  const jobs = useAsync(
    () => (wid ? api.ingestionJobs(wid) : Promise.resolve({ jobs: [] })),
    [wid],
  );

  const process = async () => {
    setMsg(null);
    try {
      const r = await api.processIngestion(wid);
      setMsg(`Processed ${r.processed} job(s)`);
      jobs.reload();
    } catch (e: any) {
      setMsg(`Error: ${e?.message ?? e}`);
    }
  };

  return (
    <div>
      <h2>Ingestion Queue</h2>
      <WorkspaceBar />
      {!wid && <p className="muted">Select a workspace first.</p>}
      <div className="row">
        <button disabled={!wid} onClick={process}>Process pending</button>
        <button className="secondary" onClick={jobs.reload}>Refresh</button>
        {msg && <span className="muted">{msg}</span>}
      </div>
      {jobs.error && <p className="error">{jobs.error}</p>}
      <table>
        <thead>
          <tr><th>Drive ID</th><th>Operation</th><th>Status</th><th>Error</th></tr>
        </thead>
        <tbody>
          {(jobs.data?.jobs ?? []).map((j: any) => (
            <tr key={j.id}>
              <td><code>{j.external_id}</code></td>
              <td>{j.operation}</td>
              <td><Status value={j.status} /></td>
              <td className="error">{j.error ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
