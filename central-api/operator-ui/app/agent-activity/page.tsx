"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { Status, useAsync, WorkspaceBar } from "@/lib/ui";
import { useWorkspace } from "@/lib/useWorkspace";

export default function AgentActivity() {
  const [wid] = useWorkspace();
  const activity = useAsync(
    () => (wid ? api.agentActivity(wid) : Promise.resolve({ activity: [] })),
    [wid],
  );
  const [agent, setAgent] = useState("agent-1");
  const [action, setAction] = useState("bulk_sync");
  const [msg, setMsg] = useState<string | null>(null);

  const request = async () => {
    setMsg(null);
    try {
      const r = await fetch(`${api.base}/agent-activity`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_id: agent, action, workspace_id: wid }),
      }).then((x) => x.json());
      setMsg(`${r.decision} — ${r.reason}`);
      activity.reload();
    } catch (e: any) {
      setMsg(`Error: ${e?.message ?? e}`);
    }
  };

  return (
    <div>
      <h2>Agent Activity</h2>
      <WorkspaceBar />
      {!wid && <p className="muted">Select a workspace first.</p>}
      <div className="card">
        <h3>Simulate an agent action request</h3>
        <div className="row">
          <input value={agent} onChange={(e) => setAgent(e.target.value)} />
          <select value={action} onChange={(e) => setAction(e.target.value)}>
            <option>bulk_sync</option>
            <option>connector_disconnect</option>
            <option>search_governed_documents</option>
            <option>list_workspaces</option>
            <option>rm_rf_everything</option>
          </select>
          <button disabled={!wid} onClick={request}>Request</button>
          {msg && <span className="muted">{msg}</span>}
        </div>
        <p className="muted">High-impact actions return <code>requires_approval</code>; unknown actions are denied.</p>
      </div>
      <table>
        <thead>
          <tr><th>When</th><th>Agent</th><th>Action</th><th>Decision</th><th>Reason</th></tr>
        </thead>
        <tbody>
          {(activity.data?.activity ?? []).map((a: any) => (
            <tr key={a.id}>
              <td className="muted">{a.created_at}</td>
              <td>{a.agent_id}</td>
              <td><code>{a.requested_action}</code></td>
              <td><Status value={a.decision} /></td>
              <td className="muted">{a.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
