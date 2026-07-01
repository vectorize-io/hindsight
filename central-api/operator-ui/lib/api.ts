// The ONLY network surface for the Operator GUI. Every call targets the
// CollabMind Central API. The GUI must never call Google Drive, Qdrant, or the
// database directly — that is the whole point of the control plane.

const BASE = process.env.NEXT_PUBLIC_CENTRAL_API_URL ?? "http://localhost:8000";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text}` : ""}`);
  }
  return res.json() as Promise<T>;
}

const qs = (params: Record<string, string | undefined>) => {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v) p.set(k, v);
  const s = p.toString();
  return s ? `?${s}` : "";
};

export const api = {
  base: BASE,
  me: () => req<any>("/me"),

  listWorkspaces: () => req<{ workspaces: any[] }>("/workspaces"),
  createWorkspace: (name: string) =>
    req<any>("/workspaces", { method: "POST", body: JSON.stringify({ name }) }),

  listConnectors: (workspaceId: string) =>
    req<{ connectors: any[] }>(`/connectors${qs({ workspace_id: workspaceId })}`),

  gdriveOauthConfig: () => req<any>("/connectors/google-drive/oauth-config"),
  gdriveStatus: (workspaceId: string) =>
    req<any>(`/connectors/google-drive/status${qs({ workspace_id: workspaceId })}`),
  gdriveConnect: (body: {
    workspace_id: string;
    folder_ids: string[];
    account_email?: string;
    seed?: unknown;
  }) => req<any>("/connectors/google-drive/connect", { method: "POST", body: JSON.stringify(body) }),
  gdriveDisconnect: (workspaceId: string) =>
    req<any>("/connectors/google-drive/disconnect", {
      method: "POST",
      body: JSON.stringify({ workspace_id: workspaceId }),
    }),
  gdriveSync: (workspaceId: string) =>
    req<any>("/connectors/google-drive/sync", {
      method: "POST",
      body: JSON.stringify({ workspace_id: workspaceId }),
    }),
  gdriveFiles: (workspaceId: string) =>
    req<{ files: any[] }>(`/connectors/google-drive/files${qs({ workspace_id: workspaceId })}`),
  gdriveFilePermissions: (documentId: string, workspaceId: string) =>
    req<{ permissions: any[] }>(
      `/connectors/google-drive/files/${documentId}/permissions${qs({ workspace_id: workspaceId })}`,
    ),

  sourceDocuments: (workspaceId: string) =>
    req<{ documents: any[] }>(`/source-documents${qs({ workspace_id: workspaceId })}`),
  disableDocument: (documentId: string, workspaceId: string) =>
    req<any>(`/source-documents/${documentId}/disable${qs({ workspace_id: workspaceId })}`, {
      method: "POST",
    }),

  ingestionJobs: (workspaceId: string, status?: string) =>
    req<{ jobs: any[] }>(`/ingestion-jobs${qs({ workspace_id: workspaceId, status })}`),
  processIngestion: (workspaceId: string) =>
    req<any>("/ingestion/process", {
      method: "POST",
      body: JSON.stringify({ workspace_id: workspaceId }),
    }),

  auditEvents: (workspaceId: string) =>
    req<{ events: any[] }>(`/audit-events${qs({ workspace_id: workspaceId })}`),
  agentActivity: (workspaceId: string) =>
    req<{ activity: any[] }>(`/agent-activity${qs({ workspace_id: workspaceId })}`),
};
