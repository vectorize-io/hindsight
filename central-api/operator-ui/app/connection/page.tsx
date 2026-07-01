"use client";

import { useEffect, useState } from "react";
import {
  createOperatorApiClient,
  type ApiMode,
  type ConnectionConfig,
  type DependencyHealthResponse,
  type MemorySearchResponse,
  type PolicyCheckResponse,
  type ServiceHealthResponse,
} from "@/lib/operatorApiClient";

type ProbeState = "idle" | "ok" | "degraded" | "unreachable" | "not_configured";

type ProbeRow = {
  name: string;
  state: ProbeState;
  status: number | null;
  detail: string;
};

type StoredConnection = {
  apiBaseUrl: string;
  mode: ApiMode;
  apiKey: string;
  jwtToken: string;
  connected: boolean;
  lastCheckedAt: string;
  lastError: string;
};

const STORAGE_KEY = "cm_operator_connection_v1";
const DEFAULT_BASE_URL = process.env.NEXT_PUBLIC_CENTRAL_API_URL ?? "http://localhost:8000";
const DEFAULT_QUERY = "operator connection smoke";
const DEFAULT_CONTENT = "connection smoke content";

function emptyProbeRows(): ProbeRow[] {
  return [
    { name: "Health", state: "idle", status: null, detail: "" },
    { name: "Dependency health", state: "idle", status: null, detail: "" },
    { name: "Auth validation", state: "idle", status: null, detail: "" },
    { name: "Memory search", state: "idle", status: null, detail: "" },
    { name: "Audit events", state: "idle", status: null, detail: "" },
    { name: "Governance policy", state: "idle", status: null, detail: "" },
    { name: "Execution ledger", state: "idle", status: null, detail: "" },
    { name: "Adapter status", state: "idle", status: null, detail: "" },
    { name: "Service health", state: "idle", status: null, detail: "" },
  ];
}

function probeLabel(state: ProbeState): string {
  switch (state) {
    case "ok":
      return "ok";
    case "degraded":
      return "degraded";
    case "unreachable":
      return "unreachable";
    case "not_configured":
      return "not configured";
    default:
      return "idle";
  }
}

function normalizeStatus(result: { ok: boolean; status: number }): ProbeState {
  if (!result.ok) return result.status === 404 || result.status === 405 ? "not_configured" : result.status === 0 ? "unreachable" : "degraded";
  return "ok";
}

function summarizeError(error: unknown): string {
  if (typeof error === "string") return error;
  if (error && typeof error === "object") {
    const record = error as Record<string, unknown>;
    const message = record.message;
    if (typeof message === "string") return message;
    const detail = record.details;
    if (typeof detail === "string") return detail;
  }
  return "request failed";
}

export default function ConnectionPage() {
  const [ready, setReady] = useState(false);
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_BASE_URL);
  const [mode, setMode] = useState<ApiMode>("remote");
  const [apiKeyDraft, setApiKeyDraft] = useState("");
  const [jwtDraft, setJwtDraft] = useState("");
  const [storedApiKey, setStoredApiKey] = useState("");
  const [storedJwtToken, setStoredJwtToken] = useState("");
  const [probeQuery, setProbeQuery] = useState(DEFAULT_QUERY);
  const [probeContent, setProbeContent] = useState(DEFAULT_CONTENT);
  const [connected, setConnected] = useState(false);
  const [lastCheckedAt, setLastCheckedAt] = useState("");
  const [lastError, setLastError] = useState("");
  const [rows, setRows] = useState<ProbeRow[]>(emptyProbeRows());
  const [running, setRunning] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        setReady(true);
        return;
      }

      const stored = JSON.parse(raw) as Partial<StoredConnection>;
      if (typeof stored.apiBaseUrl === "string" && stored.apiBaseUrl.trim()) {
        setApiBaseUrl(stored.apiBaseUrl);
      }
      if (stored.mode === "local" || stored.mode === "remote") {
        setMode(stored.mode);
      }
      if (typeof stored.apiKey === "string") setStoredApiKey(stored.apiKey);
      if (typeof stored.jwtToken === "string") setStoredJwtToken(stored.jwtToken);
      if (typeof stored.connected === "boolean") setConnected(stored.connected);
      if (typeof stored.lastCheckedAt === "string") setLastCheckedAt(stored.lastCheckedAt);
      if (typeof stored.lastError === "string") setLastError(stored.lastError);
    } catch {
      // ignore corrupt local storage and fall back to defaults
    } finally {
      setReady(true);
    }
  }, []);

  const connectionConfig: ConnectionConfig = {
    apiBaseUrl,
    apiKey: storedApiKey || undefined,
    jwtToken: storedJwtToken || undefined,
    mode,
    connected,
    lastCheckedAt: lastCheckedAt || undefined,
    lastError: lastError || undefined,
  };

  const safeNormalizeBaseUrl = (): string | null => {
    try {
      return createOperatorApiClient({ apiBaseUrl, mode, connected: false }).connection.apiBaseUrl;
    } catch {
      return null;
    }
  };

  const persist = (next: StoredConnection) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  const saveConfig = () => {
    const normalized = safeNormalizeBaseUrl();
    if (!normalized) {
      setLastError("API base URL is invalid");
      return;
    }
    const next = {
      apiBaseUrl: normalized,
      mode,
      apiKey: apiKeyDraft || storedApiKey,
      jwtToken: jwtDraft || storedJwtToken,
      connected,
      lastCheckedAt,
      lastError,
    };
    setStoredApiKey(next.apiKey);
    setStoredJwtToken(next.jwtToken);
    setApiKeyDraft("");
    setJwtDraft("");
    persist(next);
  };

  const clearSecrets = () => {
    setStoredApiKey("");
    setStoredJwtToken("");
    setApiKeyDraft("");
    setJwtDraft("");
    persist({
      apiBaseUrl: apiBaseUrl.trim(),
      mode,
      apiKey: "",
      jwtToken: "",
      connected,
      lastCheckedAt,
      lastError,
    });
  };

  const testConnection = async () => {
    setRunning(true);
    const normalized = safeNormalizeBaseUrl();
    if (!normalized) {
      const message = "API base URL is invalid";
      setConnected(false);
      setLastError(message);
      setLastCheckedAt(new Date().toISOString());
      setRows((current) => current.map((row) => ({ ...row, state: "unreachable", detail: message })));
      setRunning(false);
      return;
    }
    const effectiveConfig = {
      ...connectionConfig,
      apiBaseUrl: normalized,
      apiKey: apiKeyDraft || storedApiKey || undefined,
      jwtToken: jwtDraft || storedJwtToken || undefined,
    };
    const client = createOperatorApiClient(effectiveConfig);

    const nextRows = emptyProbeRows();
    const run = async <T,>(
      index: number,
      label: string,
      promise: Promise<{ ok: true; data: T; status: number } | { ok: false; error: unknown; status: number }>,
      format: (data: T, status: number) => string,
    ) => {
      const result = await promise;
      if (result.ok) {
        nextRows[index] = { name: label, state: "ok", status: result.status, detail: format(result.data, result.status) };
      } else {
        const state = normalizeStatus(result);
        nextRows[index] = { name: label, state, status: result.status, detail: summarizeError(result.error) };
      }
      return result;
    };

    try {
      const health = await run(0, "Health", client.health(), (data) => `${data.service ?? "central-api"} ${data.version ?? ""}`.trim());
      const deps = await run(1, "Dependency health", client.healthDependencies(), (data) => `${Object.keys(data.services).length} services · ${data.governance_healthy ? "governance ok" : "governance degraded"}`);
      const auth = await run(2, "Auth validation", client.validateAuth(), (data) => `auth ${data.auth_method ?? "unknown"}`);
      const memory = await run(3, "Memory search", client.memorySearch({ query: probeQuery, limit: 5 }), (data: MemorySearchResponse) => `${data.count} result(s)`);
      const audit = await run(4, "Audit events", client.auditEvents(10), (data) => `${data.count} event(s)`);
      const policy = await run(5, "Governance policy", client.governancePolicyCheck({ content: probeContent }), (data: PolicyCheckResponse) => `${data.reason}${data.quarantine_id ? ` · quarantine ${data.quarantine_id}` : ""}`);
      const ledger = await run(6, "Execution ledger", client.executionLedger(undefined, 10, 0), (data) => `${data.executions.length} record(s)`);
      const adapters = await run(7, "Adapter status", client.adapterStatus(), (data) => `${data.length} adapter(s)`);
      const services = await run(8, "Service health", client.serviceHealth(), (data: ServiceHealthResponse) => `${data.overall_status ?? "unknown"}`);

      const failures = [health, deps, auth, memory, audit, policy, ledger, adapters, services].filter((result) => !result.ok);
      const criticalFailure = !health.ok || !auth.ok ? summarizeError(!health.ok ? health.error : auth.error) : "";
      const overallConnected = health.ok && auth.ok && failures.length === 0;
      const overallDegraded = !overallConnected && health.ok;

      setRows(nextRows);
      setConnected(overallConnected);
      setLastCheckedAt(new Date().toISOString());
      setLastError(criticalFailure || (failures.length ? summarizeError((failures[0] as { ok: false; error: unknown }).error) : ""));
      persist({
        apiBaseUrl: normalized,
        mode,
        apiKey: storedApiKey,
        jwtToken: storedJwtToken,
        connected: overallConnected,
        lastCheckedAt: new Date().toISOString(),
        lastError: criticalFailure || (failures.length ? summarizeError((failures[0] as { ok: false; error: unknown }).error) : ""),
      });

      if (overallDegraded) {
        setConnected(false);
      }
    } catch (error) {
      const message = summarizeError(error);
      setRows(nextRows.map((row) => row.state === "idle" ? { ...row, state: "unreachable", detail: message } : row));
      setConnected(false);
      setLastCheckedAt(new Date().toISOString());
      setLastError(message);
      persist({
        apiBaseUrl: normalized,
        mode,
        apiKey: storedApiKey,
        jwtToken: storedJwtToken,
        connected: false,
        lastCheckedAt: new Date().toISOString(),
        lastError: message,
      });
    } finally {
      setRunning(false);
    }
  };

  const status = connected ? "connected" : lastError ? "degraded" : ready ? "unreachable" : "idle";

  return (
    <>
      <div className="topbar">
        <span className="topbar-title">Connection</span>
        <span className={`badge ${status === "connected" ? "badge-ok" : status === "degraded" ? "badge-err" : "badge-dim"}`}>
          {status}
        </span>
      </div>
      <div className="page">
        <h1>API Connection</h1>
        <p className="dim" style={{ marginBottom: 20 }}>
          Save the API base URL and credential locally, then test the live control plane without inventing data.
        </p>

        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-label" style={{ marginBottom: 12 }}>Connection</div>
          <div className="field">
            <label htmlFor="apiBaseUrl">API base URL</label>
            <input
              id="apiBaseUrl"
              value={apiBaseUrl}
              onChange={(e) => setApiBaseUrl(e.target.value)}
              placeholder="https://api.example.com"
            />
          </div>
          <div className="field">
            <label htmlFor="mode">Mode</label>
            <select id="mode" value={mode} onChange={(e) => setMode(e.target.value as ApiMode)}>
              <option value="remote">remote</option>
              <option value="local">local</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="apiKey">API key</label>
            <input
              id="apiKey"
              type="password"
              value={apiKeyDraft}
              onChange={(e) => setApiKeyDraft(e.target.value)}
              placeholder={storedApiKey ? "saved locally" : "paste API key"}
              autoComplete="off"
            />
            {storedApiKey && !apiKeyDraft && <span className="dim">API key saved locally</span>}
          </div>
          <div className="field">
            <label htmlFor="jwtToken">JWT token</label>
            <input
              id="jwtToken"
              type="password"
              value={jwtDraft}
              onChange={(e) => setJwtDraft(e.target.value)}
              placeholder={storedJwtToken ? "saved locally" : "paste JWT token"}
              autoComplete="off"
            />
            {storedJwtToken && !jwtDraft && <span className="dim">JWT saved locally</span>}
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <button onClick={saveConfig} disabled={!ready}>Save</button>
            <button className="secondary" onClick={testConnection} disabled={!ready || running}>
              {running ? "Testing…" : "Test connection"}
            </button>
            <button className="secondary" onClick={clearSecrets} disabled={!ready}>
              Clear secrets
            </button>
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <span className="badge badge-dim">last checked: {lastCheckedAt || "—"}</span>
            <span className="badge badge-dim">state: {probeLabel(status as ProbeState)}</span>
          </div>
          {lastError && <div className="error" style={{ marginTop: 12 }}>{lastError}</div>}
        </div>

        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-label" style={{ marginBottom: 12 }}>Probe inputs</div>
          <div className="field">
            <label htmlFor="probeQuery">Memory search query</label>
            <input id="probeQuery" value={probeQuery} onChange={(e) => setProbeQuery(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="probeContent">Governance content</label>
            <input id="probeContent" value={probeContent} onChange={(e) => setProbeContent(e.target.value)} />
          </div>
        </div>

        <div className="card">
          <div className="card-label" style={{ marginBottom: 12 }}>Probe results</div>
          <table>
            <thead>
              <tr>
                <th>Probe</th>
                <th>Status</th>
                <th>HTTP</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.name}>
                  <td>{row.name}</td>
                  <td>
                    <span className={`badge ${row.state === "ok" ? "badge-ok" : row.state === "degraded" ? "badge-err" : row.state === "not_configured" ? "badge-dim" : "badge-dim"}`}>
                      {probeLabel(row.state)}
                    </span>
                  </td>
                  <td className="dim">{row.status ?? "—"}</td>
                  <td className="dim">{row.detail || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
