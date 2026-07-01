"use client";

import { useCallback, useEffect, useState } from "react";
import { useWorkspace } from "./useWorkspace";

export function WorkspaceBar() {
  const [wid] = useWorkspace();
  return (
    <div className="banner">
      Active workspace: {wid ? <code>{wid}</code> : <em>none selected — pick one on Workspaces</em>}
      . The GUI calls only the Central API; it never touches Google Drive or the vector DB directly.
    </div>
  );
}

// Tiny data-loading hook with manual refresh.
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
} {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const run = useCallback(() => {
    setLoading(true);
    setError(null);
    fn()
      .then(setData)
      .catch((e) => setError(String(e?.message ?? e)))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(run, [run]);
  return { data, error, loading, reload: run };
}

export function Status({ value }: { value: string }) {
  const cls =
    ["connected", "indexed", "ok", "allowed", "success"].includes(value)
      ? "ok"
      : ["failed", "denied", "down", "error"].includes(value)
        ? "bad"
        : ["pending", "requires_approval", "skipped", "disconnected"].includes(value)
          ? "warn"
          : "";
  return <span className={`badge ${cls}`}>{value}</span>;
}
