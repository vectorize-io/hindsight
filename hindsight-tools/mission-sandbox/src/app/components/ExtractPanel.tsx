"use client";

import { useState } from "react";

interface PreviewFact {
  text: string;
  factType: string;
  occurredStart: string | null;
  occurredEnd: string | null;
  entities: string[];
}

/**
 * Dry-run extraction preview: paste text + an optional mission, see what the retain step would
 * extract — with no ingestion, no persistence. Backed by the /memories/extract API.
 */
export function ExtractPanel({
  project,
  defaultMission,
}: {
  project: string;
  defaultMission: string | null;
}) {
  const [content, setContent] = useState("");
  const [mission, setMission] = useState(defaultMission ?? "");
  const [facts, setFacts] = useState<PreviewFact[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    setFacts(null);
    try {
      const res = await fetch("/api/extract", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ project, content, retainMission: mission || null }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setFacts(data.facts as PreviewFact[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <details className="mt-5 rounded-lg border border-[var(--border)] p-4">
      <summary className="cursor-pointer text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
        Dry-run extraction (preview a mission, no ingest)
      </summary>

      <label className="mt-3 block text-xs uppercase tracking-wider text-[var(--muted)]">
        text
      </label>
      <textarea
        className="mt-1 h-28 w-full rounded-md border border-[var(--border)] bg-[var(--surface-2)] p-2 text-sm"
        placeholder="Paste a document / chunk to extract facts from…"
        value={content}
        onChange={(e) => setContent(e.target.value)}
      />

      <label className="mt-3 block text-xs uppercase tracking-wider text-[var(--muted)]">
        retain mission (override — defaults to the project&apos;s current mission)
      </label>
      <textarea
        className="mt-1 h-20 w-full rounded-md border border-[var(--border)] bg-[var(--surface-2)] p-2 text-xs"
        value={mission}
        onChange={(e) => setMission(e.target.value)}
      />

      <button
        className="mt-3 rounded-md border border-[var(--accent)] px-3 py-1.5 text-sm text-[var(--accent)] disabled:opacity-50"
        onClick={run}
        disabled={loading || !content.trim()}
      >
        {loading ? "Extracting…" : "Extract (dry-run)"}
      </button>

      {error ? <p className="mt-2 text-sm text-[var(--bad)]">{error}</p> : null}

      {facts ? (
        <div className="mt-3">
          <div className="text-xs uppercase tracking-wider text-[var(--muted)]">
            {facts.length} fact{facts.length === 1 ? "" : "s"} extracted
          </div>
          <ul className="mt-1 space-y-1">
            {facts.map((f, i) => (
              <li key={i} className="border-l-2 border-[var(--border)] pl-3 text-sm">
                {f.text}
                <span className="ml-1 text-xs text-[var(--muted)]">
                  [{f.factType}
                  {f.occurredStart ? ` · ${f.occurredStart.slice(0, 10)}` : ""}
                  {f.entities.length ? ` · ${f.entities.join(", ")}` : ""}]
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </details>
  );
}
