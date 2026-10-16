import { client } from "@/lib/api";

/**
 * Last graph read per query, kept across tab switches so the Memories tabs
 * open on what they showed last instead of a spinner. On a big bank one read
 * is several MB and seconds of server time, so it also dedupes reads already
 * in flight (a prefetch the user then clicks into is awaited, not repeated).
 */

export type GraphParams = Parameters<typeof client.getGraph>[0];

type Entry = { data: unknown; at: number };

// Younger than this, a cached read is shown as-is; older, it is shown and
// refreshed in the background.
export const GRAPH_FRESH_MS = 30_000;
// Each entry is a parsed multi-MB response; keep a handful, drop the oldest.
const MAX_ENTRIES = 9;

const results = new Map<string, Entry>();
const inflight = new Map<string, Promise<unknown>>();

function graphKey(p: GraphParams): string {
  return JSON.stringify([
    p.bank_id,
    p.type ?? "",
    p.limit ?? 0,
    p.q ?? "",
    p.tags ?? [],
    p.tags_match ?? "",
    p.document_id ?? "",
    p.chunk_id ?? "",
  ]);
}

export function cachedGraph(p: GraphParams): Entry | undefined {
  return results.get(graphKey(p));
}

export function isFresh(entry: Entry, now = Date.now()): boolean {
  return now - entry.at < GRAPH_FRESH_MS;
}

export function storeGraph(p: GraphParams, data: unknown, now = Date.now()): void {
  const key = graphKey(p);
  results.delete(key);
  results.set(key, { data, at: now });
  while (results.size > MAX_ENTRIES) {
    results.delete(results.keys().next().value as string);
  }
}

/** Read the graph, joining a read of the same query that is already running. */
export function fetchGraph(p: GraphParams, signal?: AbortSignal): Promise<unknown> {
  const running = inflight.get(graphKey(p));
  if (running) return running;
  return client.getGraph(p, signal);
}

/**
 * Warm the cache for a query nobody is looking at yet. Skipped when there is
 * anything cached (a stale entry is refreshed when its tab opens) or a read is
 * already running. Not abortable: the whole point is that it outlives the tab
 * that started it.
 */
export function prefetchGraph(p: GraphParams): Promise<void> {
  const key = graphKey(p);
  if (results.has(key) || inflight.has(key)) return Promise.resolve();
  const read = client
    .getGraph(p)
    .then((data) => {
      storeGraph(p, data);
      return data;
    })
    .finally(() => inflight.delete(key));
  inflight.set(key, read);
  return read.then(
    () => undefined,
    () => undefined
  );
}

type GraphShape = {
  total_units?: number;
  edges?: unknown[];
  table_rows?: { id: string; text?: string }[];
};

/** Same memories, same text, same edge count: nothing on screen would change. */
export function sameGraph(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  const x = a as GraphShape | null;
  const y = b as GraphShape | null;
  if (!x || !y) return false;
  if (x.total_units !== y.total_units) return false;
  if (x.edges?.length !== y.edges?.length) return false;
  const rx = x.table_rows ?? [];
  const ry = y.table_rows ?? [];
  if (rx.length !== ry.length) return false;
  return rx.every((r, i) => r.id === ry[i].id && r.text === ry[i].text);
}
