/**
 * Pure rendering for the sync status-bar item. Kept free of Obsidian/DOM so the
 * label/tooltip logic is unit-testable; `main.ts` owns the actual status-bar
 * element and feeds it this state. The goal (customer feedback): background
 * sync must never happen invisibly — there is always a glanceable indicator.
 */

export interface SyncStatus {
  /** Whether an API URL is configured (no URL → nothing can sync). */
  configured: boolean;
  /** Number of sync operations currently in flight (reconcile/flush/ingest). */
  syncing: number;
  /** Notes edited but not yet flushed to Hindsight. */
  pending: number;
  /** Total notes currently tracked in the local sync index. */
  synced: number;
  /** Epoch ms of the last successful sync, or null if none yet. */
  lastSyncAt: number | null;
  /** Whether the most recent sync attempt failed. */
  error: boolean;
}

/** "1 note" / "412 notes". */
function notes(n: number): string {
  return `${n} note${n === 1 ? "" : "s"}`;
}

export interface SyncStatusView {
  text: string;
  tooltip: string;
}

/** Human-friendly "x ago" for a past timestamp. */
export function relativeTime(thenMs: number, nowMs: number): string {
  const sec = Math.max(0, Math.round((nowMs - thenMs) / 1000));
  if (sec < 45) return "just now";
  const min = Math.round(sec / 60);
  if (min < 60) return `${Math.max(1, min)}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.round(hr / 24);
  return `${day}d ago`;
}

/**
 * Compute a sync indicator's label + tooltip from the current state.
 *
 * `brand` prefixes the label (default "Hindsight" for the status bar, where the
 * item stands alone). Pass "" for the chat header, which already shows the
 * Hindsight wordmark next to it.
 */
export function renderSyncStatus(
  s: SyncStatus,
  nowMs: number,
  brand = "Hindsight"
): SyncStatusView {
  const p = brand ? `${brand} ` : "";
  const pendingNote = s.pending > 0 ? ` · ${s.pending} pending` : "";

  if (!s.configured) {
    return {
      text: `${p}⚙ set API URL`,
      tooltip: "Configure an API URL in settings to enable vault sync.",
    };
  }
  if (s.syncing > 0) {
    return {
      text: `${p}⟳ syncing…${s.pending > 0 ? ` (${s.pending})` : ""}`,
      tooltip: `Syncing your vault to Hindsight… ${notes(s.synced)} tracked${pendingNote}.`,
    };
  }
  if (s.error) {
    return {
      text: `${p}⚠ sync failed`,
      tooltip: `Last sync failed — see the developer console. ${notes(s.synced)} tracked. Click to retry.`,
    };
  }
  if (s.lastSyncAt !== null) {
    const rel = relativeTime(s.lastSyncAt, nowMs);
    const tail = s.pending > 0 ? `${s.pending} pending` : rel;
    return {
      text: `${p}✓ ${notes(s.synced)} · ${tail}`,
      tooltip: `${notes(s.synced)} synced · last synced ${rel}${pendingNote}. Click to sync now.`,
    };
  }
  return {
    text: `${p}✓ not synced yet`,
    tooltip: "No sync yet. Click to sync your vault now.",
  };
}
