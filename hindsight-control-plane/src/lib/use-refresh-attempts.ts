"use client";

import { useCallback, useEffect, useState } from "react";
import { client } from "./api";

/** A refresh that has not finished: waiting for its turn, or running now. */
export type RefreshAttempt = {
  /** True while the worker runs it; false while it waits in the queue. */
  running: boolean;
  /**
   * True once an earlier attempt of this refresh failed. A refresh waiting out
   * its minimum interval, or on its first attempt, is not a retry (#6864).
   */
  retry: boolean;
  /** When a waiting refresh is due to start, or null when unknown or running. */
  nextAttemptAt: string | null;
};

const REFRESH_OPERATION_TYPE = "refresh_mental_model";
const IN_FLIGHT = new Set(["pending", "processing"]);

/**
 * The refreshes a bank has in flight, by mental-model id.
 *
 * A failed refresh is stamped on the model at the *first* failed attempt, while
 * the worker still has retries left (`HINDSIGHT_API_WORKER_MAX_RETRIES`, 60s
 * apart by default). Without this the UI called that model paused for the four
 * minutes it was still trying — true in the end, wrong at the time (#4532).
 *
 * Read from the operations list the bank already exposes rather than a new field
 * on the mental model: an operation is the only place the *next attempt time*
 * exists, and one request per view covers every model on screen.
 */
export function useRefreshAttempts(bankId: string | null, pollMs?: number) {
  const [attempts, setAttempts] = useState<Map<string, RefreshAttempt>>(new Map());

  const load = useCallback(async () => {
    if (!bankId) {
      setAttempts(new Map());
      return;
    }
    try {
      const result = await client.listOperations(bankId, {
        type: REFRESH_OPERATION_TYPE,
        limit: 100,
      });
      const next = new Map<string, RefreshAttempt>();
      for (const op of result.operations) {
        if (!op.mental_model_id || !IN_FLIGHT.has(op.status)) continue;
        const running = op.status === "processing";
        next.set(op.mental_model_id, {
          running,
          // Either signal alone misses a case: a task put back after a worker
          // restart has a count but no error text.
          retry: Boolean(op.retry_count) || Boolean(op.error_message),
          // next_retry_at is not cleared when the task starts, so it is only a
          // schedule while the task still waits.
          nextAttemptAt: running ? null : (op.next_retry_at ?? null),
        });
      }
      setAttempts(next);
    } catch {
      // Best-effort decoration: a model still renders its own state without this.
    }
  }, [bankId]);

  useEffect(() => {
    load();
    if (!pollMs) return;
    const id = setInterval(load, pollMs);
    return () => clearInterval(id);
  }, [load, pollMs]);

  return attempts;
}
