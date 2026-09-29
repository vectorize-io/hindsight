"use client";

import { useCallback, useEffect, useState } from "react";
import { client } from "./api";

/** A refresh attempt that has not finished: running now, or waiting out its retry gap. */
export type RefreshAttempt = {
  /** When the worker will pick it up again, or null when it is running now. */
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
        next.set(op.mental_model_id, { nextAttemptAt: op.next_retry_at ?? null });
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
