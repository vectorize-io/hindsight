/**
 * Hindsight Memory Client
 * Singleton wrapper around @vectorize-io/hindsight-client
 * Provides typed helpers for retain / recall / reflect operations.
 */

import { HindsightClient } from "@vectorize-io/hindsight-client";

let _client: HindsightClient | null = null;

export function getHindsightClient(): HindsightClient {
  if (!_client) {
    const apiKey = process.env.HINDSIGHT_API_KEY;
    const baseUrl =
      process.env.HINDSIGHT_BASE_URL ?? "https://api.hindsight.vectorize.io";

    if (!apiKey) {
      throw new Error(
        "HINDSIGHT_API_KEY is not set. Add it to .env.local to enable memory features."
      );
    }

    _client = new HindsightClient({ baseUrl, apiKey });
  }
  return _client;
}

/** Bank ID is per-user. For local dev we use DEFAULT_USER_ID. */
export function getBankId(userId?: string): string {
  return (
    userId ??
    process.env.DEFAULT_USER_ID ??
    "reelcraft-user-default"
  );
}
