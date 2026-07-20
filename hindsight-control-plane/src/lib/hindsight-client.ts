/**
 * Shared Hindsight API client instance for the control plane.
 * Configured to connect to the dataplane API server.
 */

import {
  HindsightClient,
  HindsightError,
  createClient,
  createConfig,
  sdk,
} from "@vectorize-io/hindsight-client";
import type { NextRequest } from "next/server";

import { getProviderDataplaneHeaders } from "@/lib/auth/provider";

export const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";

function getDataplaneApiKey(): string {
  return process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";
}

/**
 * Auth headers for direct fetch calls to the dataplane API.
 */
export function getDataplaneHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const apiKey = getDataplaneApiKey();
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }
  return headers;
}

export function getDataplaneHeadersForRequest(
  request: NextRequest | Request,
  extra?: Record<string, string>
): Record<string, string> {
  return getProviderDataplaneHeaders(request, extra);
}

export function createDataplaneClientForRequest(request: NextRequest | Request) {
  return createClient(
    createConfig({
      baseUrl: DATAPLANE_URL,
      headers: getDataplaneHeadersForRequest(request),
    })
  );
}

export function createHindsightClientForRequest(request: NextRequest | Request): HindsightClient {
  return new HindsightClient({
    baseUrl: DATAPLANE_URL,
    headers: getDataplaneHeadersForRequest(request),
  });
}

/**
 * Build a dataplane URL for a bank-scoped endpoint with the bank id properly encoded.
 * Bank ids may contain `:`, `/`, `%`, etc. (e.g. openclaw `agent::channel::user`),
 * which must be percent-encoded before being interpolated into a URL path.
 */
export function dataplaneBankUrl(bankId: string, suffix = ""): string {
  return `${DATAPLANE_URL}/v1/default/banks/${encodeURIComponent(bankId)}${suffix}`;
}

/**
 * High-level client with convenience methods
 */
export const hindsightClient = new HindsightClient({
  baseUrl: DATAPLANE_URL,
  apiKey: getDataplaneApiKey() || undefined,
});

/**
 * Low-level client for direct SDK access
 */
export const lowLevelClient = createClient(
  createConfig({
    baseUrl: DATAPLANE_URL,
    headers: getDataplaneApiKey() ? { Authorization: `Bearer ${getDataplaneApiKey()}` } : undefined,
  })
);

/**
 * Export SDK functions for direct API access
 */
export { sdk };

/**
 * Export HindsightError for error handling
 */
export { HindsightError };
