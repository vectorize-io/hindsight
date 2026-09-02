/**
 * HTTP transport seam for {@link HindsightClient}.
 *
 * The plugin runs inside Obsidian's renderer and must use `requestUrl` to escape
 * the CORS sandbox; a headless CLI runs in Node and uses `fetch`. Keeping the
 * client transport-agnostic lets both frontends share one client (and therefore
 * one set of request semantics) instead of maintaining divergent copies.
 */

export interface TransportRequest {
  url: string;
  method: string;
  headers: Record<string, string>;
  /** JSON-encoded request body, omitted for GET/DELETE. */
  body?: string;
}

export interface TransportResponse {
  status: number;
  /** Raw response body; used to build error messages on non-2xx. */
  text: string;
  /** Parsed body when it is valid JSON, otherwise undefined. */
  json: unknown;
}

export type Transport = (req: TransportRequest) => Promise<TransportResponse>;
