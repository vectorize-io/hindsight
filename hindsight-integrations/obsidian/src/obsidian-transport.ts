/**
 * Obsidian-renderer {@link Transport}: wraps `requestUrl`, which (unlike `fetch`)
 * runs outside the renderer's CORS sandbox. Plugin-only — this is the sole place
 * the client path touches `obsidian`.
 */

import { requestUrl } from "obsidian";
import type { Transport } from "./transport";

/** Parse a body as JSON, returning undefined for empty or non-JSON responses. */
function tryParseJson(text: string): unknown {
  if (!text) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

export const obsidianTransport: Transport = async (req) => {
  const resp = await requestUrl({
    url: req.url,
    method: req.method,
    headers: req.headers,
    body: req.body,
    // Handle non-2xx in the client so it can surface a useful message.
    throw: false,
  });
  const text = resp.text ?? "";
  return { status: resp.status, text, json: tryParseJson(text) };
};
