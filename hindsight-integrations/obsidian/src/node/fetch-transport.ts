/**
 * Node {@link Transport}: a `fetch`-based implementation for the headless CLI.
 * Outside Obsidian's renderer there is no CORS sandbox, so plain `fetch`
 * (global since Node 18) is all we need.
 */

import type { Transport } from "../transport";

function tryParseJson(text: string): unknown {
  if (!text) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

export const fetchTransport: Transport = async (req) => {
  const resp = await fetch(req.url, {
    method: req.method,
    headers: req.headers,
    body: req.body,
  });
  const text = await resp.text();
  return { status: resp.status, text, json: tryParseJson(text) };
};
