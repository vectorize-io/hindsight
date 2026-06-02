/**
 * Helpers for extracting which notes a reflect answer actually drew on.
 *
 * reflect's `based_on.memories` do NOT carry document_ids, but the recall/expand
 * tool outputs do (including nested `source_facts` for observations). So the
 * authoritative "which notes were retrieved" list comes from walking the trace.
 */
import type { ReflectResponse } from "./types";

/** Recursively collect distinct `document_id` strings from any value. */
export function collectDocIds(value: unknown, acc: Set<string> = new Set()): Set<string> {
  if (Array.isArray(value)) {
    for (const v of value) collectDocIds(v, acc);
  } else if (value && typeof value === "object") {
    for (const [key, v] of Object.entries(value)) {
      if (key === "document_id" && typeof v === "string" && v) acc.add(v);
      else collectDocIds(v, acc);
    }
  }
  return acc;
}

/** Sorted, de-duped note ids retrieved across a reflect response (tools + based_on). */
export function retrievedNotes(response: ReflectResponse): string[] {
  const ids = new Set<string>();
  for (const call of response.trace?.tool_calls ?? []) collectDocIds(call.output, ids);
  for (const m of response.based_on?.memories ?? []) {
    if (m.document_id) ids.add(m.document_id);
  }
  return [...ids].sort();
}
