import type { client } from "@/lib/api";

type RetainItem = Parameters<typeof client.retain>[0]["items"][number];

/**
 * The retain item for saving edited document content. The server replaces the
 * document, so the item restates what the original retain supplied (stored in
 * `retain_params`); anything left out re-extracts under bank defaults.
 */
export function editedRetainItem(
  content: string,
  document: { id: string; tags?: string[] | null; retain_params?: Record<string, any> | null }
): RetainItem {
  const retainParams = document.retain_params ?? {};
  const item: RetainItem = { content, document_id: document.id };
  if (retainParams.context) item.context = retainParams.context;
  // A stored `event_date: null` means the document was retained undated; a
  // missing timestamp would let the server date the edit "now" instead (#5473).
  if (retainParams.event_date === null) item.timestamp = "unset";
  else if (retainParams.event_date) item.timestamp = retainParams.event_date;
  if (retainParams.metadata && Object.keys(retainParams.metadata).length > 0) {
    item.metadata = retainParams.metadata;
  }
  if (document.tags && document.tags.length > 0) item.tags = document.tags;
  if (retainParams.strategy) item.strategy = retainParams.strategy;
  if (retainParams.entities?.length) {
    item.entities = retainParams.entities;
    if (retainParams.resolve_entities !== undefined) {
      item.resolve_entities = retainParams.resolve_entities;
    }
  }
  if (retainParams.observation_scopes != null) {
    item.observation_scopes = retainParams.observation_scopes;
  }
  return item;
}
