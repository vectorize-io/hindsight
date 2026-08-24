import type { TagsMatch } from "./api";

/**
 * Whether a page's tag matching lets untagged memories through.
 *
 * A page's tags are the scope it is synthesized from, not labels on it. An unset
 * `tags_match` resolves server-side to `all_strict` for a tagged page — a memory
 * must carry EVERY tag, and untagged memories are excluded outright — so tags
 * chosen to describe a topic build an empty page unless the bank's memories were
 * retained with those exact tags (#3687). Only `all` and `any` include untagged
 * memories.
 */
export function includesUntagged(tagsMatch: TagsMatch | null | undefined): boolean {
  return tagsMatch === "all" || tagsMatch === "any";
}

/**
 * The `tags_match` to send when creating a page, or undefined to send none.
 *
 * Only meaningful alongside tags: an untagged page already matches the whole
 * bank, so pinning a mode on it would store a setting for a filter that does not
 * exist. Left unset, the page keeps the server default.
 */
export function createTagsMatch(tags: string[], includeUntagged: boolean): TagsMatch | undefined {
  if (!tags.length || !includeUntagged) return undefined;
  return "all";
}

/**
 * The `tags_match` to send when saving an edited page, or undefined to send none.
 *
 * Sending nothing when the checkbox was not moved is what keeps this two-state
 * control from flattening the modes it cannot express: a page on `any`,
 * `any_strict` or `exact` must survive a rename untouched. Clearing the tags in
 * the same edit also sends nothing — there is no filter left to widen.
 */
export function updateTagsMatch(
  tags: string[],
  includeUntagged: boolean,
  initialIncludeUntagged: boolean
): TagsMatch | undefined {
  if (!tags.length || includeUntagged === initialIncludeUntagged) return undefined;
  return includeUntagged ? "all" : "all_strict";
}
