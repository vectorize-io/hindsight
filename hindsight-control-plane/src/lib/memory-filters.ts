/**
 * Helpers for describing the memory rows a view is currently showing.
 *
 * These deliberately take the filters that were *applied* to the data on
 * screen, not the ones the user is currently editing: the text search only
 * reaches the server on Enter, so the two disagree for as long as someone is
 * typing or backspacing.
 */

/** The filters that produced the memory rows currently on screen. */
export interface AppliedMemoryFilters {
  /** Free-text query sent as `q`. */
  q: string;
  /** Tag filter sent as `tags`. */
  tags: string[];
  /** `tags_match` mode; only a selected observation scope sends "exact". */
  tagsMatch?: string;
}

/**
 * Whether the rows on screen are a filtered subset of the bank.
 *
 * A selected observation scope is the only filter that sends
 * `tags_match=exact`, which is how it stays detectable even when the scope
 * itself is the empty (global) tag set.
 */
export function hasActiveMemoryFilters(applied: AppliedMemoryFilters): boolean {
  return applied.q.trim().length > 0 || applied.tags.length > 0 || applied.tagsMatch === "exact";
}

/**
 * Whether to show the "no memories yet" screen, which replaces the entire view
 * — search box included — with an invitation to add a document.
 *
 * It must only fire for a genuinely empty bank. A search that matched nothing
 * still has to render the filter bar, otherwise clearing the query is
 * impossible without leaving the page (issue #3670).
 */
export function shouldShowEmptyBankState(
  totalUnits: number | undefined,
  applied: AppliedMemoryFilters
): boolean {
  return totalUnits === 0 && !hasActiveMemoryFilters(applied);
}
